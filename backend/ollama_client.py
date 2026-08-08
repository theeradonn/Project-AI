"""
SafeTrade — Ollama Client Module
เชื่อมต่อกับ Ollama API (qwen3:8b) เพื่อวิเคราะห์ความเสี่ยงในการโกงซื้อขายออนไลน์

ฟังก์ชันหลัก:
- analyze_risk() — ส่งบทสนทนาไปวิเคราะห์ที่ Ollama แล้วรับผล JSON กลับมา
- build_prompt() — สร้าง prompt จากประวัติแชท
- parse_risk_response() — แยกวิเคราะห์ JSON พร้อม error handling หลายชั้น

สถาปัตยกรรมการให้คะแนน:
โมเดลมีหน้าที่แค่ "ตรวจจับ pattern" จากข้อความของผู้ขาย + คัดลอก base_score ที่กำหนดไว้แล้วมาใส่ output
ห้ามให้โมเดลคำนวณ/บวกเลขเอง — risk_percentage คำนวณแบบ deterministic ที่ฝั่ง Python
(ผลรวม base_score ของทุก pattern ที่ตรวจพบ ไม่ซ้ำ) เพราะโมเดลขนาดเล็กทำเลขในหัวไม่แม่นยำพอ
"""

import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)

# ==========================================
# Configuration
# ==========================================
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:8b"
REQUEST_TIMEOUT = 120.0  # วินาที — qwen3:8b อาจใช้เวลาในการ generate


# ==========================================
# Pattern Catalogue — Single Source of Truth
# ==========================================
# base_score ของแต่ละ pattern เป็นค่าคงที่ที่กำหนดไว้ล่วงหน้า
# ใช้ dict นี้ทั้งสร้างข้อความใน SYSTEM_PROMPT และตรวจสอบ/คำนวณคะแนนฝั่ง Python
# เพื่อไม่ให้ prompt กับ logic การคำนวณคะแนนหลุดไม่ตรงกัน (single source of truth)
PATTERN_CATALOGUE = {
    "A1": {
        "group": "A",
        "label": "ไม่รับคนกลาง/ไม่ยอมให้ตรวจสินค้าก่อนจ่าย",
        "base_score": 40,
        "definition": "ผู้ขายปฏิเสธเมื่อผู้ซื้อเสนอให้มีคนกลางช่วยตรวจสอบ หรือปฏิเสธการให้ตรวจสอบสินค้าก่อนโอนเงินเต็มจำนวน",
        "positive_example": '"ไม่เอาคนกลางนะ ยุ่งยาก" / "ต้องโอนก่อนถึงจะส่งของให้ตรวจได้"',
        "negative_example": None,
    },
    "A2": {
        "group": "A",
        "label": "ขอให้โอนเข้าบัญชีชื่อไม่ตรงกับโปรไฟล์ผู้ขาย",
        "base_score": 35,
        "definition": "ผู้ขายให้เลขบัญชีที่ชื่อเจ้าของบัญชีไม่ตรงกับชื่อโปรไฟล์หรือชื่อที่คุยกันมาตลอด และไม่อธิบายเหตุผลชัดเจน",
        "positive_example": '"โอนเข้าชื่อ [ชื่ออื่น] แทนนะ บัญชีตัวเองมีปัญหา"',
        "negative_example": '"โอนเข้าชื่อร้าน [ชื่อร้าน] นะคะ" (ถ้าชื่อร้านตรงกับที่โฆษณาไว้ตั้งแต่ต้น)',
    },
    "A3": {
        "group": "A",
        "label": "ผลักดันให้ย้ายช่องทางคุยออกจากแพลตฟอร์มเดิม",
        "base_score": 35,
        "definition": "ผู้ขายชวนให้ไปคุยต่อที่ Line, เบอร์โทร, หรือแอปอื่น ก่อนที่การซื้อขายจะเสร็จสมบูรณ์",
        "positive_example": '"แอดไลน์คุยต่อนะ 0891234567" / "ทักมาทาง Line จะได้คุยง่ายกว่า"',
        "negative_example": None,
    },
    "B1": {
        "group": "B",
        "label": "ราคาต่ำผิดปกติตอนแรก แล้วบวกเพิ่มภายหลังโดยไม่สมเหตุผล",
        "base_score": 20,
        "definition": "ราคาที่ตกลงกันตอนแรกเปลี่ยนแปลงเพิ่มขึ้นระหว่างคุย โดยอ้างเหตุผลที่ฟังดูไม่สมเหตุผลหรือไม่ชัดเจน",
        "positive_example": 'ตอนแรกบอกราคา 500 บาท ต่อมาบอก "ต้องบวกค่าดำเนินการอีก 300 นะ"',
        "negative_example": "แจ้งค่าส่งเพิ่มตั้งแต่แรกอย่างชัดเจนตามน้ำหนัก/ระยะทาง",
    },
    "B2": {
        "group": "B",
        "label": "เลี่ยงหรือปฏิเสธการนัดเจอรับสินค้าโดยไม่มีเหตุผลรองรับ",
        "base_score": 20,
        "definition": "ผู้ขายไม่ยอมนัดเจอ ทั้งที่ผู้ซื้ออยู่ในพื้นที่ใกล้เคียง และ 'ไม่ได้อ้างเหตุผลเรื่องระยะทาง' เลย — ถ้าอ้างว่าอยู่ไกล/ต่างจังหวัด ให้ไปนับเป็น D3 แทน ห้ามนับ B2",
        "positive_example": '"ไม่สะดวกนัดเจอค่ะ ส่งไปรษณีย์อย่างเดียว" (ทั้งที่ผู้ซื้ออยู่จังหวัดเดียวกัน ไม่บอกเหตุผลว่าทำไม)',
        "negative_example": '"พอดีอยู่เชียงใหม่ครับ ส่ง EMS สะดวกกว่า" (อ้างระยะทางชัดเจน = D3 ไม่ใช่ B2)',
    },
    "B3": {
        "group": "B",
        "label": "ปฏิเสธการเก็บเงินปลายทาง (COD)",
        "base_score": 20,
        "definition": "ผู้ขายปฏิเสธการเก็บเงินปลายทาง/COD โดยเฉพาะ (มีคำว่า ปลายทาง, COD, เก็บเงินปลายทาง) จุดเด่นคือเรื่อง 'ช่องทางจ่ายเงิน' ไม่เกี่ยวกับเวลา/ความเร็วในการตัดสินใจ",
        "positive_example": '"ไม่รับเก็บปลายทางนะคะ" / "ไม่มีปลายทางครับ โอนก่อนเท่านั้น" (ปฏิเสธ COD ล้วนๆ = B3 แม้ไม่มีการเร่งเวลา) / "ไม่มีปลายทาง"/ "ไม่รับปลายทาง"',
        "negative_example": None,
    },
    "B4": { 
        "group": "B",
        "label": "เร่งรัดให้โอนเงินโดยเร็วผิดปกติ",
        "base_score": 20,
        "definition": "ผู้ขายกดดันเรื่อง 'เวลา' ให้รีบตัดสินใจ/รีบโอนภายในเวลาสั้นๆ ต้องมีคำที่สื่อถึงความเร่งด่วนหรือการจำกัดเวลาชัดเจน เช่น รีบ, ด่วน, เดี๋ยวนี้, อีกกี่นาที, มีคนรอคิว, เดี๋ยวของหมด — ถ้าไม่มีคำเร่งเวลาเหล่านี้ ห้ามนับเป็น B4",
        "positive_example": '"รีบโอนเลยนะ อีก 10 นาทีให้คนอื่นแล้ว" / "มีคนรอคิวโอนอยู่ รีบหน่อยนะ"',
        "negative_example": '"ไม่รับเก็บปลายทางนะคะ" (นี่คือ B3 ปฏิเสธ COD ไม่ใช่การเร่งเวลา) / "ตอนนี้ 2000 ถ้ารับเลยเหลือ 1500" (นี่คือ C4 เสนอส่วนลดถ้าซื้อทันที ไม่ใช่การเร่งเวลา)',
    },
    "C1": {
        "group": "C",
        "label": "เปลี่ยนราคาหรือเงื่อนไขกะทันหันระหว่างบทสนทนา",
        "base_score": 15,
        "definition": "เงื่อนไขการซื้อขาย (ไม่ใช่แค่ราคา) เปลี่ยนกลางคันโดยไม่มีเหตุผลรองรับ เช่น จากส่งฟรีเป็นเก็บค่าส่ง",
        "positive_example": None,
        "negative_example": None,
    },
    "C2": {
        "group": "C",
        "label": "หลีกเลี่ยงคำถามเกี่ยวกับสภาพหรือรายละเอียดสินค้าด้วยเหตุผลต่างๆ",
        "base_score": 10,
        "definition": "ผู้ซื้อถามรายละเอียดเฉพาะเจาะจง (เช่น ตำหนิ เลขซีเรียล วันหมดอายุ สภาพแบตเตอรี่) แต่ผู้ขายไม่ตอบตรงคำถาม แล้วยกเหตุผลมาเลี่ยงแทน",
        "positive_example": 'ผู้ซื้อ: "แบตเสื่อมกี่ % ครับ" / ผู้ขาย: "ของแพ็คไว้แล้วครับ เช็คให้ไม่ได้"',
        "negative_example": "ตอบตรงคำถาม เช่น \"แบต 87% ครับ\" หรือบอกว่าไม่รู้แต่ยินดีไปเช็คให้ = ไม่ใช่ C2",
    },
    "C3": {
        "group": "C",
        "label": "อ้างเหตุผลเร่งด่วนส่วนตัวเพื่อบีบให้ตัดสินใจเร็ว",
        "base_score": 10,
        "definition": "ผู้ขายเล่าเรื่องราวส่วนตัวเพื่อสร้างความเห็นใจและกดดันให้ซื้อ/โอนเร็วขึ้น",
        "positive_example": '"จะย้ายบ้านพรุ่งนี้แล้ว ต้องรีบขายวันนี้เลย"',
        "negative_example": None,
    },
    "C4": {
        "group": "C",
        "label": "เสนอส่วนลดพิเศษถ้าโอนเงินก่อนหรือโอนเต็มจำนวนทันที",
        "base_score": 15,
        "definition": "ผู้ขายเสนอส่วนลด/ลดราคาเป็นแรงจูงใจให้โอนเงินก่อนหรือรับของทันที จุดเด่นคือ 'ลดราคาแลกกับการรีบซื้อ' (ไม่ใช่การกดดันเรื่องเวลาแบบ B4)",
        "positive_example": '"โอนเต็มจำนวนวันนี้ลดให้อีก 100 นะ" / "ตอนนี้ 2000 ถ้ารับเลยเหลือ 1500 ครับ"',
        "negative_example": None,
    },
    "D1": {
        "group": "D",
        "label": "ปฏิเสธการส่งรูปเพิ่มเติมตามที่ผู้ซื้อขอ",
        "base_score": 10,
        "definition": "ผู้ขายปฏิเสธการส่งรูปเพิ่มเติมตามที่ผู้ซื้อขอโดยอ้างเหตุผลต่างๆ — จะนับได้ก็ต่อเมื่อผู้ซื้อ 'ขอรูปเพิ่มอย่างชัดเจน' แล้วผู้ขายปฏิเสธ ถ้าผู้ซื้อไม่ได้ขอรูป ห้ามนับ D1 เด็ดขาด",
        "positive_example": 'ผู้ซื้อ: "ขอรูปเพิ่มอีกมุมได้ไหมครับ" / ผู้ขาย: "รูปในโพสต์ก็ชัดแล้วครับ ถ่ายเพิ่มไม่ได้"',
        "negative_example": 'ผู้ขายตอบสั้นเรื่องอื่น เช่น "ไม่มีครับ" (ตอบเรื่องปลายทาง ไม่เกี่ยวกับรูป) = ไม่ใช่ D1',
    },
    "D2": {
        "group": "D",
        "label": "ใช้ภาษาเป็นทางการหรือซ้ำแพทเทิร์นคล้ายสคริปต์ผิดธรรมชาติ",
        "base_score": 10,
        "definition": "จะนับได้ก็ต่อเมื่อผู้ขายพิมพ์ข้อความ 'ยาวและเป็นทางการผิดปกติ' หรือมีโครงสร้างซ้ำแบบก๊อปวางเหมือนสคริปต์บอททุกข้อความ — ข้อความแชทสั้นๆ ปกติ (เช่น 'ได้ครับ', 'ไม่มีครับ', 'นัดรับได้นะ') ไม่ใช่ D2 เด็ดขาด",
        "positive_example": "ทุกข้อความขึ้นต้นด้วยประโยคทางการเหมือนกันเป๊ะ เช่น 'เรียนลูกค้าผู้มีอุปการคุณ ทางร้านขอเรียนแจ้งว่า...' ซ้ำทุกครั้ง",
        "negative_example": "ภาษาพูดวัยรุ่นทั่วไป/ตอบสั้นๆ = ไม่ใช่ D2",
    },
    "D3": {
        "group": "D",
        "label": "ไม่นัดเจอ ขอส่งผ่านขนส่งแทน โดยอ้างว่าอยู่ไกล/ต่างจังหวัด",
        "base_score": 5,
        "definition": "ผู้ขายไม่นัดเจอรับสินค้า แต่ 'มีการอ้างเหตุผลเรื่องระยะทาง' (เช่น อยู่ต่างจังหวัด อยู่ไกล คนละภาค) แล้วเสนอส่งผ่านขนส่งแทน (EMS, Flash, Kerry, J&T, ไปรษณีย์) — ต่างจาก B2 ตรงที่ 'มีเหตุผลรองรับ' จึงเสี่ยงต่ำมาก เป็นพฤติกรรมปกติของผู้ขายต่างจังหวัดทั่วไป",
        "positive_example": '"พอดีผมอยู่เชียงใหม่ครับพี่ สะดวกส่งเป็น EMS หรือ Flash Express มากกว่าครับ" / "อยู่คนละจังหวัดเลยส่งไปรษณีย์นะคะ"',
        "negative_example": '"ไม่สะดวกนัดเจอค่ะ ส่งไปรษณีย์อย่างเดียว" (ไม่ได้อ้างเรื่องระยะทางเลย = B2 ไม่ใช่ D3)',
    },
}

_GROUP_LABELS = {
    "A": "กลุ่ม A: เสี่ยงสูงมาก",
    "B": "กลุ่ม B: เสี่ยงสูง",
    "C": "กลุ่ม C: เสี่ยงปานกลาง",
    "D": "กลุ่ม D: เสี่ยงต่ำ",
}


def _render_pattern_catalogue() -> str:
    """เรนเดอร์ PATTERN_CATALOGUE เป็นข้อความสำหรับใส่ใน SYSTEM_PROMPT"""
    lines: list[str] = []
    current_group = None
    for pattern_id, info in PATTERN_CATALOGUE.items():
        if info["group"] != current_group:
            current_group = info["group"]
            lines.append(f"\n--- {_GROUP_LABELS[current_group]} ---\n")
        lines.append(f"[{pattern_id}] {info['label']}")
        lines.append(f"base_score: {info['base_score']}")
        lines.append(f"คำจำกัดความ: {info['definition']}")
        if info.get("positive_example"):
            lines.append(f"ตัวอย่างข้อความที่เข้าข่าย: {info['positive_example']}")
        if info.get("negative_example"):
            lines.append(f"ตัวอย่างข้อความที่ไม่เข้าข่าย: {info['negative_example']}")
        lines.append("")
    return "\n".join(lines).strip()


# ==========================================
# System Prompt ภาษาไทย สำหรับ qwen3:8b
# หมายเหตุ: prompt นี้เขียนสำหรับโมเดลขนาดเล็ก/ความสามารถจำกัด
# เน้นความชัดเจน แบ่งเป็นขั้นตอนย่อย ลด ambiguity ให้มากที่สุด ห้ามให้โมเดลตีความเอง
# ห้ามให้โมเดลคำนวณคะแนนเอง — หน้าที่โมเดลคือ detect pattern + แปะคะแนนที่กำหนดไว้แล้วเท่านั้น
# ==========================================
_SYSTEM_PROMPT_HEADER = """คุณคือระบบตรวจจับพฤติกรรมเสี่ยงโกงในบทสนทนาซื้อขายออนไลน์ภาษาไทย
หน้าที่ของคุณมีอย่างเดียว: อ่านบทสนทนา แล้วบอกว่าข้อความไหนของ "ผู้ขาย" ตรงกับ pattern ที่กำหนดไว้บ้าง
คุณห้ามคิดคะแนนเอง ห้ามบวกเลข ห้ามคูณเลข หน้าที่คุณคือ copy ตัวเลข base_score ที่กำหนดไว้แล้วมาใส่ใน output เท่านั้น
ห้ามสรุปว่า "โกงแน่นอน" ห้ามให้ pattern ที่ไม่อยู่ในรายการด้านล่าง

=====================================================
ขั้นตอนที่ 1: อ่านกฎการนับ pattern ต่อไปนี้ให้ครบทุกข้อ ก่อนเริ่มวิเคราะห์
=====================================================

กฎสำคัญที่สุด: ตรวจจับ pattern จาก "พฤติกรรมของผู้ขาย" เท่านั้น (evidence ที่ใส่ใน output ต้องเป็นข้อความของผู้ขายเสมอ)
ห้ามจับผู้ซื้อเป็นผู้ต้องสงสัย และห้ามใช้ข้อความของผู้ซื้อเป็น evidence
แต่คุณสามารถใช้คำถาม/ข้อความของผู้ซื้อเป็น "บริบท" เพื่อช่วยตีความคำตอบสั้นๆ ของผู้ขายได้
ตัวอย่าง: ผู้ซื้อ: "มีปลายทางไหมครับ" / ผู้ขาย: "ไม่มีครับ" → เข้าใจได้ว่าผู้ขายกำลังปฏิเสธการเก็บเงินปลายทาง (COD) = B3 โดยใช้ข้อความผู้ขาย "ไม่มีครับ" เป็น evidence

กฎที่สอง: 1 pattern ต้องมีข้อความจริงจากบทสนทนามารองรับเสมอ
ถ้าไม่มีข้อความจริงรองรับ ห้ามใส่ pattern นั้นใน output แม้จะรู้สึกว่า "น่าจะใช่"

กฎที่สาม: วิเคราะห์ทีละ pattern ตามลำดับที่กำหนดไว้ด้านล่าง
อย่าข้ามขั้นตอน อย่ารวบตัดสิน ให้เช็คทีละอันจนครบ

กฎที่สี่: base_score ของแต่ละ pattern เป็นตัวเลขคงที่ที่กำหนดไว้แล้ว
หน้าที่คุณคือ copy ตัวเลขนั้นมาใส่ใน output ตรงๆ ห้ามปรับ ห้ามประเมินเอง ห้ามบวกรวมกับ pattern อื่น

=====================================================
ขั้นตอนที่ 2: รายการ pattern ทั้งหมด พร้อม base_score คำอธิบาย และตัวอย่างประกอบ
=====================================================
"""

_SYSTEM_PROMPT_FOOTER = """

=====================================================
ขั้นตอนที่ 3: วิธีเช็คทีละ pattern (ทำตามลำดับนี้เป๊ะๆ)
=====================================================

1. อ่านบทสนทนาทั้งหมดตั้งแต่ต้นจนจบก่อน 1 รอบ
2. เปิดรายการ pattern A1 ถึง D2 ทีละตัว
3. สำหรับแต่ละ pattern ถามตัวเองว่า: "มีข้อความของผู้ขายในบทสนทนานี้ที่ตรงกับคำจำกัดความและตัวอย่างหรือไม่"
4. ถ้าตรง -> คัดลอกข้อความจริงมาใส่เป็น evidence + คัดลอก base_score ของ pattern นั้นมาใส่ตรงๆ (ห้ามคิดเลขใหม่)
5. ถ้าไม่ตรง หรือไม่แน่ใจ -> ข้ามไป ไม่ต้องใส่ใน output
6. ทำจนครบทุก pattern ในหมวด A ถึง D แล้วค่อยสร้าง output

กฎเหล็ก: pattern_id ต้องตรงกับ 'เนื้อหาจริง' ของ evidence เท่านั้น ห้ามแปะ pattern_id ที่ไม่ตรงกับข้อความ
- ผู้ขายพูดว่า "ไม่มีปลายทาง / ไม่รับปลายทาง / ไม่รับเก็บเงินปลายทาง / โอนก่อนเท่านั้น" -> ต้องนับเป็น B3 เสมอ (แม้จะเป็นการตอบคำถามผู้ซื้อก็ตาม) และห้ามนับเป็น B4
- evidence จะเป็น B4 ได้ ก็ต่อเมื่อมีคำเร่งเวลาจริงๆ (รีบ, ด่วน, เดี๋ยวนี้, อีกกี่นาที, มีคนรอคิว, เดี๋ยวของหมด)
- evidence เป็นการลดราคาถ้าซื้อทันที (เช่น "รับเลยเหลือ...") -> เป็น C4 ไม่ใช่ B4
- ผู้ขายไม่นัดเจอ แต่ "อ้างว่าอยู่ไกล/ต่างจังหวัด" แล้วขอส่งขนส่ง (EMS/Flash/Kerry/ไปรษณีย์) -> เป็น D3 (เสี่ยงต่ำ) ห้ามนับเป็น B2
- ผู้ขายไม่นัดเจอ แต่ "ไม่บอกเหตุผลเรื่องระยะทางเลย" -> ถึงจะเป็น B2
- 1 ข้อความ (evidence) ให้เลือก pattern_id ที่ตรงที่สุดเพียงอันเดียว อย่าแปะหลาย pattern ให้ข้อความเดียวกันถ้าไม่ได้เข้าข่ายจริง

=====================================================
ขั้นตอนที่ 4: รูปแบบ Output (บังคับ)
=====================================================

1. คุณต้องตอบกลับเป็น JSON ที่ถูกต้อง (Valid JSON) เท่านั้น ห้ามมีข้อความทักทาย อธิบาย หรือ Markdown Code blocks คร่อมเด็ดขาด ให้เริ่มด้วย {{ และจบด้วย }}
2. โครงสร้าง JSON ต้องมี 2 field เป๊ะๆ คือ "matched_patterns" และ "summary"
3. "matched_patterns" เป็น array ของ pattern ที่ตรวจพบ (ถ้าไม่พบเลยให้ใส่ array ว่าง []) แต่ละ item ต้องมี 3 field เป๊ะๆ:
{{
  "pattern_id": "รหัส pattern เช่น A2",
  "evidence": "ข้อความจริงจากผู้ขายที่ตรงกับ pattern นี้ (คัดลอกมาจากบทสนทนาเป๊ะๆ)",
  "base_score": ตัวเลข base_score ของ pattern นั้น (copy จากรายการ ห้ามคิดใหม่)
}}
4. "summary" เป็นข้อความสั้นๆ 1 ประโยค (ไม่เกิน 20 คำ) สรุปเหตุผลโดยอ้างอิงเฉพาะ pattern ที่อยู่ใน "matched_patterns" เท่านั้น ห้ามพูดถึงพฤติกรรมอื่นที่ไม่ได้ตรวจพบ ถ้า "matched_patterns" ว่างเปล่า ให้เขียนว่า "ไม่พบพฤติกรรมที่น่าสงสัย"

## ตัวอย่าง Output 1 (พบหลาย pattern)
Input: ผู้ซื้อ: "พี่ครับ iPhone 14 สะดวกนัดรับแนว BTS ไหม หรือส่งปลายทางได้ไหมครับ" / ผู้ขาย: "ผมอยู่แม่ฮ่องสอนครับ นัดรับไม่ได้ ปลายทางไม่รับครับเคยโดนตีกลับ โอนยอดเต็มเข้าบัญชีแฟนพี่ได้เลยครับ นางสาวสมศรี รีบหน่อยนะมีคิว 2 รอโอนอยู่"
Output:
{{
  "matched_patterns": [
    {{"pattern_id": "A2", "evidence": "โอนยอดเต็มเข้าบัญชีแฟนพี่ได้เลยครับ นางสาวสมศรี", "base_score": 30}},
    {{"pattern_id": "B3", "evidence": "ปลายทางไม่รับครับเคยโดนตีกลับ", "base_score": 17}},
    {{"pattern_id": "B4", "evidence": "รีบหน่อยนะมีคิว 2 รอโอนอยู่", "base_score": 16}}
  ],
  "summary": "ให้โอนเข้าบัญชีชื่ออื่น ปฏิเสธเก็บเงินปลายทาง และเร่งรัดให้รีบโอน"
}}

## ตัวอย่าง Output 2 (ไม่พบ pattern)
Input: ผู้ซื้อ: "ขอดูรูปกระเป๋าเพิ่มหน่อยครับ รบกวนเขียนป้ายชื่อเฟสพี่วางขนาบไว้ด้วยนะครับ" / ผู้ขาย: "ได้ค่า รอสักครู่นะคะ เดี๋ยวกำลังหยิบมาถ่ายให้ พร้อมส่งวิดีโอตอนแพ็คของให้ดูด้วยค่ะ นัดรับได้นะคะถ้าอยู่ใกล้เซ็นทรัลลาดพร้าว"
Output:
{{
  "matched_patterns": [],
  "summary": "ไม่พบพฤติกรรมที่น่าสงสัย"
}}
"""

SYSTEM_PROMPT = (
    _SYSTEM_PROMPT_HEADER
    + _render_pattern_catalogue()
    + _SYSTEM_PROMPT_FOOTER.format()
)


# ==========================================
# Prompt Builder
# ==========================================
def build_prompt(messages: list[dict]) -> str:
    """
    สร้าง prompt จากประวัติแชทเพื่อส่งให้ Ollama วิเคราะห์

    Args:
        messages: รายการข้อความ [{"sender": "buyer"|"seller", "text": "..."}, ...]

    Returns:
        prompt string ที่จัดรูปแบบแล้ว
    """
    if not messages:
        return "ยังไม่มีบทสนทนา"

    # จัดรูปแบบบทสนทนาให้โมเดลอ่านง่าย
    conversation_lines = []
    for msg in messages:
        role_label = "ผู้ซื้อ" if msg.get("sender") == "buyer" else "ผู้ขาย"
        conversation_lines.append(f"{role_label}: {msg.get('text', '')}")

    conversation_text = "\n".join(conversation_lines)

    # เพิ่ม /no_think เพื่อให้ qwen3 ตอบ JSON โดยตรงโดยไม่มี <think> block
    return (
        f"วิเคราะห์บทสนทนาต่อไปนี้ แล้วตรวจจับ pattern ที่ตรงกับข้อความของผู้ขาย:\n\n"
        f"--- บทสนทนา ---\n"
        f"{conversation_text}\n"
        f"--- จบบทสนทนา ---\n\n"
        f'ตอบเป็น JSON เท่านั้น: {{"matched_patterns": [{{"pattern_id": "...", '
        f'"evidence": "...", "base_score": ตัวเลข}}], "summary": "เหตุผลสั้นๆ 1 ประโยค"}} /no_think'
    )


# ==========================================
# Response Parser (with multi-layer fallback)
# ==========================================
def parse_risk_response(raw_text: str) -> dict:
    """
    แยกวิเคราะห์ผลลัพธ์ JSON จาก Ollama พร้อม Error Handling หลายชั้น

    ลำดับการ parse:
    1. พยายาม parse JSON ตรงๆ
    2. ค้นหา JSON object ใน text (กรณีมีข้อความแถมมา)
    3. Fallback: ไม่พบ JSON เลย ถือว่าวิเคราะห์ไม่สำเร็จ

    Args:
        raw_text: ข้อความดิบจาก Ollama response

    Returns:
        dict: {"risk_percentage": int (0-100), "summary": str,
               "detected_flags": list[str], "scam_pattern": str}
    """
    cleaned = raw_text.strip()

    # === ชั้นที่ 1: Parse JSON โดยตรง ===
    try:
        data = json.loads(cleaned)
        return _validate_risk_data(data)
    except json.JSONDecodeError:
        pass

    # === ชั้นที่ 2: ค้นหา JSON object ใน text ===
    # กรณี Ollama แถมข้อความก่อน/หลัง JSON
    json_pattern = r'\{[^{}]*"matched_patterns"\s*:\s*\[.*?\][^{}]*\}'
    match = re.search(json_pattern, cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return _validate_risk_data(data)
        except json.JSONDecodeError:
            pass

    # === ชั้นที่ 3: ไม่พบ JSON ที่ parse ได้เลย — ไม่มีทางเดาคะแนนจากตัวเลขลอยๆ ได้อีกต่อไป ===
    # (สถาปัตยกรรมใหม่: คะแนนมาจากผลรวม base_score ของ pattern เท่านั้น ไม่ใช่ตัวเลขที่โมเดลเขียนลอยๆ)
    logger.warning(f"⚠️ ไม่สามารถ parse JSON ได้จาก response: {cleaned[:200]}")
    return {
        "risk_percentage": 0,
        "summary": "ไม่สามารถวิเคราะห์ได้ — โมเดลไม่ตอบกลับเป็น JSON ที่ถูกต้อง",
        "detected_flags": [],
        "scam_pattern": "ไม่ระบุ",
    }


def _validate_risk_data(data: dict) -> dict:
    """
    ตรวจสอบและคำนวณ risk_percentage แบบ deterministic จาก matched_patterns

    ขั้นตอน:
    1. อ่าน matched_patterns (list), default [] ถ้าไม่มี/ผิดชนิด
    2. เช็ค pattern_id แต่ละตัวกับ PATTERN_CATALOGUE — ตัวที่ไม่รู้จัก (โมเดลมโนขึ้นมาเอง) จะถูกตัดทิ้ง
    3. ตัด pattern ซ้ำ (นับ pattern_id ละ 1 ครั้งต่อการวิเคราะห์)
    4. ไม่เชื่อ base_score ที่โมเดล copy มา — ใช้ค่าจาก PATTERN_CATALOGUE เป็นหลักเสมอ (log ถ้าไม่ตรงกัน)
    5. risk_percentage = ผลรวม base_score ของทุก pattern ที่ตรวจพบ (ไม่ซ้ำ) clamp 0-100
    6. summary: ใช้ข้อความที่โมเดลเขียนเอง (สั้นๆ อ้างอิง pattern ที่พบ) ถ้าโมเดลไม่ได้ส่งมา/ส่งมาว่าง
       จะ fallback เป็นข้อความที่สร้างจาก label ของ pattern แทน
    """
    raw_matches = data.get("matched_patterns")
    if not isinstance(raw_matches, list):
        raw_matches = []

    matched: dict[str, dict] = {}  # pattern_id -> {label, base_score, evidence}
    for item in raw_matches:
        if not isinstance(item, dict):
            continue

        pattern_id = str(item.get("pattern_id", "")).strip().upper()
        catalogue_entry = PATTERN_CATALOGUE.get(pattern_id)
        if catalogue_entry is None:
            logger.warning(f"⚠️ โมเดลระบุ pattern_id ที่ไม่รู้จัก: {pattern_id!r} — ข้าม")
            continue

        if pattern_id in matched:
            continue  # นับ pattern ซ้ำแค่ครั้งเดียว

        model_base_score = item.get("base_score")
        authoritative_score = catalogue_entry["base_score"]
        if model_base_score is not None and model_base_score != authoritative_score:
            logger.warning(
                f"⚠️ base_score ของ {pattern_id} ที่โมเดลส่งมา ({model_base_score}) "
                f"ไม่ตรงกับค่าจริง ({authoritative_score}) — ใช้ค่าจริงแทน"
            )

        matched[pattern_id] = {
            "label": catalogue_entry["label"],
            "base_score": authoritative_score,
            "evidence": str(item.get("evidence", "")).strip(),
        }

    risk = min(sum(m["base_score"] for m in matched.values()), 100)
    model_summary = str(data.get("summary", "")).strip()

    if matched:
        detected_flags = [
            f"[{pid}] {m['label']} (+{m['base_score']}): {m['evidence']}"
            for pid, m in matched.items()
        ]
        top_pattern_id = max(matched, key=lambda pid: matched[pid]["base_score"])
        scam_pattern = matched[top_pattern_id]["label"]
        summary = model_summary or (
            "พบพฤติกรรมเข้าข่าย: " + ", ".join(m["label"] for m in matched.values())
        )
    else:
        detected_flags = []
        scam_pattern = "ปลอดภัย"
        summary = model_summary or "ไม่พบพฤติกรรมที่น่าสงสัยจากผู้ขายในบทสนทนานี้"

    return {
        "risk_percentage": risk,
        "summary": summary,
        "detected_flags": detected_flags,
        "scam_pattern": scam_pattern,
    }


# ==========================================
# Main Analysis Function
# ==========================================
def analyze_risk(messages: list[dict]) -> dict:
    """
    ฟังก์ชันหลัก — วิเคราะห์ความเสี่ยงจากบทสนทนา

    ขั้นตอน:
    1. สร้าง prompt จากข้อความแชท
    2. ส่งไปที่ Ollama API (qwen3:8b) — โมเดลตรวจจับ pattern เท่านั้น ไม่คำนวณคะแนน
    3. Parse ผลลัพธ์ JSON แล้วคำนวณ risk_percentage แบบ deterministic ที่ฝั่ง Python
    4. คืนค่า risk_percentage + summary + detected_flags + scam_pattern

    Args:
        messages: รายการข้อความจาก Firestore

    Returns:
        dict: {"risk_percentage": int, "summary": str,
               "detected_flags": list[str], "scam_pattern": str}
              risk_percentage = -1 หมายถึงเกิดข้อผิดพลาดในการเชื่อมต่อ
    """
    prompt = build_prompt(messages)

    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # บังคับ Ollama ให้ return JSON format เท่านั้น
        "options": {
            "temperature": 0.3,   # ค่าต่ำ → ผลลัพธ์สม่ำเสมอ, ลด randomness
            "num_predict": 512,   # จำกัดความยาว — เผื่อพื้นที่สำหรับ matched_patterns หลายรายการ
            "top_p": 0.9,
        },
    }

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            logger.info(
                f"📡 กำลังส่งบทสนทนา ({len(messages)} ข้อความ) ไปวิเคราะห์ที่ Ollama..."
            )

            response = client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
            )
            response.raise_for_status()

            result = response.json()
            raw_text = result.get("response", "")
            logger.info(f"📨 ได้รับผลลัพธ์จาก Ollama: {raw_text[:150]}")

            return parse_risk_response(raw_text)

    except httpx.TimeoutException:
        logger.error("⏰ Ollama timeout — ใช้เวลานานเกินไป")
        return {
            "risk_percentage": -1,
            "summary": "ระบบ AI ไม่ตอบกลับ — กรุณาตรวจสอบว่า Ollama กำลังทำงานอยู่",
            "detected_flags": [],
            "scam_pattern": "ไม่ระบุ",
        }

    except httpx.ConnectError:
        logger.error("🔌 ไม่สามารถเชื่อมต่อ Ollama ได้ที่ localhost:11434")
        return {
            "risk_percentage": -1,
            "summary": "ไม่สามารถเชื่อมต่อ AI ได้ — กรุณาเปิด Ollama ที่ localhost:11434",
            "detected_flags": [],
            "scam_pattern": "ไม่ระบุ",
        }

    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")
        return {
            "risk_percentage": -1,
            "summary": f"เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)[:100]}",
            "detected_flags": [],
            "scam_pattern": "ไม่ระบุ",
        }
