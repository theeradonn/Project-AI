"""
SafeTrade — Phase 1 Dataset Generator
สร้างชุดข้อมูลแชทซื้อขายออนไลน์จำลอง (fraud / normal / suspicious) ด้วย Gemini API
เพื่อใช้เป็น dataset สำหรับ blind test และ DPO fine-tuning ในเฟสถัดไป

Schema ต่อ 1 แชท (ตรงกับ database schema ของโปรเจกต์):
{
  "conversation_id": "conv_001",
  "category": "fraud" | "normal" | "suspicious",
  "messages": [
    {
      "turn": 1,
      "sender": "buyer" | "seller",
      "text": "...",
      "risk_after": 0-100,
      "detected_flags": ["A2", ...],   # pattern_id จาก PATTERN_CATALOGUE ที่เพิ่งปรากฏใน turn นี้
      "evidence": [{"pattern_id": "A2", "text": "<วลีที่คัดลอกจาก text เป๊ะๆ>"}],  # วัตถุดิบของ Evidence Bank
      "scam_pattern": "..."            # label ของ pattern ที่ turn นั้นเจอ (base_score สูงสุดใน turn) หรือ null ถ้าไม่มี flag
    }
  ]
}

กฎที่บังคับใช้หลัง generate (ไม่พึ่งพาโมเดลให้ทำตามเอง — สอดคล้องกับ ollama_client.py):
- risk_after ของแต่ละ turn = ผลรวม base_score ของทุก pattern_id ที่ตรวจพบสะสมตั้งแต่ต้นแชทจนถึง turn นั้น
  (นับ pattern_id ซ้ำแค่ครั้งเดียว, clamp 0-100) — คำนวณที่ Python ทั้งหมด ไม่เชื่อตัวเลขจาก Gemini
- risk_after จึงไม่มีทางลดลงจาก turn ก่อนหน้าโดยธรรมชาติของการสะสม (running sum)
- evidence.text ต้องเป็น substring จริงของ text ใน turn นั้น (เทียบแบบตัดช่องว่าง)
  ถ้าโมเดลเรียบเรียงใหม่/มโน จะถูกทิ้งแล้ว fallback ไปใช้ข้อความเต็มของ turn แทน

รันคำสั่ง:
    python generate_dataset.py           # รันต่อจากของเดิมอัตโนมัติ (resume)
    python generate_dataset.py --fresh   # เริ่มใหม่ทั้งหมด ทับไฟล์เดิม

Resume: ถ้ามีไฟล์ output อยู่แล้ว สคริปต์จะนับว่าแต่ละหมวดสร้างไปกี่แชทแล้ว
แล้วสร้างต่อเฉพาะส่วนที่ขาด (หมวดที่ครบแล้วจะถูกข้าม) — สำคัญมากเวลาโดน
rate limit รายวันของ Gemini กลางคัน จะได้ไม่ต้องเสียโควตาสร้างของเดิมซ้ำ

ต้องตั้งค่า GEMINI_API_KEY ใน backend/.env ก่อนรัน (ห้าม hardcode key ในไฟล์นี้)
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from ollama_client import PATTERN_CATALOGUE, _render_pattern_catalogue  # noqa: E402  (ใช้ pattern catalogue เดียวกับระบบจริง)

load_dotenv(BACKEND_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# Configuration
# ==========================================
# รองรับหลาย API key คั่นด้วย comma (GEMINI_API_KEYS) เพื่อหมุนใช้สลับกัน — ถ้าไม่มีจะ fallback ไปใช้ GEMINI_API_KEY เดี่ยว
def _load_api_keys() -> list[str]:
    multi = os.getenv("GEMINI_API_KEYS", "")
    keys = [k.strip() for k in multi.split(",") if k.strip()]
    if not keys:
        single = os.getenv("GEMINI_API_KEY", "").strip()
        if single:
            keys = [single]
    return keys


GEMINI_API_KEYS = _load_api_keys()
# หมายเหตุ: gemini-2.5-flash ถูกตัดสิทธิ์สำหรับบัญชีที่สร้างใหม่แล้ว — ใช้ตัวที่ใหม่กว่าเพื่อให้ทุก key (รวมบัญชีใหม่) ใช้ได้
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "dataset_500v3.json"

BATCH_SIZE = 10          # จำนวนแชทต่อ 1 การเรียก Gemini (ยิ่งมาก = ยิง API น้อยครั้ง = โดน rate limit น้อยลง)
MAX_RETRIES = 2          # จำนวนครั้งที่ retry ต่อ batch หาก JSON ไม่ผ่าน validation
SLEEP_BETWEEN_CALLS = 13  # วินาที — คุมความเร็ว 'รวม' ไว้ที่ ~5 req/นาที (60/5=12 ตั้ง 13 เผื่อ) แม้จะหมุนหลาย key
RATE_LIMIT_MAX_WAITS = 6  # จำนวนครั้งสูงสุดที่ยอมรอเมื่อ 'ทุก key' โดน rate limit พร้อมกัน ต่อ 1 batch ก่อนยอมแพ้
MAX_CONSECUTIVE_FAILURES = 5  # ถ้า batch ล้มเหลวติดกันเกินนี้ = หยุดทั้งหมด (กัน infinite loop จาก error ถาวร)

CATEGORY_TARGETS = {
    "fraud": 200,
    "normal": 200,
    "suspicious": 100,
}

PATTERN_CATALOGUE_TEXT = _render_pattern_catalogue()

CATEGORY_INSTRUCTIONS = {
    "fraud": (
        "สร้างแชท 'มิจฉาชีพ' ที่หลอกลวงชัดเจน ต้องมี pattern จากกลุ่ม A (A1/A2/A3) อย่างน้อย 1 ตัวในแชท "
        "และควรมี pattern จากกลุ่ม B/C ประกอบด้วยเพื่อความสมจริง"
    ),
    "normal": (
        "สร้างแชทซื้อขายปกติที่ปลอดภัย ผู้ขายให้ความร่วมมือตรวจสอบ (ถ่ายรูปคู่ป้ายชื่อ, วิดีโอคอล, "
        "ยอมรับ COD หรือนัดรับได้) ห้ามมี pattern ใดๆ เลยตลอดทั้งแชท (detected_flags ต้องว่างทุก turn)"
    ),
    "suspicious": (
        "สร้างแชทที่มีสัญญาณเตือนจากกลุ่ม B, C หรือ D บ้าง แต่ 'ห้ามมี' pattern จากกลุ่ม A เลย "
        "(เช่น เร่งรัดกดดัน B4, ราคาผิดปกติ B1, บ่ายเบี่ยงเล็กน้อย C2/D1) "
        "สำคัญ: ให้กระจาย pattern ให้หลากหลาย อย่าใช้แต่ B3/B4 ซ้ำๆ — "
        "ต้องมีแชทที่ใช้ C1 (เปลี่ยนเงื่อนไขกลางคัน), C2 (ตอบกำกวมซ้ำๆ), D2 (พิมพ์ทางการเหมือนสคริปต์บอท) "
        "และ D3 (ผู้ขายอ้างว่าอยู่ต่างจังหวัด/อยู่ไกล แล้วขอส่ง EMS/Flash/ไปรษณีย์แทนการนัดเจอ) ปนอยู่ด้วย"
    ),
}

GENERATION_CONFIG = {
    "temperature": 0.9,
    "response_mime_type": "application/json",
}


# ==========================================
# Prompt Builder
# ==========================================
def build_generation_prompt(category: str, batch_size: int, start_index: int) -> str:
    return f"""คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์พฤติกรรมการซื้อขายออนไลน์ในประเทศไทย
จงสร้างบทสนทนาแชทซื้อขายออนไลน์ระหว่าง "ผู้ซื้อ" (buyer) และ "ผู้ขาย" (seller) จำนวน {batch_size} แชท
โดยแต่ละแชทมีความยาว 11-13 turn (ผลัดกันพูดไม่จำเป็นต้องสลับกันทุกครั้ง)

หมวดหมู่ของแชททั้งหมดนี้คือ: "{category}"
คำอธิบายหมวดหมู่: {CATEGORY_INSTRUCTIONS[category]}

อ้างอิงรายการ pattern ต่อไปนี้ (ใช้เฉพาะ pattern_id ในรายการนี้เท่านั้น เช่น "A2", "B4" ห้ามสร้าง pattern_id ใหม่เอง):
{PATTERN_CATALOGUE_TEXT}

[ข้อกำหนดสำคัญ - ต้องทำตามอย่างเคร่งครัด]
1. ภาษาที่ใช้ต้องสมจริง เป็นภาษาวัยรุ่นไทย มีคำสแลง (เช่น เตง, ค้าบ, ป่าว, แฟลช, จ้า) และพิมพ์ผิดบ้างเล็กน้อยเพื่อความสมจริง
2. ความยาวข้อความสั้นยาวสลับกันไป ไม่เป็นแพทเทิร์นหุ่นยนต์
3. pattern ต้องมาจาก "พฤติกรรมของผู้ขาย" เท่านั้น (detected_flags แปะได้เฉพาะ turn ของผู้ขาย) ห้ามจับผู้ซื้อเป็นผู้ต้องสงสัย — แต่ควรออกแบบให้ผู้ซื้อถามนำก่อน แล้วผู้ขายตอบสั้นๆ เพื่อให้ต้องใช้บริบทตีความ (เช่น ผู้ซื้อ: "มีปลายทางไหม" / ผู้ขาย: "ไม่มีครับ" -> turn ผู้ขายนี้เป็น B3) ให้มีเคสแบบนี้ปนอยู่ในชุด fraud/suspicious ด้วย
4. detected_flags ของแต่ละ turn ให้ใส่เฉพาะ pattern_id ที่ "เพิ่งปรากฏใน turn นั้นเป็นครั้งแรก" ในแชท (ไม่ต้องใส่ pattern_id ซ้ำที่เจอไปแล้วใน turn ก่อนหน้า) ถ้าไม่มีให้ใส่ array ว่าง
4.1 turn ไหนที่มี detected_flags ต้องใส่ field "evidence" ด้วย เป็น array ของ object {{"pattern_id": "...", "text": "..."}}
    กฎของ evidence.text (สำคัญมาก):
    - ต้องเป็น "วลีสั้นๆ ที่คัดลอกมาจากข้อความ turn นั้นแบบเป๊ะตัวอักษร" (substring จริง) ห้ามเรียบเรียงใหม่ ห้ามสรุป ห้ามแต่งเพิ่ม
    - เอาเฉพาะ "ช่วงที่เข้าข่าย pattern นั้น" เท่านั้น ห้ามใส่ทั้งข้อความ — ปกติยาวประมาณ 4-15 คำ
    - ถ้า turn เดียวมีหลาย pattern ให้แยก evidence คนละรายการ และ "แต่ละรายการต้องเป็นคนละวลีกัน" (ห้ามใช้ข้อความเดียวกันซ้ำ)
    ตัวอย่าง: ข้อความเต็ม = "ไม่รับเก็บปลายทางนะคะ โอนก่อนเท่านั้นค่า รีบหน่อยนะเดี๋ยวของหมด"
       -> evidence B3 = "ไม่รับเก็บปลายทางนะคะ"   (ไม่ใช่ข้อความเต็ม)
       -> evidence B4 = "รีบหน่อยนะเดี๋ยวของหมด"  (คนละวลีกับ B3)
    - turn ที่ไม่มี flag ให้ใส่ array ว่าง
5. scam_pattern ปล่อยเป็น "ปลอดภัย" ไปก่อนได้ทุก turn — ระบบจะคำนวณให้เองจาก detected_flags ของแต่ละ turn (turn ไหนไม่มี flag จะเป็น null) ไม่ต้องเดาเอง
6. risk_after ปล่อยเป็น 0 ไปก่อนได้ทุก turn — ระบบจะคำนวณค่าจริงให้เองจาก pattern ที่ตรวจพบ ไม่ต้องคำนวณเอง

ตอบกลับเป็น JSON Array เท่านั้น ห้ามมีข้อความอื่นใดนอกเหนือจาก JSON โครงสร้างนี้:
[
  {{
    "conversation_id": "conv_{start_index:03d}",
    "messages": [
      {{
        "turn": 1,
        "sender": "buyer",
        "text": "...",
        "risk_after": 0,
        "detected_flags": [],
        "evidence": [],
        "scam_pattern": "ปลอดภัย"
      }},
      {{
        "turn": 2,
        "sender": "seller",
        "text": "ไม่รับเก็บปลายทางนะคะ โอนก่อนเท่านั้นค่า",
        "risk_after": 0,
        "detected_flags": ["B3"],
        "evidence": [{{"pattern_id": "B3", "text": "ไม่รับเก็บปลายทางนะคะ"}}],
        "scam_pattern": "ปฏิเสธการเก็บเงินปลายทาง (COD)"
      }}
    ]
  }}
]"""


# ==========================================
# Rule Enforcement (ไม่พึ่งพาโมเดลให้ทำตามกฎเอง)
# ==========================================
# สถิติคุณภาพ evidence ของทั้งรอบการ generate (ใช้รายงานตอนจบ)
EVIDENCE_STATS: dict[str, int] = {
    "verbatim": 0,           # เป็นวลีที่คัดลอกมาจริง (ดีที่สุด — Evidence Bank ใช้ได้เต็มประสิทธิภาพ)
    "rejected_not_substring": 0,  # Gemini เรียบเรียงใหม่/มโน → ถูกทิ้ง
    "fallback_full_text": 0,      # ต้องใช้ข้อความเต็มของ turn แทน (คุณภาพด้อยกว่า)
}


def _normalize_for_match(text: str) -> str:
    """ตัดช่องว่างทั้งหมดออกเพื่อเทียบ substring (คนไทย/โมเดลเว้นวรรคไม่แน่นอน)"""
    return "".join(str(text).split())


def _sanitize_evidence(
    raw_evidence: object,
    valid_flags: list[str],
    message_text: str,
) -> list[dict]:
    """
    ทำความสะอาด field evidence ที่ Gemini สร้างมา ให้สอดคล้องกับ detected_flags ที่ผ่านการกรองแล้ว

    กฎ:
    - เก็บเฉพาะ evidence ที่ pattern_id อยู่ใน valid_flags (ตัดของที่ Gemini มโน)
    - evidence.text ต้องเป็น substring จริงของข้อความ turn นั้น (เทียบแบบตัดช่องว่าง)
      ถ้าไม่ใช่ = Gemini เรียบเรียงใหม่/มโนขึ้นมา → ทิ้งแล้ว fallback ไปใช้ข้อความเต็มแทน
      เพราะ Evidence Bank ต้องการ "ข้อความที่ผู้ขายพิมพ์จริง" ไม่ใช่คำสรุปของโมเดล
    - pattern ที่มี flag แต่ไม่มี evidence → fallback ใช้ข้อความเต็มของ turn นั้น
      (กัน Evidence Bank ว่าง และรองรับ dataset เก่าที่ยังไม่มี field evidence)

    Args:
        raw_evidence: ค่า evidence ดิบจาก Gemini (คาดว่าเป็น list ของ dict)
        valid_flags: pattern_id ที่ผ่านการตรวจสอบแล้วของ turn นี้
        message_text: ข้อความเต็มของ turn นี้ (ใช้ตรวจ substring + เป็น fallback)

    Returns:
        list ของ {"pattern_id": str, "text": str}
    """
    if not valid_flags:
        return []

    full_text = str(message_text).strip()
    normalized_full = _normalize_for_match(full_text)

    cleaned: dict[str, str] = {}
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue
            pattern_id = str(item.get("pattern_id", "")).strip().upper()
            text = str(item.get("text", "")).strip()
            if pattern_id not in valid_flags or not text or pattern_id in cleaned:
                continue

            # ตรวจว่าเป็นข้อความที่มาจาก turn นี้จริงหรือไม่
            if normalized_full and _normalize_for_match(text) in normalized_full:
                cleaned[pattern_id] = text
                EVIDENCE_STATS["verbatim"] += 1
            else:
                EVIDENCE_STATS["rejected_not_substring"] += 1  # ปล่อยให้ตกไป fallback ด้านล่าง

    # pattern ที่ยังไม่มี evidence ใช้ได้ (ไม่ได้ให้มา / ถูกทิ้ง) → ใช้ข้อความเต็มของ turn แทน
    for pattern_id in valid_flags:
        if pattern_id not in cleaned and full_text:
            cleaned[pattern_id] = full_text
            EVIDENCE_STATS["fallback_full_text"] += 1

    return [{"pattern_id": pid, "text": text} for pid, text in cleaned.items()]


def enforce_score_rules(messages: list[dict]) -> list[dict]:
    """
    คำนวณ risk_after + scam_pattern แบบ deterministic — ไม่เชื่อค่าที่ Gemini ใส่มา คำนวณใหม่จาก detected_flags:
    - risk_after = ผลรวม base_score ของทุก pattern_id ที่ตรวจพบสะสมตั้งแต่ต้นแชท (นับซ้ำแค่ครั้งเดียว) clamp 0-100
    - scam_pattern = label ของ pattern ที่มี base_score สูงสุด 'ใน turn นั้นเอง' (ตรงกับ detected_flags ของ turn นั้น)
      ถ้า turn ไหนไม่มี flag → None (รวมถึง buyer ทุก turn ที่ detected_flags ว่างเสมอ)
    - evidence = เก็บเฉพาะรายการที่ pattern_id ยังอยู่ใน detected_flags หลังกรอง และมีข้อความจริง
      (ใช้เป็นวัตถุดิบของ Evidence Bank ในระบบ pre-filter)
    """
    seen_pattern_ids: set[str] = set()
    running_total = 0

    for msg in messages:
        raw_flags = msg.get("detected_flags")
        valid_flags = []
        if isinstance(raw_flags, list):
            for flag in raw_flags:
                pattern_id = str(flag).strip().upper()
                if pattern_id not in PATTERN_CATALOGUE:
                    continue  # Gemini มโน pattern_id ที่ไม่มีจริง — ตัดทิ้ง
                valid_flags.append(pattern_id)
                if pattern_id not in seen_pattern_ids:
                    seen_pattern_ids.add(pattern_id)
                    running_total += PATTERN_CATALOGUE[pattern_id]["base_score"]

        msg["detected_flags"] = valid_flags
        msg["evidence"] = _sanitize_evidence(msg.get("evidence"), valid_flags, msg.get("text", ""))
        msg["risk_after"] = min(running_total, 100)
        # scam_pattern = label ของ pattern base_score สูงสุด 'ใน turn นี้' (ไม่มี flag = null)
        if valid_flags:
            top_pid = max(valid_flags, key=lambda pid: PATTERN_CATALOGUE[pid]["base_score"])
            msg["scam_pattern"] = PATTERN_CATALOGUE[top_pid]["label"]
        else:
            msg["scam_pattern"] = None

    return messages


def validate_conversation(conv: dict) -> bool:
    """ตรวจสอบโครงสร้างพื้นฐานก่อนรับเข้า dataset"""
    if not isinstance(conv, dict):
        return False
    messages = conv.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if msg.get("sender") not in ("buyer", "seller"):
            return False
        if not str(msg.get("text", "")).strip():
            return False
        if not isinstance(msg.get("detected_flags"), list):
            msg["detected_flags"] = []
    return True


# ==========================================
# Gemini Batch Generation
# ==========================================
def _is_rate_limit_error(msg: str) -> bool:
    """เช็คว่า error เป็นการโดน rate limit (429 / quota) หรือไม่"""
    low = msg.lower()
    return "429" in msg or "quota" in low or "exhausted" in low or "rate limit" in low


def _extract_retry_delay(msg: str, default: float = 45.0) -> float:
    """ดึงเวลาที่ Gemini บอกให้รอจาก error message (เช่น 'retry in 39.39s' หรือ 'seconds: 39')"""
    m = re.search(r"retry in ([\d.]+)s", msg)
    if m:
        return float(m.group(1)) + 2  # เผื่อ buffer 2 วิ
    m = re.search(r"seconds:\s*(\d+)", msg)
    if m:
        return float(m.group(1)) + 2
    return default


class KeyRotator:
    """
    หมุนใช้ Gemini API key หลายตัวสลับกัน (แต่ละ key = คนละ project = คนละโควตา)
    - current_model(): คืน model ที่ผูกกับ key ปัจจุบัน
    - rotate(): สลับไป key ถัดไปแบบวน แล้วคืน model ตัวใหม่
    """

    def __init__(self, keys: list[str]):
        self.keys = keys
        self.idx = 0

    def current_model(self):
        genai.configure(api_key=self.keys[self.idx])
        return genai.GenerativeModel(GEMINI_MODEL, generation_config=GENERATION_CONFIG)

    def rotate(self):
        self.idx = (self.idx + 1) % len(self.keys)
        return self.current_model()

    @property
    def label(self) -> str:
        return f"key#{self.idx + 1}/{len(self.keys)}"


def generate_batch(rotator: "KeyRotator", category: str, batch_size: int, start_index: int) -> list[dict]:
    prompt = build_generation_prompt(category, batch_size, start_index)
    model = rotator.current_model()

    parse_attempts = 0            # นับเฉพาะความล้มเหลวจาก parse/validation
    consecutive_rate_limits = 0   # นับ key ที่โดน rate limit ติดต่อกัน (ถ้าครบทุก key = รอ)

    while parse_attempts < MAX_RETRIES + 1:
        try:
            response = model.generate_content(prompt)
            data = json.loads(response.text)
            if not isinstance(data, list):
                raise ValueError("ผลลัพธ์ไม่ใช่ JSON array")

            valid_conversations = []
            for conv in data:
                if not validate_conversation(conv):
                    continue
                conv["category"] = category
                conv["messages"] = enforce_score_rules(conv["messages"])
                for i, msg in enumerate(conv["messages"]):
                    msg["turn"] = i + 1
                valid_conversations.append(conv)

            if valid_conversations:
                return valid_conversations

            parse_attempts += 1
            logger.warning(f"⚠️ [{category}] batch ไม่มีแชทที่ผ่าน validation เลย (ครั้งที่ {parse_attempts})")
            time.sleep(SLEEP_BETWEEN_CALLS)

        except (json.JSONDecodeError, ValueError) as e:
            parse_attempts += 1
            logger.warning(f"⚠️ [{category}] parse ล้มเหลว (ครั้งที่ {parse_attempts}): {e}")
            time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            msg = str(e)
            if _is_rate_limit_error(msg):
                consecutive_rate_limits += 1
                # ถ้ามีหลาย key: สลับไป key ถัดไปแล้วยิงต่อทันที (แต่ละ key มีโควตาแยกกัน)
                if len(rotator.keys) > 1 and consecutive_rate_limits < len(rotator.keys):
                    old = rotator.label
                    model = rotator.rotate()
                    logger.warning(
                        f"🔄 [{category}] {old} โดน rate limit — สลับไป {rotator.label} แล้วยิงต่อ"
                    )
                    time.sleep(1)  # หน่วงสั้นๆ พอเป็นพิธี
                    continue
                # ถ้าทุก key โดนพร้อมกัน (หรือมี key เดียว) = รอตามที่ Gemini แจ้ง
                if consecutive_rate_limits > len(rotator.keys) * RATE_LIMIT_MAX_WAITS:
                    logger.error(f"❌ [{category}] ทุก key โดน rate limit ต่อเนื่องเกินไป — ข้าม batch นี้")
                    return []
                delay = _extract_retry_delay(msg)
                logger.warning(
                    f"⏳ [{category}] ทุก key ({len(rotator.keys)} ตัว) โดน rate limit พร้อมกัน — "
                    f"รอ {delay:.0f} วินาทีแล้วลองใหม่"
                )
                time.sleep(delay)
                model = rotator.rotate()  # เริ่มรอบใหม่จาก key ถัดไป
                # ไม่เพิ่ม parse_attempts เพราะ rate limit ไม่ใช่ความผิดของ prompt
            else:
                parse_attempts += 1
                logger.error(f"❌ [{category}] เกิดข้อผิดพลาดจาก Gemini (ครั้งที่ {parse_attempts}): {e}")
                time.sleep(SLEEP_BETWEEN_CALLS)

    logger.error(f"❌ [{category}] batch ล้มเหลวหลังจาก retry ครบแล้ว — ข้าม batch นี้")
    return []


# ==========================================
# Resume — รันต่อจากของเดิมแทนการเริ่มใหม่
# ==========================================
def load_existing(path: Path) -> tuple[list[dict], dict[str, int], int]:
    """
    โหลด dataset ที่เคยสร้างค้างไว้ เพื่อรันต่อโดยไม่ต้องเริ่มใหม่ (ประหยัดโควตา Gemini)

    Args:
        path: ไฟล์ output ที่อาจมีอยู่แล้ว

    Returns:
        (บทสนทนาเดิมทั้งหมด, จำนวนที่มีแล้วแยกตามหมวด, เลข conversation ถัดไปที่ควรใช้)
        ถ้าไม่มีไฟล์/ไฟล์เสีย จะคืน ([], {}, 1) = เริ่มใหม่ตามปกติ
    """
    if not path.exists():
        return [], {}, 1

    try:
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        if not isinstance(existing, list) or not existing:
            return [], {}, 1
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"⚠️ อ่านไฟล์เดิมไม่ได้ ({e}) — เริ่มสร้างใหม่ทั้งหมด")
        return [], {}, 1

    counts: dict[str, int] = {}
    max_id = 0
    for conv in existing:
        category = conv.get("category")
        if category:
            counts[category] = counts.get(category, 0) + 1
        # ดึงเลขจาก conversation_id เช่น "conv_210" -> 210 เพื่อนับต่อไม่ให้ id ซ้ำ
        match = re.search(r"(\d+)", str(conv.get("conversation_id", "")))
        if match:
            max_id = max(max_id, int(match.group(1)))

    return existing, counts, max_id + 1


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="สร้าง dataset แชทซื้อขายจำลองด้วย Gemini")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="เริ่มสร้างใหม่ทั้งหมด (ทับไฟล์เดิม) — ปกติจะรันต่อจากของเดิมอัตโนมัติ",
    )
    args = parser.parse_args()

    if not GEMINI_API_KEYS:
        logger.error(
            "❌ ไม่พบ API key — ตั้งค่า GEMINI_API_KEYS (คั่นด้วย comma) หรือ GEMINI_API_KEY ใน backend/.env ก่อนรัน"
        )
        return

    rotator = KeyRotator(GEMINI_API_KEYS)
    logger.info(
        f"🔑 ใช้ {len(GEMINI_API_KEYS)} API key หมุนสลับกัน "
        f"(คุมความเร็วรวม ~5 req/นาที — แต่ละ key แทบไม่ถูกใช้ เพื่อยืดโควตาให้อยู่นาน)"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # === Resume: รันต่อจากของเดิม (เว้นแต่สั่ง --fresh) ===
    if args.fresh:
        all_conversations, existing_counts, conv_counter = [], {}, 1
        logger.info("🆕 โหมด --fresh: เริ่มสร้างใหม่ทั้งหมด (ทับไฟล์เดิม)")
    else:
        all_conversations, existing_counts, conv_counter = load_existing(OUTPUT_FILE)
        if all_conversations:
            summary = ", ".join(f"{c}={n}" for c, n in sorted(existing_counts.items()))
            logger.info(
                f"♻️ พบข้อมูลเดิม {len(all_conversations)} แชท ({summary}) "
                f"— จะสร้างต่อเฉพาะส่วนที่ยังขาด (id ถัดไป: conv_{conv_counter:03d})"
            )

    consecutive_failures = 0  # กัน infinite loop กรณี error ถาวร (โมเดลผิด/สิทธิ์ไม่พอ)

    for category, target_count in CATEGORY_TARGETS.items():
        generated_in_category = existing_counts.get(category, 0)

        if generated_in_category >= target_count:
            logger.info(f"✅ หมวด '{category}' ครบแล้ว ({generated_in_category}/{target_count}) — ข้าม")
            continue

        logger.info(
            f"🚀 สร้าง dataset หมวด '{category}': มีแล้ว {generated_in_category}/{target_count} "
            f"— ต้องสร้างเพิ่มอีก {target_count - generated_in_category} แชท"
        )

        while generated_in_category < target_count:
            remaining = target_count - generated_in_category
            batch_size = min(BATCH_SIZE, remaining)

            batch = generate_batch(rotator, category, batch_size, conv_counter)

            if not batch:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        f"🛑 batch ล้มเหลวติดกัน {consecutive_failures} ครั้ง — หยุดการทำงาน "
                        f"(น่าจะเป็นปัญหาถาวร เช่น โมเดลใช้ไม่ได้/สิทธิ์ API ไม่พอ) "
                        f"ข้อมูลที่ได้ {len(all_conversations)} แชท ถูกเซฟไว้ที่ {OUTPUT_FILE} แล้ว"
                    )
                    return
                rotator.rotate()
                time.sleep(SLEEP_BETWEEN_CALLS)
                continue

            consecutive_failures = 0  # สำเร็จแล้ว รีเซ็ตตัวนับ
            for conv in batch:
                conv["conversation_id"] = f"conv_{conv_counter:03d}"
                conv_counter += 1
                generated_in_category += 1
                all_conversations.append(conv)

            logger.info(
                f"✅ [{category}] {generated_in_category}/{target_count} แชทแล้ว "
                f"(รวมทั้งหมด {len(all_conversations)})"
            )

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_conversations, f, ensure_ascii=False, indent=2)

            # หมุนไป key ถัดไปทุก batch (กระจายโหลดให้แต่ละ key เท่าๆ กัน — แต่ละ key แทบไม่ถูกใช้ = โควตาอยู่นาน)
            rotator.rotate()
            # พักเต็ม SLEEP_BETWEEN_CALLS เพื่อคุมความเร็วรวมไว้ที่ ~5 req/นาที (conservative)
            time.sleep(SLEEP_BETWEEN_CALLS)

    # === สรุปผล ===
    logger.info("🎉 สร้าง dataset เสร็จสิ้น")
    for category in CATEGORY_TARGETS:
        cat_convs = [c for c in all_conversations if c["category"] == category]
        avg_final_risk = (
            sum(c["messages"][-1]["risk_after"] for c in cat_convs) / len(cat_convs)
            if cat_convs
            else 0
        )
        logger.info(
            f"📊 {category}: {len(cat_convs)} แชท | risk_after เฉลี่ยตอนจบแชท = {avg_final_risk:.1f}%"
        )

    # === คุณภาพ evidence (สำคัญต่อ Evidence Bank ของ pre-filter) ===
    total_ev = sum(EVIDENCE_STATS.values()) - EVIDENCE_STATS["rejected_not_substring"]
    if total_ev > 0:
        verbatim_pct = EVIDENCE_STATS["verbatim"] / total_ev * 100
        logger.info("🧾 คุณภาพ evidence:")
        logger.info(f"     เป็นวลีคัดลอกจริง (ดี)      : {EVIDENCE_STATS['verbatim']} ({verbatim_pct:.1f}%)")
        logger.info(f"     ต้อง fallback ใช้ข้อความเต็ม : {EVIDENCE_STATS['fallback_full_text']}")
        logger.info(f"     ถูกทิ้งเพราะไม่ใช่ substring : {EVIDENCE_STATS['rejected_not_substring']}")
        if verbatim_pct < 70:
            logger.warning(
                "⚠️ evidence ที่เป็นวลีจริงต่ำกว่า 70% — Evidence Bank จะแม่นยำน้อยลง "
                "(โมเดลไม่ค่อยทำตามคำสั่งให้คัดลอกวลี)"
            )

    # === ความครอบคลุมของ pattern (เช็คว่า pattern ไหนไม่มีตัวอย่างเลย) ===
    seen_patterns = {
        pid
        for conv in all_conversations
        for msg in conv["messages"]
        for pid in msg.get("detected_flags", [])
    }
    missing_patterns = sorted(set(PATTERN_CATALOGUE) - seen_patterns)
    if missing_patterns:
        logger.warning(f"⚠️ pattern ที่ไม่มีตัวอย่างเลยใน dataset นี้: {', '.join(missing_patterns)}")
    else:
        logger.info(f"✅ มีตัวอย่างครบทั้ง {len(PATTERN_CATALOGUE)} pattern")

    logger.info(f"💾 บันทึกไฟล์ทั้งหมดที่: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
