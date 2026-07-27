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
      "scam_pattern": "..."            # label ของ pattern ที่ turn นั้นเจอ (base_score สูงสุดใน turn) หรือ null ถ้าไม่มี flag
    }
  ]
}

กฎที่บังคับใช้หลัง generate (ไม่พึ่งพาโมเดลให้ทำตามเอง — สอดคล้องกับ ollama_client.py):
- risk_after ของแต่ละ turn = ผลรวม base_score ของทุก pattern_id ที่ตรวจพบสะสมตั้งแต่ต้นแชทจนถึง turn นั้น
  (นับ pattern_id ซ้ำแค่ครั้งเดียว, clamp 0-100) — คำนวณที่ Python ทั้งหมด ไม่เชื่อตัวเลขจาก Gemini
- risk_after จึงไม่มีทางลดลงจาก turn ก่อนหน้าโดยธรรมชาติของการสะสม (running sum)

รันคำสั่ง:
    python generate_dataset.py
ต้องตั้งค่า GEMINI_API_KEY ใน backend/.env ก่อนรัน (ห้าม hardcode key ในไฟล์นี้)
"""

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
OUTPUT_FILE = OUTPUT_DIR / "dataset_500v2.json"

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
        "(เช่น เร่งรัดกดดัน B4, ราคาผิดปกติ B1, บ่ายเบี่ยงเล็กน้อย C2/D1)"
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
        "scam_pattern": "ปลอดภัย"
      }}
    ]
  }}
]"""


# ==========================================
# Rule Enforcement (ไม่พึ่งพาโมเดลให้ทำตามกฎเอง)
# ==========================================
def enforce_score_rules(messages: list[dict]) -> list[dict]:
    """
    คำนวณ risk_after + scam_pattern แบบ deterministic — ไม่เชื่อค่าที่ Gemini ใส่มา คำนวณใหม่จาก detected_flags:
    - risk_after = ผลรวม base_score ของทุก pattern_id ที่ตรวจพบสะสมตั้งแต่ต้นแชท (นับซ้ำแค่ครั้งเดียว) clamp 0-100
    - scam_pattern = label ของ pattern ที่มี base_score สูงสุด 'ใน turn นั้นเอง' (ตรงกับ detected_flags ของ turn นั้น)
      ถ้า turn ไหนไม่มี flag → None (รวมถึง buyer ทุก turn ที่ detected_flags ว่างเสมอ)
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
# Main
# ==========================================
def main():
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

    all_conversations: list[dict] = []
    conv_counter = 1

    consecutive_failures = 0  # กัน infinite loop กรณี error ถาวร (โมเดลผิด/สิทธิ์ไม่พอ)

    for category, target_count in CATEGORY_TARGETS.items():
        logger.info(f"🚀 เริ่มสร้าง dataset หมวด '{category}' เป้าหมาย {target_count} แชท")
        generated_in_category = 0

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
    logger.info(f"💾 บันทึกไฟล์ทั้งหมดที่: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
