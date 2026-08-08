"""
SafeTrade — Dataset Converter / Validator

แปลง dataset จาก schema เดิม (flat: ข้อมูลทุกอย่างอยู่ใน messages[]) ให้เป็น schema ใหม่
ที่แยก evidence / score_by_turn ออกมาเป็น array ระดับบทสนทนา เพื่อให้:
  1. ดึง evidence รายประโยคไปสร้าง Evidence Bank ได้ง่าย (ระบบ pre-filter)
  2. เก็บ score_by_turn ที่มาจาก deterministic scoring เดียวกับ production (กัน scale เพี้ยนตอน DPO)

Schema เดิม (input):
{
  "conversation_id": "conv_001",
  "category": "fraud|normal|suspicious",
  "messages": [
    {"turn": 1, "sender": "buyer|seller", "text": "...", "risk_after": 0,
     "detected_flags": ["A2"], "evidence": [{"pattern_id": "A2", "text": "..."}],
     "scam_pattern": "..."}
  ]
}

Schema ใหม่ (output):
{
  "conversation_id": "conv_001",
  "category": "fraud|normal|suspicious",
  "patterns": ["A2", "B3"],                <- array เพราะ 1 บทสนทนามีได้หลาย pattern
  "messages":      [{"turn": 1, "role": "buyer|seller", "text": "..."}],
  "evidence":      [{"turn": 6, "text": "...", "pattern": "A2"}],
  "score_by_turn": [{"turn": 1, "risk_after": 0}]
}

รันคำสั่ง:
    python convert_dataset.py                 # แปลง + validate
    python convert_dataset.py --validate-only # ตรวจอย่างเดียว ไม่เขียนไฟล์
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from ollama_client import PATTERN_CATALOGUE  # noqa: E402

from generate_dataset import enforce_score_rules  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
# ใช้ v3 (มี evidence เป็นวลีจริง + ครอบคลุม D3) ถ้ามี — ไม่งั้น fallback ไป v2 ตัวเก่า
_V3 = OUTPUT_DIR / "dataset_500v3.json"
DEFAULT_INPUT = _V3 if _V3.exists() else OUTPUT_DIR / "dataset_500v2.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "dataset_converted.json"

VALID_ROLES = {"buyer", "seller"}


def convert_conversation(conv: dict) -> dict:
    """
    แปลงบทสนทนา 1 รายการจาก schema เดิม → schema ใหม่

    หมายเหตุ: จะเรียก enforce_score_rules() ซ้ำเสมอ เพื่อรับประกันว่า risk_after
    มาจาก deterministic scoring function เดียวกับ production (ไม่ใช่ค่าที่ติดมากับไฟล์)

    Args:
        conv: บทสนทนา schema เดิม

    Returns:
        บทสนทนา schema ใหม่
    """
    messages = conv.get("messages", [])

    # คำนวณคะแนน/flag/evidence ใหม่ทั้งหมดด้วย logic เดียวกับ production
    messages = enforce_score_rules(messages)

    out_messages: list[dict] = []
    out_evidence: list[dict] = []
    out_scores: list[dict] = []
    patterns: list[str] = []

    for idx, msg in enumerate(messages):
        turn = msg.get("turn", idx + 1)
        role = msg.get("role") or msg.get("sender")

        out_messages.append({"turn": turn, "role": role, "text": msg.get("text", "")})
        out_scores.append({"turn": turn, "risk_after": msg.get("risk_after", 0)})

        for ev in msg.get("evidence", []) or []:
            pattern_id = ev.get("pattern_id") or ev.get("pattern")
            text = str(ev.get("text", "")).strip()
            if not pattern_id or not text:
                continue
            out_evidence.append({"turn": turn, "text": text, "pattern": pattern_id})
            if pattern_id not in patterns:
                patterns.append(pattern_id)

    return {
        "conversation_id": conv.get("conversation_id"),
        "category": conv.get("category"),
        "patterns": patterns,
        "messages": out_messages,
        "evidence": out_evidence,
        "score_by_turn": out_scores,
    }


def validate_conversation(conv: dict) -> list[str]:
    """
    ตรวจสอบว่าบทสนทนาตรงตาม schema ใหม่หรือไม่

    Args:
        conv: บทสนทนา schema ใหม่

    Returns:
        list ของข้อความ error — list ว่าง = ผ่านทั้งหมด
    """
    errors: list[str] = []
    cid = conv.get("conversation_id", "<ไม่มี id>")

    for field_name in ("conversation_id", "messages", "evidence", "score_by_turn", "patterns"):
        if field_name not in conv:
            errors.append(f"[{cid}] ขาด field '{field_name}'")

    messages = conv.get("messages", [])
    if not messages:
        errors.append(f"[{cid}] messages ว่างเปล่า")

    turns = set()
    for m in messages:
        if m.get("role") not in VALID_ROLES:
            errors.append(f"[{cid}] turn {m.get('turn')}: role ไม่ถูกต้อง ({m.get('role')!r})")
        if not str(m.get("text", "")).strip():
            errors.append(f"[{cid}] turn {m.get('turn')}: text ว่างเปล่า")
        turns.add(m.get("turn"))

    # score_by_turn ต้องครบทุก turn และห้ามลดลง (risk สะสมขึ้นอย่างเดียว)
    scores = conv.get("score_by_turn", [])
    if len(scores) != len(messages):
        errors.append(f"[{cid}] score_by_turn ({len(scores)}) ไม่เท่ากับจำนวน messages ({len(messages)})")

    previous = -1
    for s in scores:
        risk = s.get("risk_after")
        if not isinstance(risk, (int, float)):
            errors.append(f"[{cid}] turn {s.get('turn')}: risk_after ไม่ใช่ตัวเลข")
            continue
        if risk < previous:
            errors.append(f"[{cid}] turn {s.get('turn')}: risk_after ลดลง ({previous} → {risk})")
        previous = risk

    # evidence ต้องอ้าง turn ที่มีจริง และ pattern ต้องอยู่ใน catalogue
    for ev in conv.get("evidence", []):
        if ev.get("turn") not in turns:
            errors.append(f"[{cid}] evidence อ้าง turn {ev.get('turn')} ที่ไม่มีอยู่จริง")
        if ev.get("pattern") not in PATTERN_CATALOGUE:
            errors.append(f"[{cid}] evidence ใช้ pattern ที่ไม่รู้จัก: {ev.get('pattern')!r}")
        if not str(ev.get("text", "")).strip():
            errors.append(f"[{cid}] evidence turn {ev.get('turn')}: text ว่างเปล่า")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="แปลง/ตรวจสอบ dataset ให้ตรง schema ใหม่")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true", help="ตรวจอย่างเดียว ไม่เขียนไฟล์")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"❌ ไม่พบไฟล์ input: {args.input}")
        return

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"📂 โหลด {len(data)} บทสนทนาจาก {args.input.name}")

    converted = [convert_conversation(c) for c in data]

    all_errors: list[str] = []
    for conv in converted:
        all_errors.extend(validate_conversation(conv))

    if all_errors:
        logger.warning(f"⚠️ พบปัญหา {len(all_errors)} รายการ (แสดง 20 รายการแรก):")
        for err in all_errors[:20]:
            logger.warning(f"   {err}")
    else:
        logger.info("✅ ทุกบทสนทนาผ่าน validation")

    # === สรุปสถิติ ===
    total_evidence = sum(len(c["evidence"]) for c in converted)
    with_evidence = sum(1 for c in converted if c["evidence"])
    logger.info(f"📊 บทสนทนาทั้งหมด: {len(converted)}")
    logger.info(f"📊 บทสนทนาที่มี evidence: {with_evidence}")
    logger.info(f"📊 evidence รวมทั้งหมด: {total_evidence} ประโยค")

    pattern_counts: dict[str, int] = {}
    for conv in converted:
        for ev in conv["evidence"]:
            pattern_counts[ev["pattern"]] = pattern_counts.get(ev["pattern"], 0) + 1
    if pattern_counts:
        logger.info("📊 evidence แยกตาม pattern:")
        for pid in sorted(pattern_counts, key=lambda p: -pattern_counts[p]):
            logger.info(f"     {pid}: {pattern_counts[pid]}")

    if args.validate_only:
        logger.info("ℹ️ โหมด --validate-only: ไม่เขียนไฟล์")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 บันทึกไฟล์แล้ว: {args.output}")


if __name__ == "__main__":
    main()
