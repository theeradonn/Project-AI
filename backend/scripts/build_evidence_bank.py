"""
SafeTrade — Evidence Bank Builder

ดึง field `evidence` จากทุกบทสนทนาใน dataset (schema ใหม่) มารวมเป็น Evidence Bank
แล้วคำนวณ embedding ล่วงหน้าครั้งเดียว เก็บเป็นไฟล์ เพื่อให้ pre-filter ด่านที่ 2
เทียบ cosine similarity ได้เร็ว โดยไม่ต้อง encode คลังใหม่ทุกครั้ง

Output 2 ไฟล์ (คู่กัน — index ตรงกันแถวต่อแถว):
    output/evidence_bank.json    list ของ {"text", "pattern", "source_conversation_id", "turn"}
    output/evidence_vectors.npy  numpy array shape (n_evidence, embedding_dim)

ต้องเตรียมก่อนรัน:
    ollama pull qwen3-embedding:0.6b
    pip install numpy

รันคำสั่ง:
    python build_evidence_bank.py
    python build_evidence_bank.py --input output/dataset_converted.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from embedding_client import EMBEDDING_MODEL, EmbeddingError, embed_texts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_INPUT = OUTPUT_DIR / "dataset_converted.json"
BANK_PATH = OUTPUT_DIR / "evidence_bank.json"
VECTORS_PATH = OUTPUT_DIR / "evidence_vectors.npy"

# ตัดประโยคที่สั้นเกินไปทิ้ง (เช่น "ไม่มีครับ") เพราะ embedding ของประโยคสั้นมาก
# มักคล้ายกับข้อความปกติทั่วไป ทำให้เกิด false positive เยอะเกินจำเป็น
MIN_EVIDENCE_LENGTH = 6

# pattern ที่ "ความเสี่ยงอยู่ที่พฤติกรรมข้าม turn ไม่ใช่ที่ตัวประโยค" — ห้ามเอาเข้า Evidence Bank
#
# C2 (ตอบกำกวมซ้ำๆ): evidence ของมันคือประโยคอย่าง "ใช้งานปกติทุกอย่างครับ" ซึ่ง
# แยกไม่ออกจากคำตอบของผู้ขายสุจริตเลยแม้แต่น้อย มันเป็น C2 ได้เพราะ "ผู้ขายเลี่ยงตอบซ้ำหลายครั้ง"
# ซึ่งเป็นบริบทข้าม turn ที่ embedding ระดับประโยคเดียวมองไม่เห็น
#
# วัดจริงกับ dataset 500 แชท: การมี C2 ใน bank ทำให้ข้อความปกติถูก escalate ผิด
# (false positive 13% -> 8% เมื่อตัดออก) เพราะข้อความปกติไปแมตช์ evidence ของ C2 ที่ ~0.84
#
# ผลข้างเคียงที่ยอมรับไว้: prefilter จะไม่จับ C2 จากด่าน embedding อีกต่อไป (จับได้ 95% -> 0%)
# แต่ C2 ยังอยู่ใน PATTERN_CATALOGUE ปกติ — Qwen ยังตรวจเจอและคิดคะแนน +10 ให้เหมือนเดิม
# เมื่อแชทนั้นถูก escalate ด้วย pattern อื่น (ซึ่งแชทโกงมักมีหลาย pattern อยู่แล้ว)
CONTEXT_DEPENDENT_PATTERNS = {"C2"}


def extract_evidence(dataset: list[dict]) -> list[dict]:
    """
    ดึง evidence ทั้งหมดจาก dataset แล้วตัดรายการซ้ำออก

    ข้าม pattern ใน CONTEXT_DEPENDENT_PATTERNS เพราะประโยคของมันดูเหมือนข้อความปกติ
    ทำให้ Evidence Bank เสียความแม่นยำ (ดูเหตุผลละเอียดที่ comment ของตัวแปรนั้น)

    Args:
        dataset: บทสนทนา schema ใหม่ (มี field evidence[])

    Returns:
        list ของ {"text", "pattern", "source_conversation_id", "turn"} ที่ไม่ซ้ำกัน
    """
    seen_texts: set[str] = set()
    skipped_context_dependent = 0
    bank: list[dict] = []
    skipped_short = 0

    for conv in dataset:
        conv_id = conv.get("conversation_id")
        for ev in conv.get("evidence", []) or []:
            text = str(ev.get("text", "")).strip()
            pattern = ev.get("pattern")

            if not text or not pattern:
                continue
            if pattern in CONTEXT_DEPENDENT_PATTERNS:
                skipped_context_dependent += 1
                continue
            if len(text) < MIN_EVIDENCE_LENGTH:
                skipped_short += 1
                continue

            dedupe_key = f"{pattern}::{text}"
            if dedupe_key in seen_texts:
                continue
            seen_texts.add(dedupe_key)

            bank.append(
                {
                    "text": text,
                    "pattern": pattern,
                    "source_conversation_id": conv_id,
                    "turn": ev.get("turn"),
                }
            )

    if skipped_short:
        logger.info(f"ℹ️ ข้าม evidence ที่สั้นกว่า {MIN_EVIDENCE_LENGTH} ตัวอักษร: {skipped_short} รายการ")
    if skipped_context_dependent:
        logger.info(
            f"ℹ️ ข้าม pattern ที่พึ่งบริบทข้าม turn ({', '.join(sorted(CONTEXT_DEPENDENT_PATTERNS))}): "
            f"{skipped_context_dependent} รายการ — ประโยคดูเหมือนข้อความปกติ ทำให้ bank เสียความแม่นยำ"
        )

    return bank


def main() -> None:
    parser = argparse.ArgumentParser(description="สร้าง Evidence Bank + embedding ล่วงหน้า")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(
            f"❌ ไม่พบไฟล์ {args.input} — รัน convert_dataset.py ก่อนเพื่อสร้าง dataset schema ใหม่"
        )
        return

    try:
        import numpy as np
    except ImportError:
        logger.error("❌ ไม่มี numpy — ติดตั้งก่อน: pip install numpy")
        return

    with open(args.input, encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"📂 โหลด {len(dataset)} บทสนทนาจาก {args.input.name}")

    bank = extract_evidence(dataset)
    if not bank:
        logger.error("❌ ไม่พบ evidence เลยใน dataset — ตรวจสอบว่า dataset มี field evidence หรือยัง")
        return

    logger.info(f"🧾 evidence ที่ไม่ซ้ำกัน: {len(bank)} ประโยค")

    pattern_counts: dict[str, int] = {}
    for item in bank:
        pattern_counts[item["pattern"]] = pattern_counts.get(item["pattern"], 0) + 1
    logger.info("📊 แยกตาม pattern:")
    for pid in sorted(pattern_counts, key=lambda p: -pattern_counts[p]):
        logger.info(f"     {pid}: {pattern_counts[pid]}")

    logger.info(f"🧠 กำลังสร้าง embedding ด้วย {EMBEDDING_MODEL} ...")
    try:
        vectors = embed_texts([item["text"] for item in bank], batch_size=args.batch_size)
    except EmbeddingError as e:
        logger.error(f"❌ สร้าง embedding ไม่สำเร็จ: {e}")
        return

    matrix = np.array(vectors, dtype=np.float32)
    logger.info(f"✅ ได้ embedding matrix: {matrix.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(BANK_PATH, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)
    np.save(VECTORS_PATH, matrix)

    logger.info(f"💾 บันทึกแล้ว: {BANK_PATH.name} + {VECTORS_PATH.name}")
    logger.info("🎉 Evidence Bank พร้อมใช้งาน — pre-filter ด่านที่ 2 จะเริ่มทำงานอัตโนมัติ")


if __name__ == "__main__":
    main()
