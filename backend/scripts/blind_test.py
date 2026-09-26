"""
SafeTrade — Phase 2: Blind Test

วัด baseline ว่า qwen3:8b (ยังไม่ fine-tune) วิเคราะห์ได้ตรงกับ label ใน dataset แค่ไหน
ป้อนบทสนทนาให้โมเดลโดยไม่ให้เห็นเฉลย แล้วเก็บคำตอบดิบไว้ให้ grade_results.py คำนวณ metric ต่อ

วิธีวัด:
- วนเฉพาะ turn ของ "ผู้ขาย" (pattern เป็น seller-only ผู้ซื้อไม่มีทางเข้าข่าย — ประหยัดเวลาครึ่งหนึ่ง)
- แต่ละ turn ป้อน messages[0..n] (prefix สะสม) เหมือน production ที่ส่ง full history ทุกครั้ง
- รัน prefilter บันทึกผลไว้ด้วย แต่ "เรียกโมเดลเสมอ" ไม่ว่า prefilter จะบอกข้ามหรือไม่
  → รอบเดียวได้ข้อมูลครบทั้งมุม "โมเดลล้วนๆ" และ "ระบบจริงที่มี prefilter"

ข้อจำกัดที่ต้องระบุในรายงาน:
เฉลยใน dataset สร้างโดย Gemini ที่ได้ PATTERN_CATALOGUE ชุดเดียวกัน ตัวเลขที่ได้จึงเป็น
"qwen3:8b เห็นตรงกับ Gemini แค่ไหน" ไม่ใช่ "ถูกต้องตามความจริงแค่ไหน"

รันคำสั่ง:
    python blind_test.py --limit 3     # ทดสอบเล็กก่อน
    python blind_test.py               # รันเต็ม (~4 ชม. ควรปล่อยข้ามคืน)

ก่อนรัน: เปิด Ollama, อุ่นโมเดล 1 ครั้ง, และอย่าเปิด backend พร้อมกัน (แย่ง VRAM)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from ollama_client import analyze_risk, build_prompt  # noqa: E402
from prefilter import DEFAULT_CONTEXT_TURNS, should_escalate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)  # ปิด log HTTP ของทุก request

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_INPUT = OUTPUT_DIR / "dataset_converted.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "blind_test_results.json"

SAVE_EVERY = 10  # เซฟทุกกี่บทสนทนา (กันผลหายถ้า crash กลางทาง)

# ข้อความที่ parse_risk_response() คืนเมื่อโมเดลไม่ได้ตอบ JSON — ผลลัพธ์นี้ได้ risk 0
# ซึ่งหน้าตาเหมือน "ไม่พบความเสี่ยง" ทุกประการ ถ้าไม่นับแยกจะปนเข้าไปในผลแบบเงียบๆ
PARSE_FAIL_MARKER = "ไม่สามารถวิเคราะห์ได้"

# บทสนทนาสั้นที่รู้คำตอบแน่นอน ใช้ยิงทดสอบก่อนเริ่มรันยาว
PREFLIGHT_MESSAGES = [
    {"sender": "buyer", "text": "สนใจครับ ขอดูรูปเพิ่มได้ไหม"},
    {"sender": "seller", "text": "ทักไลน์มาคุยกันดีกว่าจ้า แอดไลน์ @sale_iphone_th2 น้า ในนี้ตอบช้า"},
]
PREFLIGHT_ROUNDS = 3


def is_failed(result: dict) -> bool:
    """ผลนี้คือความล้มเหลว (เชื่อมต่อไม่ได้ หรือโมเดลไม่ตอบ JSON) ไม่ใช่คำตอบจริง"""
    return (result.get("risk_percentage") or 0) < 0 or PARSE_FAIL_MARKER in str(result.get("summary", ""))


def preflight() -> bool:
    """
    ยิงทดสอบก่อนรันยาวหลายชั่วโมง

    เคยเกิดมาแล้ว: Ollama อัปเดตตัวเองแล้วเปลี่ยนให้ qwen3 คิดยาวจนตอบว่าง ทุก turn กลายเป็น
    risk 0 โดยไม่มี error สักตัว ถ้าไม่ตรวจก่อน จะได้ผลขยะ 4 ชั่วโมงเต็ม

    Returns:
        True ถ้าตอบ JSON ได้ครบทุกรอบ
    """
    for i in range(1, PREFLIGHT_ROUNDS + 1):
        result = analyze_risk(PREFLIGHT_MESSAGES)
        if is_failed(result):
            logger.error(f"❌ preflight รอบ {i}: โมเดลไม่ได้ตอบ JSON — {result.get('summary')}")
            logger.error("   ตรวจ Ollama (เวอร์ชัน/การตั้งค่า think) ก่อน แล้วค่อยรันใหม่")
            return False
    logger.info(f"✅ preflight ผ่าน {PREFLIGHT_ROUNDS}/{PREFLIGHT_ROUNDS} รอบ")
    return True


def to_model_messages(messages: list[dict]) -> list[dict]:
    """
    แปลง schema ของ dataset ให้ตรงกับที่ ollama_client ต้องการ

    dataset_converted.json ใช้ field 'role' แต่ build_prompt() อ่าน 'sender'
    """
    return [{"sender": m["role"], "text": m["text"]} for m in messages]


def extract_pattern_ids(detected_flags: list[str]) -> list[str]:
    """
    ดึง pattern_id จาก detected_flags ที่โมเดลคืนมา

    รูปแบบที่ _validate_risk_data() สร้าง: "[B3] <label> (+20): <evidence>"

    Args:
        detected_flags: list ของ string ที่ format แล้ว

    Returns:
        list ของ pattern_id เช่น ["B3", "A2"]
    """
    ids: list[str] = []
    for flag in detected_flags or []:
        match = re.match(r"\[([A-D]\d)\]", str(flag).strip())
        if match:
            ids.append(match.group(1))
    return ids


def load_existing_results(path: Path) -> tuple[list[dict], set[str]]:
    """
    โหลดผลที่รันค้างไว้ เพื่อรันต่อโดยไม่ต้องเริ่มใหม่ (การรันเต็มใช้เวลา ~4 ชม.)

    Returns:
        (records เดิม, set ของ conversation_id ที่ทำเสร็จแล้ว)
    """
    if not path.exists():
        return [], set()
    try:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        done = {r["conversation_id"] for r in records}
        return records, done
    except (json.JSONDecodeError, OSError, KeyError) as e:
        logger.warning(f"⚠️ อ่านผลเดิมไม่ได้ ({e}) — เริ่มใหม่ทั้งหมด")
        return [], set()


def evaluate_conversation(conv: dict) -> list[dict]:
    """
    ประเมิน 1 บทสนทนา — ยิงโมเดลทีละ turn ของผู้ขาย

    Args:
        conv: บทสนทนา schema ใหม่ (มี messages/evidence/score_by_turn)

    Returns:
        list ของ record (1 record ต่อ 1 turn ผู้ขาย)
    """
    messages = conv["messages"]

    # เฉลย: turn ไหนมี pattern อะไรบ้าง / คะแนนสะสมเท่าไหร่
    gt_patterns: dict[int, list[str]] = {}
    for ev in conv.get("evidence", []):
        gt_patterns.setdefault(ev["turn"], []).append(ev["pattern"])
    gt_risk = {s["turn"]: s["risk_after"] for s in conv.get("score_by_turn", [])}

    records: list[dict] = []

    for idx, msg in enumerate(messages):
        if msg["role"] != "seller":
            continue  # pattern เป็น seller-only

        prefix = messages[: idx + 1]
        model_messages = to_model_messages(prefix)

        # --- prefilter (บันทึกไว้เฉยๆ ไม่ได้ใช้ตัดสินว่าจะยิงโมเดลไหม) ---
        window_start = max(0, idx - DEFAULT_CONTEXT_TURNS)
        window = to_model_messages(messages[window_start : idx + 1])
        try:
            decision = should_escalate(
                current_message=window[-1],
                recent_messages=window[:-1],
                context_turns=len(window) - 1,
            )
            pf = {
                "escalate": decision.escalate,
                "stage": decision.stage,
                "similarity": decision.similarity,
            }
        except Exception as e:  # prefilter ล้มไม่ควรทำให้ทั้งงานหยุด
            logger.warning(f"⚠️ prefilter ล้มเหลวที่ {conv['conversation_id']} turn {msg['turn']}: {e}")
            pf = {"escalate": True, "stage": "error", "similarity": None}

        # --- เรียกโมเดลเสมอ เพื่อวัด baseline ของโมเดลล้วนๆ ---
        result = analyze_risk(model_messages)

        records.append(
            {
                "conversation_id": conv["conversation_id"],
                "category": conv.get("category"),
                "turn": msg["turn"],
                "text": msg["text"],
                "ground_truth": {
                    "patterns": sorted(set(gt_patterns.get(msg["turn"], []))),
                    "risk_after": gt_risk.get(msg["turn"]),
                },
                "prefilter": pf,
                "model": {
                    "patterns": sorted(set(extract_pattern_ids(result.get("detected_flags", [])))),
                    "risk_percentage": result.get("risk_percentage"),
                    "summary": result.get("summary"),
                    "detected_flags": result.get("detected_flags", []),
                },
                # เก็บ prompt ไว้ใช้สร้าง DPO pair ใน Phase 4 (จะได้ไม่ต้องประกอบใหม่)
                "prompt": build_prompt(model_messages),
            }
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind test — วัด baseline ของโมเดลก่อน fine-tune")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="จำกัดจำนวนบทสนทนา (ใช้ทดสอบ)")
    parser.add_argument("--fresh", action="store_true", help="เริ่มใหม่ ไม่ resume ของเดิม")
    parser.add_argument(
        "--only-conversations",
        type=Path,
        default=None,
        help="ไฟล์ holdout_conversations.json — รันเฉพาะบทสนทนาในลิสต์ holdout (ใช้วัดผลหลัง fine-tune)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"❌ ไม่พบไฟล์ {args.input} — รัน convert_dataset.py ก่อน")
        return

    with open(args.input, encoding="utf-8") as f:
        dataset = json.load(f)

    if args.only_conversations:
        if not args.only_conversations.exists():
            logger.error(f"❌ ไม่พบไฟล์ {args.only_conversations} — รัน build_dpo_pairs.py ก่อน")
            return
        with open(args.only_conversations, encoding="utf-8") as f:
            wanted = set(json.load(f)["holdout"])
        dataset = [c for c in dataset if c["conversation_id"] in wanted]
        logger.info(f"🎯 รันเฉพาะ holdout {len(dataset)} บทสนทนา")

    if args.limit:
        dataset = dataset[: args.limit]

    records, done = ([], set()) if args.fresh else load_existing_results(args.output)
    if done:
        logger.info(f"♻️ พบผลเดิม {len(records)} record จาก {len(done)} บทสนทนา — จะทำต่อจากเดิม")

    todo = [c for c in dataset if c["conversation_id"] not in done]
    if not todo:
        logger.info("✅ ทำครบทุกบทสนทนาแล้ว ไม่มีอะไรต้องรันเพิ่ม")
        return

    if not preflight():
        return

    total_turns = sum(1 for c in todo for m in c["messages"] if m["role"] == "seller")
    logger.info(f"🚀 เริ่ม blind test: {len(todo)} บทสนทนา / ~{total_turns} turn ผู้ขาย")
    logger.info("   (เรียกโมเดลทุก turn — prefilter บันทึกไว้เฉยๆ ไม่ได้ใช้ตัดสิน)")

    started = time.time()
    done_turns = 0
    errors = 0

    for i, conv in enumerate(todo, start=1):
        conv_records = evaluate_conversation(conv)
        records.extend(conv_records)
        done_turns += len(conv_records)
        errors += sum(1 for r in conv_records if is_failed(r["model"]))

        elapsed = time.time() - started
        rate = done_turns / elapsed if elapsed > 0 else 0
        eta_min = (total_turns - done_turns) / rate / 60 if rate > 0 else 0
        logger.info(
            f"[{i}/{len(todo)}] {conv['conversation_id']} — "
            f"{done_turns}/{total_turns} turn | {rate:.2f} turn/วิ | เหลือ ~{eta_min:.0f} นาที"
        )

        if i % SAVE_EVERY == 0 or i == len(todo):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 เซฟแล้ว {len(records)} record")

    logger.info("🎉 blind test เสร็จสิ้น")
    logger.info(f"   record ทั้งหมด : {len(records)}")
    logger.info(f"   ใช้เวลา       : {(time.time() - started) / 60:.1f} นาที")
    if errors:
        logger.warning(f"   ⚠️ โมเดล error หรือไม่ได้ตอบ JSON: {errors} turn — ตรวจก่อนเอาผลไปใช้")
    logger.info(f"💾 บันทึกที่: {args.output}")
    logger.info("ขั้นต่อไป: python grade_results.py")


if __name__ == "__main__":
    main()
