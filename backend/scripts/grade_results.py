"""
SafeTrade — Phase 3: Grading

อ่านผลดิบจาก blind_test.py แล้วคำนวณว่าโมเดลแม่นแค่ไหน
ไม่เรียกโมเดลเลย จึงรันซ้ำได้ทันทีถ้าอยากเปลี่ยนเกณฑ์วัด

สำคัญ — วิธีเทียบ:
เฉลยใน dataset บันทึก pattern แบบ "เพิ่งโผล่ใหม่ใน turn นั้น" แต่โมเดลเห็น messages[0..n]
ทั้งหมด จึงรายงาน pattern ที่เจอมาก่อนหน้าด้วย ถ้าเทียบตรงๆ pattern เก่าจะถูกนับเป็น
false positive ทั้งที่ตอบถูก

จึงเทียบแบบ "สะสม" เป็นหลัก: pattern ที่โมเดลตอบใน turn n เทียบกับ union ของเฉลย turn 1..n
ซึ่งตรงกับสิ่งที่โมเดลมองเห็นจริง

รันคำสั่ง:
    python grade_results.py
    python grade_results.py --input output/blind_test_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Windows console ใช้ codepage ไทย (cp874) ซึ่ง encode emoji ไม่ได้ — บังคับเป็น UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from ollama_client import PATTERN_CATALOGUE  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_INPUT = OUTPUT_DIR / "blind_test_results.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "grading_report.json"

# เกณฑ์คัด hard case ไปทำ DPO ใน Phase 4
RISK_ERROR_THRESHOLD = 20

# ข้อความที่ parse_risk_response() คืนเมื่อโมเดลไม่ได้ตอบ JSON (ได้ risk 0 หน้าตาเหมือนไม่พบความเสี่ยง)
PARSE_FAIL_MARKER = "ไม่สามารถวิเคราะห์ได้"


def build_cumulative_view(records: list[dict]) -> list[dict]:
    """
    เติมมุมมองแบบสะสมให้แต่ละ record

    - gt_cumulative      : union ของ pattern เฉลยตั้งแต่ turn แรกถึง turn นี้
    - model_cumulative   : union ของ pattern ที่โมเดลเคยตอบตั้งแต่ turn แรกถึง turn นี้
    - system_risk        : คะแนนที่ "ระบบจริง" จะแสดง (prefilter ข้าม = คงคะแนนเดิม, max ไม่ให้ลด)
    - system_patterns    : pattern ที่ระบบจริงจะรู้ (turn ที่ prefilter ข้าม = ไม่ได้ pattern ใหม่)

    Args:
        records: ผลดิบจาก blind_test.py (ต้องเรียงตาม conversation แล้ว turn)

    Returns:
        records ชุดเดิมที่เติม field แล้ว
    """
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_conv[r["conversation_id"]].append(r)

    for conv_records in by_conv.values():
        conv_records.sort(key=lambda r: r["turn"])

        gt_seen: set[str] = set()
        model_seen: set[str] = set()
        system_seen: set[str] = set()
        system_risk = 0

        for r in conv_records:
            gt_seen |= set(r["ground_truth"]["patterns"])
            model_seen |= set(r["model"]["patterns"])

            r["gt_cumulative"] = sorted(gt_seen)
            r["model_cumulative"] = sorted(model_seen)

            # จำลองระบบจริง: เรียกโมเดลเฉพาะตอน prefilter บอก escalate
            if r["prefilter"]["escalate"]:
                system_seen |= set(r["model"]["patterns"])
                model_risk = r["model"]["risk_percentage"] or 0
                system_risk = max(system_risk, model_risk)  # เหมือน supabase_listener
            r["system_cumulative"] = sorted(system_seen)
            r["system_risk"] = system_risk

    return records


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """คำนวณ precision / recall / f1 (คืน 0 ถ้าหารด้วยศูนย์)"""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def score_patterns(records: list[dict], gt_key: str, pred_key: str) -> dict:
    """
    คำนวณ precision/recall/f1 รายตัวทุก pattern + confusion

    Args:
        records: records ที่ผ่าน build_cumulative_view แล้ว
        gt_key: field ของเฉลย (เช่น "gt_cumulative")
        pred_key: field ของคำตอบโมเดล (เช่น "model_cumulative")

    Returns:
        dict ของผลรวมและรายละเอียดต่อ pattern
    """
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in records:
        gt = set(r[gt_key])
        pred = set(r[pred_key])

        for pid in gt & pred:
            tp[pid] += 1
        for pid in pred - gt:
            fp[pid] += 1
        for pid in gt - pred:
            fn[pid] += 1
            # โมเดลพลาด pattern นี้ แล้วไปตอบอะไรแทน
            for wrong in pred - gt:
                confusion[pid][wrong] += 1

    per_pattern = {}
    for pid in PATTERN_CATALOGUE:
        p, r_, f = prf(tp[pid], fp[pid], fn[pid])
        per_pattern[pid] = {
            "support": tp[pid] + fn[pid],
            "tp": tp[pid], "fp": fp[pid], "fn": fn[pid],
            "precision": round(p, 3), "recall": round(r_, 3), "f1": round(f, 3),
        }

    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_p, micro_r, micro_f = prf(total_tp, total_fp, total_fn)
    scored = [v for v in per_pattern.values() if v["support"] > 0]
    macro_f = sum(v["f1"] for v in scored) / len(scored) if scored else 0.0

    return {
        "per_pattern": per_pattern,
        "micro": {"precision": round(micro_p, 3), "recall": round(micro_r, 3), "f1": round(micro_f, 3)},
        "macro_f1": round(macro_f, 3),
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def score_risk(records: list[dict], pred_key: str) -> dict:
    """คำนวณความคลาดเคลื่อนของคะแนนความเสี่ยง"""
    diffs = []
    for r in records:
        gt = r["ground_truth"]["risk_after"]
        pred = r[pred_key]
        if gt is None or pred is None:
            continue
        diffs.append(abs(gt - pred))
    if not diffs:
        return {"mae": None, "over_threshold_pct": None, "count": 0}
    over = sum(1 for d in diffs if d > RISK_ERROR_THRESHOLD)
    return {
        "mae": round(sum(diffs) / len(diffs), 2),
        "over_threshold_pct": round(over / len(diffs) * 100, 1),
        "over_threshold_count": over,
        "count": len(diffs),
    }


def score_conversations(records: list[dict]) -> dict:
    """ดูภาพรวมระดับบทสนทนา — คะแนนสุดท้ายคลาดแค่ไหน แยกตามหมวด"""
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_conv[r["conversation_id"]].append(r)

    by_category: dict[str, list[float]] = defaultdict(list)
    for conv_records in by_conv.values():
        last = max(conv_records, key=lambda r: r["turn"])
        gt = last["ground_truth"]["risk_after"]
        pred = last["model"]["risk_percentage"]
        if gt is None or pred is None:
            continue
        by_category[last["category"]].append(abs(gt - pred))

    return {
        cat: {
            "conversations": len(vals),
            "final_score_mae": round(sum(vals) / len(vals), 2) if vals else None,
        }
        for cat, vals in sorted(by_category.items())
    }


def print_report(title: str, patterns: dict, risk: dict) -> None:
    """พิมพ์รายงานลง console"""
    print(f"\n{'=' * 66}")
    print(f"  {title}")
    print("=" * 66)
    print(f"{'pattern':<8}{'n':>6}{'P':>8}{'R':>8}{'F1':>8}   label")
    print("-" * 66)
    for pid, v in patterns["per_pattern"].items():
        if v["support"] == 0:
            continue
        label = PATTERN_CATALOGUE[pid]["label"][:26]
        print(f"{pid:<8}{v['support']:>6}{v['precision']:>8.3f}{v['recall']:>8.3f}{v['f1']:>8.3f}   {label}")
    m = patterns["micro"]
    print("-" * 66)
    print(f"{'micro':<8}{'':>6}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}")
    print(f"{'macro F1':<8}{'':>6}{'':>8}{'':>8}{patterns['macro_f1']:>8.3f}")
    print()
    print(f"  คะแนนความเสี่ยง: MAE = {risk['mae']} จุด | คลาดเกิน {RISK_ERROR_THRESHOLD} จุด = {risk['over_threshold_pct']}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="คำนวณความแม่นจากผล blind test")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--only-conversations",
        type=Path,
        default=None,
        help="ไฟล์ holdout_conversations.json — คิดคะแนนเฉพาะบทใน holdout (ใช้ทำ baseline ก่อน fine-tune)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ ไม่พบไฟล์ {args.input} — รัน blind_test.py ก่อน")
        return

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)

    if args.only_conversations:
        with open(args.only_conversations, encoding="utf-8") as f:
            wanted = set(json.load(f)["holdout"])
        records = [r for r in records if r["conversation_id"] in wanted]
        print(f"🎯 คิดคะแนนเฉพาะ holdout {len(wanted)} บทสนทนา")

    records = build_cumulative_view(records)
    conversations = len({r["conversation_id"] for r in records})
    errors = sum(1 for r in records if (r["model"]["risk_percentage"] or 0) < 0)
    parse_fails = sum(1 for r in records if PARSE_FAIL_MARKER in str(r["model"].get("summary", "")))

    print(f"\n📂 อ่านผล {len(records)} turn จาก {conversations} บทสนทนา")
    if errors:
        print(f"⚠️  โมเดล error (risk = -1): {errors} turn — ไม่ถูกนับในผล")
    if parse_fails:
        print(
            f"⚠️  โมเดลไม่ได้ตอบ JSON {parse_fails} turn — ถูกนับเป็น risk 0 ปนอยู่ในผล "
            f"ถ้าเยอะแปลว่าผลชุดนี้เชื่อไม่ได้"
        )

    # --- 1) โมเดลล้วนๆ (ไม่สนใจ prefilter) ---
    model_patterns = score_patterns(records, "gt_cumulative", "model_cumulative")
    # คะแนนของโมเดลคือผลรวมของรอบนั้น — ดึงขึ้นมาเป็น field ชั่วคราวเพื่อส่งให้ score_risk()
    for r in records:
        r["_model_risk"] = r["model"]["risk_percentage"]
    model_risk = score_risk(records, "_model_risk")
    print_report("โมเดลล้วนๆ (เรียกทุก turn ไม่ผ่าน prefilter)", model_patterns, model_risk)

    # --- 2) ระบบจริง (prefilter คัดก่อน) ---
    system_patterns = score_patterns(records, "gt_cumulative", "system_cumulative")
    system_risk = score_risk(records, "system_risk")
    print_report("ระบบจริง (prefilter + โมเดล)", system_patterns, system_risk)

    # --- 3) ระดับบทสนทนา ---
    per_cat = score_conversations(records)
    print(f"\n{'=' * 66}")
    print("  ระดับบทสนทนา — คะแนนสุดท้ายคลาดเฉลี่ย")
    print("=" * 66)
    for cat, v in per_cat.items():
        print(f"  {cat:<12} {v['conversations']:>4} บท   MAE = {v['final_score_mae']} จุด")

    # --- 4) pattern ที่สับสนกันบ่อย ---
    print(f"\n{'=' * 66}")
    print("  โมเดลพลาด pattern ไหน แล้วตอบอะไรแทน (5 อันดับแรก)")
    print("=" * 66)
    pairs = [
        (gt, wrong, n)
        for gt, wrongs in model_patterns["confusion"].items()
        for wrong, n in wrongs.items()
    ]
    if pairs:
        for gt, wrong, n in sorted(pairs, key=lambda x: -x[2])[:5]:
            print(f"  เฉลย {gt} → โมเดลตอบ {wrong} แทน  ({n} ครั้ง)")
    else:
        print("  (ไม่มี)")

    # --- บันทึกไฟล์ ---
    report = {
        "summary": {
            "turns": len(records),
            "conversations": conversations,
            "model_errors": errors,
            "parse_failures": parse_fails,
        },
        "model_only": {"patterns": model_patterns, "risk": model_risk},
        "full_system": {"patterns": system_patterns, "risk": system_risk},
        "by_category": per_cat,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 บันทึกรายงานที่: {args.output}\n")


if __name__ == "__main__":
    main()
