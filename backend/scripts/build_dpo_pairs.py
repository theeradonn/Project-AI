"""
SafeTrade — Phase 4: สร้าง DPO pairs

แปลงผล blind test ให้เป็นคู่ (prompt, chosen, rejected) สำหรับ DPO fine-tuning
ไม่เรียกโมเดลเลย รันบนเครื่องธรรมดาได้ ไม่ต้องใช้ GPU

หัวใจของสคริปต์นี้ — normalize ทั้งสองฝั่งด้วย serializer ตัวเดียวกัน:
DPO ผลักความน่าจะเป็นออกจาก "ทุก" ความต่างระดับ token ระหว่าง chosen กับ rejected
ถ้าปล่อยให้ chosen เป็นคำตอบจาก dataset (evidence สั้น) แต่ rejected เป็นข้อความดิบของโมเดล
(evidence ยาว + summary คนละสำนวน) สิ่งที่โมเดลจะเรียนคือ "evidence สั้นดีกว่า" ไม่ใช่ทักษะ
ตรวจจับ pattern — loss ลงสวยแต่ F1 ไม่ขยับ
จึง render ทั้งสองฝั่งผ่าน render_answer() ตัวเดียวกัน + copy evidence ของ pattern ที่ตรงกัน
ให้เหมือนกันแบบ byte-identical เหลือความต่างเฉพาะชุด pattern ที่เถียงกันจริงๆ

เฉลยแบบสะสม:
ground_truth.patterns ใน blind_test_results.json เก็บเฉพาะ pattern ที่ "เพิ่งโผล่ turn นั้น"
แต่โมเดลเห็น prefix ทั้งหมดจึงตอบแบบสะสม — ต้องประกอบเฉลยสะสมจาก evidence[] ของ turn 1..n
(ดู build_cumulative_view() ใน grade_results.py ที่แก้ปัญหาเดียวกัน)

รันคำสั่ง:
    python build_dpo_pairs.py --limit 20      # ทดสอบเล็กก่อน แล้วเปิด jsonl อ่านด้วยตา
    python build_dpo_pairs.py                 # สร้างเต็ม
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Windows console ใช้ codepage ไทย (cp874) ซึ่ง encode emoji ไม่ได้ — บังคับเป็น UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from ollama_client import PATTERN_CATALOGUE, SYSTEM_PROMPT  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_RESULTS = OUTPUT_DIR / "blind_test_results.json"
DEFAULT_DATASET = OUTPUT_DIR / "dataset_converted.json"
DEFAULT_OUTDIR = OUTPUT_DIR / "dpo"

# ลำดับ pattern ตาม catalogue — ใช้เรียง matched_patterns ให้เหมือนกันทั้งสองฝั่ง
# (ฝั่งโมเดลเรียงตามที่มันตอบมา ถ้าไม่จัดใหม่จะกลายเป็นความต่างที่ DPO ไปเรียนแทน)
CATALOGUE_ORDER = {pid: i for i, pid in enumerate(PATTERN_CATALOGUE)}

# ป้ายสั้นสำหรับเขียน summary — label เต็มใน catalogue ยาวเกินกติกา "ไม่เกิน 20 คำ" ใน SYSTEM_PROMPT
SHORT_LABEL = {
    "A1": "ไม่ยอมให้ตรวจสินค้าก่อนจ่าย",
    "A2": "ให้โอนเข้าบัญชีชื่อไม่ตรงโปรไฟล์",
    "A3": "ชวนย้ายไปคุยนอกแพลตฟอร์ม",
    "B1": "ขึ้นราคาทีหลังอย่างไม่สมเหตุผล",
    "B2": "เลี่ยงการนัดรับสินค้า",
    "B3": "ปฏิเสธเก็บเงินปลายทาง",
    "B4": "เร่งรัดให้รีบโอน",
    "C1": "เปลี่ยนเงื่อนไขกะทันหัน",
    "C2": "เลี่ยงคำถามเรื่องสภาพสินค้า",
    "C3": "อ้างเหตุด่วนบีบให้ตัดสินใจ",
    "C4": "ลดราคาแลกกับการโอนก่อน",
    "D1": "ไม่ยอมส่งรูปเพิ่ม",
    "D2": "ใช้ภาษาซ้ำแบบสคริปต์",
    "D3": "อ้างอยู่ไกลขอส่งขนส่งแทน",
}

# สำนวน summary หลายแบบ — กันโมเดลจำประโยคเดียวแล้วพ่นออกมาทุกครั้งแทนที่จะเรียนการตรวจจับ
SUMMARY_TEMPLATES = [
    "พบพฤติกรรมเสี่ยง: {body}",
    "ผู้ขาย{body}",
    "สัญญาณเสี่ยง — {body}",
    "ข้อความผู้ขาย{body}",
]

# ข้อความที่ SYSTEM_PROMPT กำหนดไว้ตรงๆ สำหรับกรณีไม่พบ pattern
EMPTY_SUMMARY = "ไม่พบพฤติกรรมที่น่าสงสัย"

# pattern ที่โมเดลมั่วบ่อยที่สุด (false positive สูงสุดจาก grading) — ใช้เป็น distractor ใน anchor pair
# ต้องอัปเดตตาม grading_report ล่าสุดเสมอ: ห้ามใส่ pattern ที่ precision สูงแต่ recall ต่ำ (เช่น A1, B2)
# เพราะจะสอนให้โมเดลกดตัวที่มันกล้าตอบน้อยอยู่แล้วให้เงียบลงไปอีก
TOP_FP_PATTERNS = ["C4", "D1", "D3", "B3", "B4"]

# ผล baseline ล่าสุดทายต่ำกว่าเฉลยมากกว่าทายเกิน (32% vs 9%) — anchor ส่วนใหญ่จึงควรมาจาก record
# ที่มี pattern จริง (สอนไม่ให้มองข้าม) ไม่ใช่ record ว่าง (สอนให้ยับยั้ง)
ANCHOR_WITH_GT_SHARE = 0.7


def stable_choice(seq: list, salt: str):
    """เลือกสมาชิกแบบสุ่มที่ให้ผลเดิมทุกครั้ง (อิง hash ของ salt ไม่ใช่ random state)"""
    idx = int(hashlib.sha1(salt.encode("utf-8")).hexdigest(), 16) % len(seq)
    return seq[idx]


def make_summary(pattern_ids: list[str], salt: str) -> str:
    """
    เขียน summary จากชุด pattern — ใช้ตัวเดียวกันทั้ง chosen และ rejected

    เอาเฉพาะ 3 pattern ที่คะแนนสูงสุด ไม่งั้นบทที่เจอ 7 pattern จะได้ summary ยาว 300 ตัวอักษร
    ซึ่งผิดกติกา "ไม่เกิน 20 คำ" ที่เขียนไว้ใน SYSTEM_PROMPT เอง

    Args:
        pattern_ids: pattern ที่ตรวจพบ
        salt: ข้อความกำกับ (conversation_id#turn) ใช้เลือกสำนวนแบบ deterministic

    Returns:
        ข้อความ summary ภาษาไทย 1 ประโยค
    """
    if not pattern_ids:
        return EMPTY_SUMMARY

    top = sorted(pattern_ids, key=lambda p: -PATTERN_CATALOGUE[p]["base_score"])[:3]
    body = " ".join(SHORT_LABEL[pid] for pid in top)
    return stable_choice(SUMMARY_TEMPLATES, salt).format(body=body)


def render_answer(items: list[tuple[str, str]], salt: str) -> str:
    """
    ประกอบคำตอบ JSON ตาม schema ที่ SYSTEM_PROMPT กำหนด

    ทั้ง chosen และ rejected ต้องผ่านฟังก์ชันนี้เท่านั้น — key order, การเว้นวรรค,
    base_score (เอาจาก catalogue เสมอ ไม่เชื่อตัวเลขที่โมเดลตอบ) และสำนวน summary
    จะได้เหมือนกันหมด เหลือความต่างเฉพาะ pattern ที่เถียงกัน

    Args:
        items: [(pattern_id, evidence), ...] จะถูกเรียงตามลำดับ catalogue ให้เอง
        salt: ใช้เลือกสำนวน summary

    Returns:
        JSON string บรรทัดเดียว
    """
    ordered = sorted(items, key=lambda it: CATALOGUE_ORDER[it[0]])
    payload = {
        "matched_patterns": [
            {
                "pattern_id": pid,
                "evidence": evidence,
                "base_score": PATTERN_CATALOGUE[pid]["base_score"],
            }
            for pid, evidence in ordered
        ],
        "summary": make_summary([pid for pid, _ in ordered], salt),
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_detected_flags(flags: list[str]) -> list[tuple[str, str]]:
    """
    ถอด detected_flags กลับเป็น (pattern_id, evidence)

    รูปแบบที่ _validate_risk_data() สร้าง: "[B3] <label> (+20): <evidence>"
    ประกอบ prefix ขึ้นมาจาก catalogue แล้วตัดตามความยาว — ไม่ใช้ split(": ") เพราะ
    evidence เองก็มี ": " ได้ (เช่นข้อความผู้ขายที่มีเครื่องหมายโคลอน)

    Args:
        flags: list ของ detected_flags จากผล blind test

    Returns:
        [(pattern_id, evidence), ...] ตัวที่ parse ไม่ได้จะถูกข้าม
    """
    items: list[tuple[str, str]] = []
    seen: set[str] = set()

    for flag in flags or []:
        text = str(flag)
        for pid, entry in PATTERN_CATALOGUE.items():
            prefix = f"[{pid}] {entry['label']} (+{entry['base_score']}): "
            if text.startswith(prefix):
                if pid not in seen:
                    seen.add(pid)
                    items.append((pid, text[len(prefix):]))
                break

    return items


def cumulative_evidence(conv: dict, turn: int) -> list[tuple[str, str]]:
    """
    เฉลยแบบสะสมถึง turn ที่กำหนด — pattern ละครั้งเดียว ยึด evidence ที่โผล่ครั้งแรก

    Args:
        conv: บทสนทนาจาก dataset_converted.json
        turn: turn ปัจจุบัน (นับรวม turn ก่อนหน้าทั้งหมด)

    Returns:
        [(pattern_id, evidence), ...] เรียงตาม turn ที่พบ
    """
    items: list[tuple[str, str]] = []
    seen: set[str] = set()

    for ev in sorted(conv.get("evidence", []), key=lambda e: e["turn"]):
        if ev["turn"] <= turn and ev["pattern"] not in seen:
            seen.add(ev["pattern"])
            items.append((ev["pattern"], ev["text"]))

    return items


def seller_lines_upto(conv: dict, turn: int) -> list[str]:
    """ข้อความผู้ขายทั้งหมดถึง turn นี้ — ใช้เป็น evidence ปลอมตอนสร้าง distractor"""
    return [
        m["text"]
        for m in conv["messages"]
        if m["role"] == "seller" and m["turn"] <= turn
    ]


def make_anchor_rejected(
    gt_items: list[tuple[str, str]], conv: dict, turn: int, salt: str
) -> tuple[str, str] | None:
    """
    สร้างคำตอบผิดแบบตั้งใจ สำหรับ record ที่โมเดลตอบถูกอยู่แล้ว

    ทำไมต้องมี: ถ้าเทรนเฉพาะ record ที่โมเดลตอบผิด โมเดลจะไม่เห็นตัวอย่างที่ตัวเอง "ทำถูก"
    เลยตลอดการเทรน เสี่ยงลืมของเดิม (catastrophic forgetting) และเอนไปทางทายเยอะไว้ก่อน
    anchor pair เอา record เหล่านั้นกลับเข้ามาโดยปลอม rejected ขึ้นมาแทน

    - drop       : ตัด pattern จริงออก 1 ตัว → สอนไม่ให้มองข้าม
    - distractor : เติม pattern ที่โมเดลมั่วบ่อย 1 ตัว → สอนไม่ให้ทายเกิน

    Args:
        gt_items: เฉลยสะสมของ record นี้
        conv: บทสนทนาต้นทาง (ใช้ดึงข้อความผู้ขายมาทำ evidence ปลอม)
        turn: turn ปัจจุบัน
        salt: ใช้เลือกแบบ deterministic

    Returns:
        (rejected_json, kind) หรือ None ถ้าสร้างไม่ได้
    """
    gt_ids = {pid for pid, _ in gt_items}
    candidates = [pid for pid in TOP_FP_PATTERNS if pid not in gt_ids]

    # ตัดออก 1 ตัว — ทำได้เฉพาะตอนมีเฉลยอยู่แล้ว
    if gt_items and stable_choice([True, False, True], salt + "#mode"):
        drop = stable_choice(gt_items, salt + "#drop")
        remaining = [it for it in gt_items if it[0] != drop[0]]
        return render_answer(remaining, salt), "anchor_drop"

    # เติม pattern มั่ว 1 ตัว พร้อม evidence ที่ยกมาจากข้อความผู้ขายจริง
    lines = seller_lines_upto(conv, turn)
    if not candidates or not lines:
        return None

    fake_pid = stable_choice(candidates, salt + "#fp")
    fake_evidence = stable_choice(lines, salt + "#line")
    kind = "anchor_hallucinate" if not gt_items else "anchor_distractor"
    return render_answer(gt_items + [(fake_pid, fake_evidence)], salt), kind


def split_conversations(
    convs: list[dict], frac: float, seed: int
) -> tuple[list[str], list[str]]:
    """
    แบ่ง train/holdout ที่ระดับบทสนทนา

    ห้ามแบ่งที่ระดับ record เด็ดขาด — prompt ของ turn 8 มีข้อความ turn 1-7 อยู่ครบแบบคำต่อคำ
    ถ้าแบ่งราย record บทเดียวกันจะไปโผล่ทั้งสองฝั่ง = ข้อสอบรั่ว ตัวเลขหลัง fine-tune เชื่อไม่ได้

    stratify ด้วย (category, มี D2/B1 ไหม) เพราะ D2 มีอยู่แค่ ~21 บท ถ้าสุ่มเฉยๆ อาจไม่ติด
    holdout เลยจนวัดผลไม่ได้

    Args:
        convs: บทสนทนาทั้งหมด
        frac: สัดส่วน holdout
        seed: กำหนดให้ผลซ้ำได้

    Returns:
        (train_ids, holdout_ids)
    """
    rng = random.Random(seed)
    buckets: dict[tuple, list[str]] = defaultdict(list)

    for conv in convs:
        pats = {ev["pattern"] for ev in conv.get("evidence", [])}
        tier = "D2" if "D2" in pats else ("B1" if "B1" in pats else "-")
        buckets[(conv["category"], tier)].append(conv["conversation_id"])

    train: list[str] = []
    holdout: list[str] = []
    for key in sorted(buckets):
        ids = sorted(buckets[key])
        rng.shuffle(ids)
        n = max(1, round(len(ids) * frac))  # ทุกกลุ่มต้องมีอย่างน้อย 1 บทใน holdout
        holdout += ids[:n]
        train += ids[n:]

    return sorted(train), sorted(holdout)


def build_pairs(
    records: list[dict],
    conv_by_id: dict[str, dict],
    train_ids: set[str],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], dict]:
    """
    สร้าง pair ทั้งหมดจากผล blind test

    Returns:
        (train_pairs, eval_pairs, stats)
    """
    rare = {p.strip().upper() for p in args.rare_patterns.split(",") if p.strip()}
    train_pairs: list[dict] = []
    eval_pairs: list[dict] = []
    anchor_candidates: list[dict] = []

    stats = {
        "kind": Counter(),
        "fn_pattern": Counter(),
        "fp_pattern": Counter(),
        "turn": Counter(),
        "risk_mismatch": 0,
        "flag_parse_dropped": 0,
    }

    for rec in records:
        conv = conv_by_id.get(rec["conversation_id"])
        if conv is None:
            continue

        turn = rec["turn"]
        salt = f"{rec['conversation_id']}#{turn}"

        gt_items = cumulative_evidence(conv, turn)
        gt_ids = {pid for pid, _ in gt_items}
        gt_evidence = dict(gt_items)

        # ตรวจความถูกต้องของการ join — คะแนนสะสมที่ประกอบเองต้องตรงกับเฉลยใน dataset
        expected_risk = min(sum(PATTERN_CATALOGUE[p]["base_score"] for p in gt_ids), 100)
        if rec["ground_truth"]["risk_after"] != expected_risk:
            stats["risk_mismatch"] += 1

        model_items = parse_detected_flags(rec["model"]["detected_flags"])
        if len(model_items) != len(rec["model"]["detected_flags"] or []):
            stats["flag_parse_dropped"] += 1
        model_ids = {pid for pid, _ in model_items}

        chosen = render_answer(gt_items, salt)

        base = {
            "id": salt,
            "conversation_id": rec["conversation_id"],
            "turn": turn,
            "category": rec["category"],
            "user": rec["prompt"],
            "chosen": chosen,
        }

        if gt_ids == model_ids:
            # โมเดลตอบถูก — เก็บไว้เป็นตัวเลือกสำหรับ anchor pair
            anchor_candidates.append({**base, "_gt_items": gt_items, "_conv": conv})
            continue

        # pattern ที่ตรงกันทั้งสองฝั่ง ให้ใช้ evidence ชุดเดียวกัน จะได้เหลือความต่าง
        # เฉพาะ pattern ที่เถียงกันจริงๆ ไม่ใช่สำนวนการยกข้อความ
        rejected_items = [
            (pid, gt_evidence.get(pid, evidence)) for pid, evidence in model_items
        ]
        rejected = render_answer(rejected_items, salt)
        if rejected == chosen:
            continue

        kind = (
            "chosen_empty" if not gt_ids
            else "model_empty" if not model_ids
            else "partial"
        )
        missed = gt_ids - model_ids
        stats["kind"][kind] += 1
        stats["turn"][turn] += 1
        for pid in missed:
            stats["fn_pattern"][pid] += 1
        for pid in model_ids - gt_ids:
            stats["fp_pattern"][pid] += 1

        pair = {
            **base,
            "rejected": rejected,
            "meta": {
                "gt_cumulative": sorted(gt_ids),
                "model_patterns": sorted(model_ids),
                "risk_gt": rec["ground_truth"]["risk_after"],
                "risk_model": rec["model"]["risk_percentage"],
                "pair_kind": kind,
                "rare_boost": bool(missed & rare),
            },
        }

        if rec["conversation_id"] in train_ids:
            train_pairs.append(pair)
            # pattern ที่หายากและโมเดลไม่เคยจับได้ ให้ใส่ซ้ำเพื่อเพิ่มน้ำหนัก
            for _ in range(args.rare_boost - 1 if missed & rare else 0):
                train_pairs.append(pair)
        else:
            eval_pairs.append(pair)

    # --- anchor pairs: เอา record ที่โมเดลตอบถูกกลับเข้ามาเทรนด้วย ---
    anchor_train = [a for a in anchor_candidates if a["conversation_id"] in train_ids]
    target = int(len(train_pairs) * args.anchor_frac)
    rng = random.Random(args.seed)

    # record ที่โมเดลตอบถูกส่วนใหญ่เป็น turn ว่าง (บทปกติ/ต้นบท) ถ้าสุ่มรวมกัน anchor จะกลายเป็น
    # "สอนให้ยับยั้ง" เกือบทั้งหมด — แยกกองแล้วให้กองที่มี pattern จริงได้โควตาก่อน
    with_gt = [a for a in anchor_train if a["_gt_items"]]
    without_gt = [a for a in anchor_train if not a["_gt_items"]]
    rng.shuffle(with_gt)
    rng.shuffle(without_gt)
    quota = int(target * ANCHOR_WITH_GT_SHARE)
    ordered = with_gt[:quota] + without_gt + with_gt[quota:]

    for cand in ordered:
        if target <= 0:
            break
        made = make_anchor_rejected(
            cand.pop("_gt_items"), cand.pop("_conv"), cand["turn"], cand["id"]
        )
        if made is None:
            continue
        rejected, kind = made
        if rejected == cand["chosen"]:
            continue

        stats["kind"][kind] += 1
        stats["turn"][cand["turn"]] += 1
        train_pairs.append({
            **cand,
            "rejected": rejected,
            "meta": {"pair_kind": kind, "rare_boost": False},
        })
        target -= 1

    return train_pairs, eval_pairs, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """เขียน JSONL — หนึ่ง object ต่อบรรทัด"""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="สร้าง DPO pairs จากผล blind test")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--limit", type=int, default=None, help="จำกัดจำนวนบทสนทนา (ใช้ทดสอบ)")
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--rare-boost", type=int, default=2, help="ใส่ pair ของ pattern หายากซ้ำกี่รอบ")
    parser.add_argument(
        "--rare-patterns",
        default="C2,D2",
        help="pattern ที่ recall ต่ำสุดใน grading ล่าสุด คั่นด้วย comma (อัปเดตทุกครั้งที่ re-baseline)",
    )
    parser.add_argument("--anchor-frac", type=float, default=0.2, help="สัดส่วน anchor pair ต่อ pair หลัก")
    parser.add_argument("--seed", type=int, default=20250921)
    args = parser.parse_args()

    for path, hint in ((args.results, "blind_test.py"), (args.dataset, "convert_dataset.py")):
        if not path.exists():
            print(f"❌ ไม่พบไฟล์ {path} — รัน {hint} ก่อน")
            return

    with open(args.results, encoding="utf-8") as f:
        records = json.load(f)
    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)

    if args.limit:
        keep = {c["conversation_id"] for c in dataset[: args.limit]}
        dataset = [c for c in dataset if c["conversation_id"] in keep]
        records = [r for r in records if r["conversation_id"] in keep]

    conv_by_id = {c["conversation_id"]: c for c in dataset}
    print(f"📂 อ่าน {len(records)} turn จาก {len(dataset)} บทสนทนา")

    train_ids, holdout_ids = split_conversations(dataset, args.holdout_frac, args.seed)
    print(f"✂️  แบ่งบทสนทนา: train {len(train_ids)} / holdout {len(holdout_ids)}")

    train_pairs, eval_pairs, stats = build_pairs(
        records, conv_by_id, set(train_ids), args
    )

    if stats["risk_mismatch"]:
        print(
            f"❌ เฉลยสะสมไม่ตรงกับ risk_after {stats['risk_mismatch']} record — "
            f"dataset กับผล blind test อาจคนละเวอร์ชัน ตรวจก่อนเอาไปเทรน"
        )
        return
    if stats["flag_parse_dropped"]:
        print(f"⚠️ parse detected_flags ไม่ครบใน {stats['flag_parse_dropped']} record")

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.outdir / "dpo_pairs_train.jsonl", train_pairs)
    write_jsonl(args.outdir / "dpo_pairs_eval.jsonl", eval_pairs)

    # เขียน system prompt แยกไฟล์ครั้งเดียว (10.5 KB × จำนวน pair จะกลายเป็นไฟล์หลายสิบ MB)
    system_path = args.outdir / "system_prompt.txt"
    system_path.write_text(SYSTEM_PROMPT, encoding="utf-8")
    system_sha = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

    holdout_patterns = Counter(
        ev["pattern"]
        for cid in holdout_ids
        for ev in conv_by_id[cid].get("evidence", [])
    )
    with open(args.outdir / "holdout_conversations.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "holdout_frac": args.holdout_frac,
                "holdout": holdout_ids,
                "train": train_ids,
                "holdout_category_counts": dict(
                    Counter(conv_by_id[c]["category"] for c in holdout_ids)
                ),
                "holdout_pattern_coverage": dict(sorted(holdout_patterns.items())),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    manifest = {
        "system_sha256": system_sha,
        "system_prompt_chars": len(SYSTEM_PROMPT),
        "seed": args.seed,
        "flags": {
            "holdout_frac": args.holdout_frac,
            "rare_boost": args.rare_boost,
            "rare_patterns": args.rare_patterns,
            "anchor_frac": args.anchor_frac,
        },
        "counts": {
            "train_pairs": len(train_pairs),
            "eval_pairs": len(eval_pairs),
            "by_kind": dict(stats["kind"]),
        },
        "fn_by_pattern": dict(sorted(stats["fn_pattern"].items())),
        "fp_by_pattern": dict(sorted(stats["fp_pattern"].items())),
        "turn_histogram": dict(sorted(stats["turn"].items())),
    }
    with open(args.outdir / "dpo_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n📊 train {len(train_pairs)} pair | eval {len(eval_pairs)} pair")
    for kind, n in stats["kind"].most_common():
        print(f"   {kind:<20} {n:>5}")
    print(f"\n   FN สูงสุด: {stats['fn_pattern'].most_common(5)}")
    print(f"   FP สูงสุด: {stats['fp_pattern'].most_common(5)}")
    print(f"\n💾 บันทึกที่: {args.outdir}")
    print("ขั้นต่อไป: อ่าน dpo_pairs_train.jsonl ด้วยตาสัก 2-3 บรรทัด")
    print("          ว่า chosen/rejected ต่างกันเฉพาะชุด pattern จริงๆ แล้วค่อยเอาขึ้น H100")


if __name__ == "__main__":
    main()
