"""
SafeTrade — Unit tests สำหรับ Pre-filter

รันแบบไม่ต้องมี Ollama/Evidence Bank โดยใช้ use_embedding=False
(ทดสอบเฉพาะด่านที่ 1 = keyword ซึ่งเป็นด่านที่ทำงานเสมอ)

รันคำสั่ง:
    python tests/test_prefilter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from fraud_keywords import FRAUD_KEYWORDS, match_keywords, normalize_text  # noqa: E402
from ollama_client import PATTERN_CATALOGUE  # noqa: E402
from prefilter import should_escalate  # noqa: E402


def seller(text: str) -> dict:
    return {"sender": "seller", "text": text}


def buyer(text: str) -> dict:
    return {"sender": "buyer", "text": text}


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    return condition


results: list[bool] = []

# ==========================================
print("\n=== 1. keyword catalogue สอดคล้องกับ PATTERN_CATALOGUE ===")
# ==========================================
unknown = set(FRAUD_KEYWORDS) - set(PATTERN_CATALOGUE)
results.append(check("ไม่มี pattern_id แปลกปลอมใน keyword list", not unknown, f"เกิน: {unknown}"))

missing = set(PATTERN_CATALOGUE) - set(FRAUD_KEYWORDS)
results.append(check("มี keyword ครบทุก pattern", not missing, f"ขาด: {missing}"))

results.append(
    check("ทุก pattern มี keyword อย่างน้อย 1 คำ", all(len(v) > 0 for v in FRAUD_KEYWORDS.values()))
)

# ==========================================
print("\n=== 2. keyword matching พื้นฐาน ===")
# ==========================================
results.append(check("จับ B3 (ปฏิเสธ COD)", "B3" in match_keywords("ไม่รับเก็บปลายทางนะคะ")))
results.append(check("จับ A2 (บัญชีชื่อไม่ตรง)", "A2" in match_keywords("โอนเข้าบัญชีแฟนผมนะครับ")))
results.append(check("จับ A3 (ย้ายช่องทาง)", "A3" in match_keywords("แอดไลน์มาคุยต่อนะ")))
results.append(check("จับ B4 (เร่งรัด)", "B4" in match_keywords("รีบโอนเลยนะ เดี๋ยวของหมด")))
results.append(check("ข้อความปกติไม่ match อะไร", match_keywords("สภาพดีมากครับ ใช้งานปกติ") == {}))

# ==========================================
print("\n=== 3. normalize — ทนต่อการเว้นวรรค/ตัวพิมพ์ ===")
# ==========================================
results.append(check("เว้นวรรคแทรกกลางยังจับได้", "B3" in match_keywords("ไม่ รับ ปลายทาง นะ")))
results.append(check("ตัวพิมพ์ใหญ่ยังจับได้ (LINE)", "A3" in match_keywords("ทักมา LINE ID: shop99")))
results.append(check("normalize ตัดช่องว่าง", normalize_text("ไม่ รับ COD") == "ไม่รับcod"))

# ==========================================
print("\n=== 4. Edge case: ข้อความสั้นมาก / ว่างเปล่า ===")
# ==========================================
results.append(check("ข้อความว่างไม่ match", match_keywords("") == {}))
results.append(check("ข้อความมีแต่ช่องว่างไม่ match", match_keywords("   ") == {}))

d = should_escalate(seller("ครับ"), use_embedding=False)
results.append(check("ข้อความสั้นมาก ('ครับ') ไม่ escalate", not d.escalate, d.reason))

d = should_escalate(seller(""), use_embedding=False)
results.append(check("ข้อความว่างเปล่าไม่ escalate", not d.escalate, d.reason))

d = should_escalate(buyer("มีปลายทางไหมครับ"), use_embedding=False)
results.append(
    check(
        "ข้อความผู้ซื้อล้วนไม่ escalate (pattern เป็น seller-only)",
        not d.escalate and d.stage == "none",
        f"stage={d.stage}",
    )
)

# ==========================================
print("\n=== 5. Edge case: ข้อความปฏิเสธที่มี keyword เสี่ยงปนอยู่ ===")
# ==========================================
# ผู้ขาย "รับ" ปลายทาง = ปลอดภัย ต้องไม่ match B3 (ซึ่ง keyword คือ 'ไม่รับปลายทาง')
d = should_escalate(seller("รับเก็บเงินปลายทางได้เลยครับ"), use_embedding=False)
results.append(check("'รับเก็บเงินปลายทางได้' ไม่ escalate", not d.escalate, f"{d.matched_keywords}"))

d = should_escalate(seller("นัดรับได้ครับ สะดวกแถวไหนบอกได้เลย"), use_embedding=False)
results.append(check("'นัดรับได้' ไม่ escalate", not d.escalate, f"{d.matched_keywords}"))

d = should_escalate(seller("ยินดีให้ตรวจสอบสินค้าก่อนโอนครับ"), use_embedding=False)
results.append(check("'ยินดีให้ตรวจสอบ' ไม่ escalate", not d.escalate, f"{d.matched_keywords}"))

d = should_escalate(seller("ผ่านคนกลางได้เลยครับ ปลอดภัยทั้งสองฝ่าย"), use_embedding=False)
results.append(check("'ผ่านคนกลางได้' ไม่ escalate", not d.escalate, f"{d.matched_keywords}"))

# ==========================================
print("\n=== 6. บริบทย้อนหลัง (เคสหลักที่ต้องจับให้ได้) ===")
# ==========================================
# ข้อความปัจจุบันดูปกติมาก แต่ turn ก่อนหน้าของผู้ขายอันตราย
d = should_escalate(
    current_message=buyer("โอนละนะ"),
    recent_messages=[seller("โอนตรงเข้าบัญชีนี้แทนนะ"), buyer("ได้ครับ")],
    use_embedding=False,
)
results.append(
    check(
        "จับสัญญาณจาก turn ก่อนหน้าได้ ('โอนละนะ' + บริบทบัญชีม้า)",
        d.escalate and "A2" in d.matched_patterns,
        f"escalate={d.escalate} patterns={d.matched_patterns}",
    )
)

# ถ้าไม่มีบริบท ข้อความเดียวกันต้องไม่ escalate
d = should_escalate(current_message=buyer("โอนละนะ"), use_embedding=False)
results.append(check("'โอนละนะ' เดี่ยวๆ ไม่ escalate", not d.escalate))

# context_turns=0 ต้องไม่มองย้อนหลัง
d = should_escalate(
    current_message=buyer("โอนละนะ"),
    recent_messages=[seller("โอนตรงเข้าบัญชีนี้แทนนะ")],
    context_turns=0,
    use_embedding=False,
)
results.append(check("context_turns=0 ไม่มองย้อนหลัง", not d.escalate))

# ==========================================
print("\n=== 7. เคสจริงจากฐานข้อมูล/dataset ===")
# ==========================================
real_risky = [
    ("ไม่มีปลายทางจ้า โอนก่อนส่งเท่านั้นน้าตัวเอง", "B3"),
    ("ไม่เอาคนกลางนะ ยุ่งยากค่ะ เสียเวลาแอดมินตรวจสอบ", "A1"),
    ("สนใจแอดไลน์คุยก่อนนะครับ Line: lookpad_shop", "A3"),
    ("อ่อ ไม่สะดวกนัดเจอเลยค่ะ ส่งแฟลชอย่างเดียวน้า", "B2"),
]
for text, expected in real_risky:
    d = should_escalate(seller(text), use_embedding=False)
    results.append(
        check(
            f"escalate + จับ {expected}: {text[:32]}...",
            d.escalate and expected in d.matched_patterns,
            f"got={d.matched_patterns}",
        )
    )

# เคสปลอดภัยจริงจาก conv ที่เคยตรวจสอบแล้ว (D3 = อ้างระยะทาง ควร escalate เพราะเป็น pattern จริง)
d = should_escalate(
    seller("พอดีผมอยู่เชียงใหม่ครับพี่ สะดวกส่งเป็น EMS หรือ Flash Express มากกว่าครับ"),
    use_embedding=False,
)
results.append(check("จับ D3 (อ้างระยะทาง+ส่งขนส่ง)", d.escalate and "D3" in d.matched_patterns))

# ==========================================
print("\n=== 8. โครงสร้างผลลัพธ์ ===")
# ==========================================
d = should_escalate(seller("ไม่รับปลายทางนะ รีบโอนด้วย"), use_embedding=False)
results.append(check("escalate=True เมื่อมีหลาย pattern", d.escalate))
results.append(check("stage เป็น 'keyword'", d.stage == "keyword"))
results.append(check("matched_patterns เรียงลำดับ", d.matched_patterns == sorted(d.matched_patterns)))
results.append(check("มี matched_keywords ประกอบการ debug", bool(d.matched_keywords)))
results.append(check("reason ไม่ว่างเปล่า", bool(d.reason)))

# ==========================================
passed, total = sum(results), len(results)
print(f"\n{'=' * 50}")
print(f"ผลรวม: {passed}/{total} ผ่าน")
print("=" * 50)
sys.exit(0 if passed == total else 1)
