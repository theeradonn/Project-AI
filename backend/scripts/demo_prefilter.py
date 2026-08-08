"""
SafeTrade — ตัวอย่างการใช้งาน Pre-filter ก่อนเรียก Qwen ตัวหลัก

สาธิตว่า pre-filter ทำงานอย่างไร และแสดง "จุดที่ integrate เข้า backend จริง"
(ของจริงอยู่ใน supabase_listener.py::_process_room — ไฟล์นี้เป็นตัวอย่างแบบรันดูได้)

รันคำสั่ง:
    python demo_prefilter.py                 # ใช้ keyword อย่างเดียว (ไม่ต้องมี Ollama)
    python demo_prefilter.py --embedding     # เปิดด่าน 2 ด้วย (ต้องมี Evidence Bank + Ollama)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows console ใช้ codepage ไทย (cp874) ซึ่ง encode emoji ไม่ได้ — บังคับเป็น UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from prefilter import should_escalate  # noqa: E402

# บทสนทนาตัวอย่าง — ผสมข้อความปกติกับข้อความเสี่ยง
DEMO_CONVERSATION = [
    {"sender": "buyer", "text": "สวัสดีครับ กล้องตัวนี้ยังอยู่ไหมครับ"},
    {"sender": "seller", "text": "ยังอยู่ครับผม สนใจสอบถามได้เลย"},
    {"sender": "buyer", "text": "สภาพเป็นยังไงบ้างครับ มีตำหนิไหม"},
    {"sender": "seller", "text": "สภาพ 95% ครับ มีรอยขนแมวนิดหน่อยตามการใช้งาน"},
    {"sender": "buyer", "text": "นัดรับได้แถวไหนบ้างครับ อยากเช็คเครื่องก่อน"},
    {"sender": "seller", "text": "ไม่สะดวกนัดเจอครับ ส่งไปรษณีย์อย่างเดียว"},
    {"sender": "buyer", "text": "งั้นเก็บเงินปลายทางได้ไหมครับ"},
    {"sender": "seller", "text": "ไม่รับเก็บปลายทางนะครับ โอนก่อนเท่านั้น"},
    {"sender": "buyer", "text": "โอเคครับ ขอเลขบัญชีหน่อย"},
    {"sender": "seller", "text": "โอนเข้าบัญชีแฟนผมนะครับ นางสาวสมศรี ใจดี"},
    {"sender": "buyer", "text": "โอนละนะ"},
]


def simulate(use_embedding: bool) -> None:
    """เดินทีละ turn แล้วแสดงผลว่า pre-filter ตัดสินอย่างไร"""
    escalate_count = 0

    print("=" * 78)
    print(f"{'turn':>4} {'ผู้พูด':<8} {'ตัดสิน':<10} {'ด่าน':<10} เหตุผล")
    print("=" * 78)

    for i, msg in enumerate(DEMO_CONVERSATION):
        context = DEMO_CONVERSATION[max(0, i - 3) : i]
        decision = should_escalate(msg, context, use_embedding=use_embedding)

        if decision.escalate:
            escalate_count += 1

        verdict = "🚨 วิเคราะห์" if decision.escalate else "⏭️  ข้าม"
        role = "ผู้ซื้อ" if msg["sender"] == "buyer" else "ผู้ขาย"
        print(f"{i + 1:>4} {role:<8} {verdict:<10} {decision.stage:<10} {decision.reason[:44]}")
        print(f"     └─ {msg['text'][:70]}")

    total = len(DEMO_CONVERSATION)
    saved = total - escalate_count
    print("=" * 78)
    print(f"สรุป: เรียก Qwen {escalate_count}/{total} ครั้ง — ประหยัดไป {saved} ครั้ง ({saved / total * 100:.0f}%)")


def integration_example() -> None:
    """แสดงรูปแบบการ integrate เข้ากับ backend (pseudo-code ของของจริง)"""
    print(
        """
==============================================================================
จุดที่ integrate เข้า backend จริง — supabase_listener.py::_process_room()
==============================================================================

    # 1) ดึงข้อความทั้งหมดในห้อง + risk เดิม
    all_messages = fetch_messages(room_id)
    previous_risk = fetch_room_risk(room_id)

    # 2) [ใหม่] ให้ pre-filter ตัดสินก่อนว่าต้องเรียก Qwen ไหม
    decision = _run_prefilter(all_messages, unprocessed_ids)

    if not decision.escalate:
        # ไม่พบสัญญาณเสี่ยง → ไม่เรียก Qwen เลย คง risk_after เดิมไว้
        _mark_skipped(supabase, all_messages, unprocessed_ids, previous_risk)
        return                        # <-- ประหยัด VRAM/เวลาตรงนี้

    # 3) เข้าข่ายเสี่ยง → ทำงานแบบเดิมทุกอย่าง (ส่ง full history ให้ Qwen)
    risk_result = analyze_risk(all_messages)
    final_risk = max(risk_result["risk_percentage"], previous_risk)
    ...

หมายเหตุด้านความปลอดภัยของดีไซน์:
Qwen วิเคราะห์ full history ใหม่ทุกครั้งที่ escalate ดังนั้นข้อความที่เคยถูกข้าม
จะถูกนำกลับมาวิเคราะห์ด้วยเสมอในรอบถัดไป — การข้ามจึงไม่ทำให้หลักฐานหายถาวร
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="สาธิต pre-filter")
    parser.add_argument("--embedding", action="store_true", help="เปิดด่าน 2 (ต้องมี Ollama + Evidence Bank)")
    args = parser.parse_args()

    simulate(use_embedding=args.embedding)
    integration_example()


if __name__ == "__main__":
    main()
