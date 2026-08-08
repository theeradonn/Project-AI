"""
SafeTrade — Supabase Listener Module
เฝ้าดูข้อความใหม่ใน Supabase ด้วยการ Polling แบบ Background Thread

Flow:
  1. Poll ทุก 3 วินาที → query ข้อความที่ processed = false จากทุกห้อง
  2. จัดกลุ่มข้อความตาม room_id
  3. สำหรับแต่ละห้องที่มีข้อความใหม่ → รวบรวมข้อความทั้งหมดเป็น context
  4. ส่ง context ไป Ollama วิเคราะห์
  5. อัปเดต risk score กลับ chatrooms table
  6. Mark ข้อความว่า processed = true
"""

import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone

from supabase import Client

from ollama_client import analyze_risk
from prefilter import DEFAULT_CONTEXT_TURNS, PreFilterDecision, should_escalate

logger = logging.getLogger(__name__)

# Threading controls
_stop_event = threading.Event()
_listener_thread: threading.Thread | None = None


# ==========================================
# Polling Loop (ทำงานใน Background Thread)
# ==========================================
def _poll_loop(supabase: Client, interval: float):
    """
    Background loop ที่ตรวจสอบข้อความใหม่จากทุกห้องแชทเป็นระยะๆ

    ทำงานใน daemon thread — จะหยุดอัตโนมัติเมื่อ main process ปิด
    หรือเมื่อเรียก stop_listener()

    Args:
        supabase: Supabase client instance
        interval: ระยะเวลา polling (วินาที)
    """
    logger.info(
        f"👂 เริ่มเฝ้าดูทุกห้องแชท (polling ทุก {interval} วินาที)"
    )

    while not _stop_event.is_set():
        try:
            had_work = _process_all_rooms(supabase)
            if had_work:
                # หากเพิ่งมีการส่งวิเคราะห์ AI (ซึ่งใช้เวลาคิด 1-3 วินาที)
                # อาจมีข้อความใหม่พิมพ์แทรกเข้ามาระหว่างนั้น ให้หน่วงเวลาแค่ 0.5 วินาทีแล้วเช็กต่อทันที
                _stop_event.wait(0.5)
            else:
                # ถ้าไม่มีข้อความใหม่เลย ให้รอตามรอบปกติ เพื่อประหยัดทรัพยากร
                _stop_event.wait(interval)
        except Exception as e:
            logger.error(f"❌ เกิดข้อผิดพลาดใน polling loop: {e}", exc_info=True)
            _stop_event.wait(interval)


def _process_all_rooms(supabase: Client) -> bool:
    """
    ตรวจสอบและประมวลผลข้อความใหม่จากทุกห้องแชทในครั้งเดียว

    ขั้นตอน:
    1. Query ข้อความ processed = false จากทุกห้อง
    2. จัดกลุ่มข้อความตาม room_id
    3. สำหรับแต่ละห้องที่มีข้อความใหม่ → ส่งวิเคราะห์ + อัปเดต risk score
    """
    # === ขั้นที่ 1: หาข้อความใหม่ที่ยังไม่ได้ประมวลผลจากทุกห้อง ===
    unprocessed_result = (
        supabase.table("messages")
        .select("id, room_id")
        .eq("processed", False)
        .execute()
    )

    unprocessed = unprocessed_result.data
    if not unprocessed:
        return False  # ไม่มีข้อความใหม่จากห้องใดเลย

    # === ขั้นที่ 2: จัดกลุ่ม unprocessed message IDs ตาม room_id ===
    room_unprocessed: dict[str, list[str]] = defaultdict(list)
    for msg in unprocessed:
        room_unprocessed[msg["room_id"]].append(msg["id"])

    logger.info(
        f"💬 พบข้อความใหม่ {len(unprocessed)} ข้อความ "
        f"จาก {len(room_unprocessed)} ห้อง — กำลังวิเคราะห์..."
    )

    # === ขั้นที่ 3: ประมวลผลแต่ละห้องที่มีข้อความใหม่ ===
    for room_id, unprocessed_ids in room_unprocessed.items():
        _process_room(supabase, room_id, unprocessed_ids)

    return True


def _run_prefilter(all_messages: list[dict], unprocessed_ids: list[str]) -> PreFilterDecision:
    """
    รัน pre-filter บน "หน้าต่าง" ที่ประกอบด้วยข้อความใหม่ + บริบทย้อนหลัง

    ต้องมองย้อนหลังด้วย เพราะสัญญาณอันตรายมักอยู่ใน turn ก่อนหน้า เช่น
    ข้อความปัจจุบัน "โอนละนะ" ดูปกติ แต่ turn ก่อนคือ "โอนตรงเข้าบัญชีนี้แทนนะ"

    Args:
        all_messages: ข้อความทั้งหมดในห้อง เรียงเก่า→ใหม่
        unprocessed_ids: id ของข้อความใหม่ที่ยังไม่ได้ประมวลผล

    Returns:
        PreFilterDecision — ถ้า escalate=False จะข้ามการเรียก Qwen
    """
    unprocessed_set = set(unprocessed_ids)
    new_indices = [i for i, m in enumerate(all_messages) if m.get("id") in unprocessed_set]

    # หาไม่เจอข้อความใหม่ (ไม่ควรเกิด) → escalate ไว้ก่อนเพื่อความปลอดภัย
    if not new_indices:
        return PreFilterDecision(
            escalate=True,
            stage="fail_open",
            reason="ระบุข้อความใหม่ในห้องไม่ได้ — วิเคราะห์ไว้ก่อนเพื่อความปลอดภัย",
        )

    start = max(0, new_indices[0] - DEFAULT_CONTEXT_TURNS)
    window = all_messages[start:]

    return should_escalate(
        current_message=window[-1],
        recent_messages=window[:-1],
        context_turns=len(window) - 1,
    )


def _mark_skipped(
    supabase: Client,
    all_messages: list[dict],
    unprocessed_ids: list[str],
    previous_risk: int,
) -> None:
    """
    Mark ข้อความว่าประมวลผลแล้ว โดยไม่เรียก Qwen (กรณี pre-filter ตัดสินว่าไม่ต้องวิเคราะห์)

    คง risk_percentage เดิมของห้องไว้ (risk สะสมขึ้นอย่างเดียว ไม่มีวันลด)
    และไม่แตะตาราง chatrooms เลย เพราะคะแนนห้องไม่เปลี่ยน

    Args:
        supabase: Supabase client
        all_messages: ข้อความทั้งหมดในห้อง (ใช้คำนวณเลข turn)
        unprocessed_ids: id ของข้อความที่ต้อง mark
        previous_risk: risk score ปัจจุบันของห้อง ที่จะคงไว้
    """
    turn_by_id = {msg["id"]: i + 1 for i, msg in enumerate(all_messages)}
    for msg_id in unprocessed_ids:
        (
            supabase.table("messages")
            .update(
                {
                    "processed": True,
                    "risk_percentage": previous_risk,
                    "reasoning": "ข้ามการวิเคราะห์ — pre-filter ไม่พบสัญญาณเสี่ยงในข้อความนี้",
                    "detected_flags": [],
                    "scam_pattern": None,
                    "turn": turn_by_id.get(msg_id),
                }
            )
            .eq("id", msg_id)
            .execute()
        )


def _process_room(supabase: Client, room_id: str, unprocessed_ids: list[str]) -> None:
    """
    ประมวลผลความเสี่ยงสำหรับห้องแชทเดียว

    Args:
        supabase: Supabase client instance
        room_id: ID ของห้องแชทที่จะประมวลผล
        unprocessed_ids: รายการ message ID ที่ยังไม่ได้ประมวลผลในห้องนี้
    """
    try:
        # === ดึง risk score ปัจจุบันของห้อง (ก่อนวิเคราะห์รอบนี้) เพื่อบังคับ score ไม่ให้ลดลง ===
        previous_room = (
            supabase.table("chatrooms")
            .select("risk_percentage")
            .eq("id", room_id)
            .execute()
        )
        previous_risk = 0
        if previous_room.data:
            prev_value = previous_room.data[0].get("risk_percentage", -1)
            if prev_value is not None and prev_value >= 0:
                previous_risk = prev_value

        # === ดึงข้อความทั้งหมดในห้องเป็น context ===
        all_result = (
            supabase.table("messages")
            .select("*")
            .eq("room_id", room_id)
            .order("created_at")
            .execute()
        )

        all_messages = all_result.data
        if not all_messages:
            return

        logger.info(f"📋 [{room_id}] รวบรวมได้ {len(all_messages)} ข้อความ")

        # === Pre-filter: ตัดสินใจก่อนว่าต้องเรียก Qwen ตัวหลักไหม ===
        # ข้อความส่วนใหญ่ในแชทซื้อขายเป็นเรื่องปกติ (ถามราคา ต่อรอง ถามสเปค) ไม่มีสัญญาณเสี่ยงเลย
        # การกรองก่อนช่วยประหยัด VRAM/เวลา เพราะไม่ต้องส่ง full history ให้ qwen3:8b ทุกข้อความ
        decision = _run_prefilter(all_messages, unprocessed_ids)

        if not decision.escalate:
            logger.info(
                f"⏭️ [{room_id}] ข้ามการวิเคราะห์ (pre-filter: {decision.stage}) — "
                f"{decision.reason} | คง risk เดิมที่ {previous_risk}%"
            )
            _mark_skipped(supabase, all_messages, unprocessed_ids, previous_risk)
            return

        logger.info(
            f"🚨 [{room_id}] pre-filter สั่งวิเคราะห์ต่อ (ด่าน: {decision.stage}) — {decision.reason}"
        )

        # === ส่งไปวิเคราะห์ที่ Ollama ===
        risk_result = analyze_risk(all_messages)
        logger.info(
            f"🎯 [{room_id}] ผลการวิเคราะห์: "
            f"Risk={risk_result['risk_percentage']}% | "
            f"Summary={risk_result['summary']}"
        )

        # === บังคับ risk score ไม่ให้ลดลงจาก turn ก่อนหน้า ===
        # (risk_percentage คำนวณแบบ deterministic จากผลรวม base_score ของ pattern ที่ตรวจพบแล้วใน ollama_client.py
        # ไม่มีแนวคิด "Critical flag กระโดด 75+" อีกต่อไป — คะแนนสูงสุดมาจากผลรวม pattern ล้วนๆ)
        final_risk = risk_result["risk_percentage"]
        if final_risk >= 0:
            final_risk = max(final_risk, previous_risk)
            final_risk = min(final_risk, 100)

        # === อัปเดต Risk Score กลับ Supabase ===
        if final_risk >= 0:
            now = datetime.now(timezone.utc).isoformat()
            (
                supabase.table("chatrooms")
                .upsert(
                    {
                        "id": room_id,
                        "risk_percentage": final_risk,
                        "reasoning": risk_result["summary"],
                        "detected_flags": risk_result["detected_flags"],
                        "scam_pattern": risk_result["scam_pattern"],
                        "last_updated": now,
                    }
                )
                .execute()
            )
            logger.info(
                f"✅ [{room_id}] อัปเดต Risk Score สำเร็จ: {final_risk}%"
            )

        # === Mark ข้อความว่า processed พร้อมบันทึกความเสี่ยงสะสม ===
        turn_by_id = {msg["id"]: i + 1 for i, msg in enumerate(all_messages)}
        for msg_id in unprocessed_ids:
            (
                supabase.table("messages")
                .update({
                    "processed": True,
                    "risk_percentage": final_risk,
                    "reasoning": risk_result["summary"],
                    "detected_flags": risk_result["detected_flags"],
                    "scam_pattern": risk_result["scam_pattern"],
                    "turn": turn_by_id.get(msg_id),
                })
                .eq("id", msg_id)
                .execute()
            )

        logger.info(f"📝 [{room_id}] Mark {len(unprocessed_ids)} ข้อความว่าประมวลผลแล้ว")

    except Exception as e:
        logger.error(f"❌ [{room_id}] เกิดข้อผิดพลาดในการประมวลผลห้อง: {e}", exc_info=True)


# ==========================================
# Listener Management
# ==========================================
def start_listener(
    supabase: Client,
    interval: float = 3.0,
):
    """
    เริ่ม background polling thread สำหรับทุกห้องแชท

    Args:
        supabase: Supabase client instance
        interval: ระยะเวลา polling (วินาที, default: 3.0)
    """
    global _listener_thread
    _stop_event.clear()

    _listener_thread = threading.Thread(
        target=_poll_loop,
        args=(supabase, interval),
        daemon=True,  # daemon thread จะปิดอัตโนมัติเมื่อ main process ปิด
        name="supabase-listener",
    )
    _listener_thread.start()
    logger.info("🚀 Background listener เริ่มทำงานแล้ว (เฝ้าดูทุกห้อง)")


def stop_listener():
    """
    หยุด background polling thread อย่างสะอาด
    เรียกใช้ตอน application shutdown
    """
    global _listener_thread
    _stop_event.set()

    if _listener_thread and _listener_thread.is_alive():
        _listener_thread.join(timeout=5)
        logger.info("🛑 Background listener หยุดแล้ว")

    _listener_thread = None
