"""
SafeTrade — Pre-filter System (2 ด่าน)

คัดกรองว่าข้อความใหม่ "ควรส่งให้ Qwen ตัวหลักวิเคราะห์เต็มรูปแบบหรือไม่"
เพื่อไม่ต้องเรียก qwen3:8b กับทุกข้อความ (ประหยัด VRAM/เวลาบน RTX 4060)

    ด่าน 1: Keyword/Regex matching (เร็วมาก ~0.1ms)  → fraud_keywords.py
    ด่าน 2: Embedding similarity vs Evidence Bank    → embedding_client.py
            (ทำงานเฉพาะเมื่อด่าน 1 ไม่ match)

กติกาสำคัญ (bias ไปทาง recall):
- ยอม false positive ดีกว่า false negative — ถ้า filter พลาด ข้อความนั้นจะไม่ถูกประเมินเลย
  ซึ่งกระทบ risk_after ที่ออกแบบให้เพิ่มขึ้นอย่างเดียว (ไม่มีวันลด)
- ถ้า embedding ใช้ไม่ได้ (Ollama ล่ม / ยังไม่ pull โมเดล) → escalate ทุกข้อความ (fail-open)
  ปลอดภัยกว่าปล่อยผ่าน

หมายเหตุเชิงสถาปัตยกรรม:
Qwen วิเคราะห์ full history ใหม่ทุกครั้งอยู่แล้ว ดังนั้นถ้า pre-filter ข้ามข้อความหนึ่งไป
ข้อความนั้นจะยังถูกนำไปวิเคราะห์ในรอบถัดไปที่มีการ escalate อยู่ดี — การข้ามจึงไม่ทำให้
หลักฐานหายถาวร ตราบใดที่มีข้อความหลังจากนั้น escalate สักครั้ง
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from embedding_client import EmbeddingError, embed_texts
from fraud_keywords import match_keywords

logger = logging.getLogger(__name__)

# ==========================================
# Configuration
# ==========================================
EVIDENCE_BANK_PATH = Path(__file__).resolve().parent / "scripts" / "output" / "evidence_bank.json"
EVIDENCE_VECTORS_PATH = Path(__file__).resolve().parent / "scripts" / "output" / "evidence_vectors.npy"

# threshold ต่ำ = escalate บ่อยขึ้น = recall สูงขึ้น (ปลอดภัยกว่า แต่เปลือง compute มากขึ้น)
# ค่า 0.65 มาจากการวัดจริงกับ dataset 500 แชท (turn เสี่ยง 70 vs turn ปกติ 70 ที่ไม่ได้อยู่ใน bank)
# หลังตัด C2 ออกจาก Evidence Bank แล้ว:
#   0.60 -> recall 98.6% / FP 17.1%
#   0.65 -> recall 95.7% / FP  7.1%   <- เลือกค่านี้ (สมดุลดีที่สุด)
#   0.70 -> recall 90.0% / FP  2.9%
# หมายเหตุ: ตัวเลข recall ข้างต้นไม่นับ C2 เพราะตัดออกจาก bank ไปแล้ว (ดู build_evidence_bank.py)
SIMILARITY_THRESHOLD = float(os.getenv("PREFILTER_SIMILARITY_THRESHOLD", "0.65"))
# จำนวน turn ย้อนหลังที่ใช้เป็นบริบท (สัญญาณอันตรายมักอยู่ใน turn ก่อนหน้า)
DEFAULT_CONTEXT_TURNS = 3


@dataclass
class PreFilterDecision:
    """
    ผลการตัดสินใจของ pre-filter

    Attributes:
        escalate: True = ต้องส่งให้ Qwen วิเคราะห์เต็มรูปแบบ / False = ข้ามได้ คง risk_after เดิม
        stage: ด่านที่ตัดสิน ("keyword" | "embedding" | "none" | "fail_open")
        matched_patterns: pattern_id ที่สงสัย (จากด่าน keyword)
        matched_keywords: keyword ที่ match จริง (ไว้ debug/log)
        similarity: cosine similarity สูงสุดที่เจอ (ด่าน embedding)
        similar_evidence: ข้อความใน Evidence Bank ที่ใกล้เคียงที่สุด
        reason: คำอธิบายสั้นๆ ว่าทำไมถึงตัดสินแบบนี้
    """

    escalate: bool
    stage: str
    reason: str
    matched_patterns: list[str] = field(default_factory=list)
    matched_keywords: dict[str, list[str]] = field(default_factory=dict)
    similarity: float | None = None
    similar_evidence: str | None = None


# ==========================================
# Evidence Bank (lazy load + cache)
# ==========================================
_bank_entries: list[dict] | None = None
_bank_vectors = None  # numpy.ndarray (normalized) — โหลดแบบ lazy
_bank_load_failed = False


def _load_evidence_bank() -> tuple[list[dict], object] | None:
    """
    โหลด Evidence Bank + vectors ที่คำนวณไว้ล่วงหน้า (cache ไว้ในหน่วยความจำ)

    Returns:
        (entries, normalized_vectors) หรือ None ถ้าโหลดไม่ได้
        (เช่น ยังไม่ได้รัน build_evidence_bank.py หรือไม่มี numpy)
    """
    global _bank_entries, _bank_vectors, _bank_load_failed

    if _bank_load_failed:
        return None
    if _bank_entries is not None and _bank_vectors is not None:
        return _bank_entries, _bank_vectors

    try:
        import numpy as np

        if not EVIDENCE_BANK_PATH.exists() or not EVIDENCE_VECTORS_PATH.exists():
            logger.warning(
                f"⚠️ ยังไม่มี Evidence Bank ({EVIDENCE_BANK_PATH.name}/{EVIDENCE_VECTORS_PATH.name}) "
                f"— รัน scripts/build_evidence_bank.py ก่อน (ด่าน 2 จะถูกข้าม)"
            )
            _bank_load_failed = True
            return None

        with open(EVIDENCE_BANK_PATH, encoding="utf-8") as f:
            entries = json.load(f)
        vectors = np.load(EVIDENCE_VECTORS_PATH)

        if len(entries) != vectors.shape[0]:
            logger.error(
                f"❌ Evidence Bank ไม่สอดคล้องกัน: entries={len(entries)} vs vectors={vectors.shape[0]} "
                f"— รัน build_evidence_bank.py ใหม่"
            )
            _bank_load_failed = True
            return None

        # normalize ไว้ล่วงหน้า เพื่อให้ cosine similarity = dot product เฉยๆ
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _bank_vectors = vectors / norms
        _bank_entries = entries

        logger.info(f"✅ โหลด Evidence Bank สำเร็จ: {len(entries)} ประโยค")
        return _bank_entries, _bank_vectors

    except ImportError:
        logger.warning("⚠️ ไม่มี numpy — ด่าน embedding จะถูกข้าม (pip install numpy)")
        _bank_load_failed = True
        return None
    except Exception as e:
        logger.error(f"❌ โหลด Evidence Bank ไม่สำเร็จ: {e}")
        _bank_load_failed = True
        return None


# ==========================================
# Helpers
# ==========================================
def _extract_seller_texts(messages: list[dict]) -> list[str]:
    """
    ดึงเฉพาะข้อความของ 'ผู้ขาย' ออกมา

    เหตุผล: PATTERN_CATALOGUE ออกแบบให้จับพฤติกรรมผู้ขายเท่านั้น
    ข้อความผู้ซื้อจึงไม่ต้องเอามาเทียบ keyword/embedding (ลด false positive)
    แต่ยังคงถูกส่งเป็นบริบทให้ Qwen ตอน escalate

    Args:
        messages: รายการข้อความ [{"sender": "buyer"|"seller", "text": "..."}, ...]

    Returns:
        list ของข้อความผู้ขายที่ไม่ว่างเปล่า
    """
    return [
        str(m.get("text", "")).strip()
        for m in messages
        if m.get("sender") == "seller" and str(m.get("text", "")).strip()
    ]


def _max_similarity(texts: list[str]) -> tuple[float, str | None]:
    """
    หา cosine similarity สูงสุดระหว่างข้อความที่ให้มา กับทุกประโยคใน Evidence Bank

    Args:
        texts: ข้อความที่ต้องการเทียบ (ปกติคือข้อความผู้ขายใน window)

    Returns:
        (similarity สูงสุด, ข้อความ evidence ที่ใกล้เคียงที่สุด)
        คืน (-1.0, None) ถ้าเทียบไม่ได้

    Raises:
        EmbeddingError: ถ้าเรียก embedding model ไม่สำเร็จ
    """
    bank = _load_evidence_bank()
    if bank is None or not texts:
        return -1.0, None

    entries, bank_vectors = bank

    import numpy as np

    query_vectors = np.array(embed_texts(texts), dtype=np.float32)
    if query_vectors.size == 0:
        return -1.0, None

    norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    query_vectors = query_vectors / norms

    # (n_query, dim) @ (dim, n_bank) -> (n_query, n_bank)
    sim_matrix = query_vectors @ bank_vectors.T
    best_idx = int(np.argmax(sim_matrix))
    best_score = float(sim_matrix.flat[best_idx])
    best_bank_idx = best_idx % len(entries)

    return best_score, entries[best_bank_idx].get("text")


# ==========================================
# Main API
# ==========================================
def should_escalate(
    current_message: dict,
    recent_messages: list[dict] | None = None,
    *,
    context_turns: int = DEFAULT_CONTEXT_TURNS,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    use_embedding: bool = True,
) -> PreFilterDecision:
    """
    ตัดสินใจว่าข้อความใหม่ควรถูกส่งให้ Qwen วิเคราะห์เต็มรูปแบบหรือไม่

    ทำงาน 2 ด่านตามลำดับ:
      1. Keyword matching — ถ้า match ใดๆ → escalate ทันที (ไม่ต้องเรียก embedding)
      2. Embedding similarity — เทียบกับ Evidence Bank ถ้าเกิน threshold → escalate

    ทั้ง 2 ด่านดูข้อความของ 'ผู้ขาย' ใน window (ข้อความปัจจุบัน + บริบทย้อนหลัง)
    เพราะสัญญาณอันตรายมักอยู่ใน turn ก่อนหน้า เช่น
    ปัจจุบัน "โอนละนะ" (ดูปกติ) แต่ turn ก่อนคือ "โอนตรงเข้าบัญชีนี้แทนนะ"

    Args:
        current_message: ข้อความใหม่ {"sender": "buyer"|"seller", "text": "..."}
        recent_messages: ข้อความย้อนหลัง (เรียงเก่า→ใหม่) ใช้เป็นบริบท
        context_turns: จำนวน turn ย้อนหลังที่จะเอามาพิจารณา
        similarity_threshold: เกณฑ์ cosine similarity ของด่าน 2 (ต่ำ = recall สูง)
        use_embedding: ปิดด่าน 2 ได้ (เช่น ตอนรัน unit test แบบไม่ต้องมี Ollama)

    Returns:
        PreFilterDecision — ดู escalate เป็นหลัก

    Example:
        >>> d = should_escalate({"sender": "seller", "text": "ไม่รับปลายทางนะคะ"})
        >>> d.escalate, d.stage, d.matched_patterns
        (True, 'keyword', ['B3'])
    """
    context = list(recent_messages or [])[-context_turns:] if context_turns > 0 else []
    window = context + [current_message]
    seller_texts = _extract_seller_texts(window)

    # ไม่มีข้อความผู้ขายใน window เลย → ไม่มีอะไรให้จับ pattern (pattern เป็น seller-only)
    if not seller_texts:
        return PreFilterDecision(
            escalate=False,
            stage="none",
            reason="ไม่มีข้อความของผู้ขายใน window — ไม่มีสิ่งที่เข้าข่าย pattern ได้",
        )

    # ---------- ด่าน 1: Keyword ----------
    all_hits: dict[str, list[str]] = {}
    for text in seller_texts:
        for pattern_id, keywords in match_keywords(text).items():
            all_hits.setdefault(pattern_id, []).extend(keywords)

    if all_hits:
        patterns = sorted(all_hits)
        return PreFilterDecision(
            escalate=True,
            stage="keyword",
            reason=f"พบ keyword เข้าข่าย pattern: {', '.join(patterns)}",
            matched_patterns=patterns,
            matched_keywords={k: sorted(set(v)) for k, v in all_hits.items()},
        )

    # ---------- ด่าน 2: Embedding ----------
    if not use_embedding:
        return PreFilterDecision(
            escalate=False,
            stage="keyword",
            reason="ไม่พบ keyword (ปิดด่าน embedding ไว้)",
        )

    try:
        similarity, evidence = _max_similarity(seller_texts)
    except EmbeddingError as e:
        # fail-open: embedding ใช้ไม่ได้ → escalate ไว้ก่อน ปลอดภัยกว่าปล่อยผ่าน
        logger.warning(f"⚠️ ด่าน embedding ล้มเหลว — escalate ไว้ก่อนเพื่อความปลอดภัย: {e}")
        return PreFilterDecision(
            escalate=True,
            stage="fail_open",
            reason=f"embedding ใช้ไม่ได้ ({e}) — escalate เพื่อไม่ให้พลาดข้อความเสี่ยง",
        )

    if similarity < 0:
        # ไม่มี Evidence Bank → ใช้ได้แค่ด่าน keyword
        return PreFilterDecision(
            escalate=False,
            stage="keyword",
            reason="ไม่พบ keyword และยังไม่มี Evidence Bank ให้เทียบ",
        )

    if similarity >= similarity_threshold:
        return PreFilterDecision(
            escalate=True,
            stage="embedding",
            reason=f"คล้ายกับหลักฐานที่เคยพบ (similarity={similarity:.3f} ≥ {similarity_threshold})",
            similarity=similarity,
            similar_evidence=evidence,
        )

    return PreFilterDecision(
        escalate=False,
        stage="embedding",
        reason=f"ไม่พบ keyword และไม่คล้ายหลักฐานเดิม (similarity={similarity:.3f} < {similarity_threshold})",
        similarity=similarity,
        similar_evidence=evidence,
    )
