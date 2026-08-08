"""
SafeTrade — Embedding Client (Ollama)

ห่อการเรียก embedding model ผ่าน Ollama สำหรับใช้ในด่านที่ 2 ของ pre-filter
และสำหรับสร้าง Evidence Bank ล่วงหน้า

ทำไมใช้ Ollama แทน sentence-transformers:
- ใช้ runtime เดียวกับ Qwen ตัวหลัก ไม่ต้องลง torch (~2-3GB)
- Ollama จัดการ VRAM ให้เอง (unload โมเดลที่ไม่ได้ใช้) ลดโอกาสแย่ VRAM กับ qwen3:8b

ติดตั้งโมเดลก่อนใช้งาน:
    ollama pull qwen3-embedding:0.6b
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# ใช้ตัว 0.6B เท่านั้น — ห้ามใช้ 8B เพราะจะแย่ VRAM กับ qwen3:8b ที่ใช้วิเคราะห์เต็มรูปแบบ
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
EMBEDDING_TIMEOUT = 60.0


class EmbeddingError(RuntimeError):
    """เกิดข้อผิดพลาดตอนเรียก embedding model"""


def embed_texts(texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
    """
    แปลงข้อความหลายรายการเป็น embedding vectors ผ่าน Ollama

    Args:
        texts: รายการข้อความ
        batch_size: จำนวนข้อความต่อ 1 request (Ollama /api/embed รับ list ได้)

    Returns:
        list ของ vector (float) เรียงตามลำดับ texts ที่ส่งเข้ามา

    Raises:
        EmbeddingError: ถ้าเชื่อมต่อ Ollama ไม่ได้ หรือโมเดลยังไม่ถูก pull
    """
    if not texts:
        return []

    vectors: list[list[float]] = []

    try:
        with httpx.Client(timeout=EMBEDDING_TIMEOUT) as client:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                response = client.post(
                    f"{OLLAMA_BASE_URL}/api/embed",
                    json={"model": EMBEDDING_MODEL, "input": batch},
                )
                if response.status_code == 404:
                    raise EmbeddingError(
                        f"ไม่พบโมเดล '{EMBEDDING_MODEL}' ใน Ollama — "
                        f"รันคำสั่งนี้ก่อน: ollama pull {EMBEDDING_MODEL}"
                    )
                response.raise_for_status()
                data = response.json()

                batch_vectors = data.get("embeddings")
                if not batch_vectors:
                    raise EmbeddingError(f"Ollama ไม่ได้คืน embeddings กลับมา: {str(data)[:200]}")
                vectors.extend(batch_vectors)

    except httpx.ConnectError as e:
        raise EmbeddingError(
            f"เชื่อมต่อ Ollama ไม่ได้ที่ {OLLAMA_BASE_URL} — เปิด ollama serve ก่อน"
        ) from e
    except httpx.HTTPError as e:
        raise EmbeddingError(f"เรียก Ollama embedding ไม่สำเร็จ: {e}") from e

    return vectors


def embed_one(text: str) -> list[float]:
    """
    แปลงข้อความเดียวเป็น embedding vector

    Args:
        text: ข้อความ

    Returns:
        vector (list ของ float)
    """
    result = embed_texts([text])
    if not result:
        raise EmbeddingError("ไม่ได้รับ embedding กลับมา")
    return result[0]


def is_available() -> bool:
    """
    เช็คว่า embedding model พร้อมใช้งานไหม (Ollama รันอยู่ + pull โมเดลแล้ว)

    Returns:
        True ถ้าพร้อมใช้ / False ถ้าไม่พร้อม (ระบบจะ fallback ไปใช้แค่ด่าน keyword)
    """
    try:
        embed_one("ทดสอบ")
        return True
    except Exception as e:
        logger.warning(f"⚠️ embedding model ไม่พร้อมใช้งาน: {e}")
        return False
