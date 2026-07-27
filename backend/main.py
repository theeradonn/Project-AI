"""
SafeTrade — FastAPI Backend (main.py)
Real-time Inference Pipeline — Supabase Edition

Architecture:
  FastAPI ← CORS → Next.js Frontend
  FastAPI → Supabase (polling listener — all rooms)
  FastAPI → Ollama API (qwen3:8b) → Risk Analysis
"""

import os
import logging
import random
import string
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from supabase import create_client, Client
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from supabase_listener import start_listener, stop_listener

# ==========================================
# Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# โหลด environment variables จากไฟล์ .env
load_dotenv()


# ==========================================
# Supabase Client Initialization
# ==========================================
def init_supabase() -> Client:
    """
    สร้าง Supabase client จาก environment variables

    ต้องตั้งค่า 2 ค่าใน .env:
    - SUPABASE_URL: URL ของ Supabase project (เช่น https://xxxxx.supabase.co)
    - SUPABASE_SERVICE_KEY: Service Role Key (ไม่ใช่ anon key)

    วิธีดูค่า:
    Supabase Dashboard → Project Settings → API → Project URL + service_role key
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise ValueError(
            "❌ ไม่พบ SUPABASE_URL หรือ SUPABASE_SERVICE_KEY\n"
            "กรุณาตั้งค่าใน .env :\n"
            "  SUPABASE_URL=https://xxxxx.supabase.co\n"
            "  SUPABASE_SERVICE_KEY=eyJhbGci...\n\n"
            "ดูค่าได้ที่: Supabase Dashboard → Settings → API"
        )

    client = create_client(url, key)
    logger.info(f"✅ Supabase เชื่อมต่อสำเร็จ ({url})")
    return client


# ==========================================
# FastAPI Lifespan (Startup / Shutdown)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    จัดการวงจรชีวิตของแอปพลิเคชัน:

    Startup:
      1. เชื่อมต่อ Supabase
      2. เริ่ม background polling listener

    Shutdown:
      1. หยุด listener
    """
    # =================== STARTUP ===================
    logger.info("🚀 กำลังเริ่มต้น SafeTrade Backend...")

    try:
        # 1. เชื่อมต่อ Supabase
        supabase = init_supabase()
        app.state.supabase = supabase  # เก็บไว้ใช้ใน endpoints

        # 2. เริ่ม background listener — poll ข้อความใหม่ทุก 3 วินาที
        start_listener(supabase, interval=3.0)

        logger.info("✅ SafeTrade Backend พร้อมทำงาน!")
        logger.info("📖 เปิด API docs ได้ที่: http://localhost:8000/docs")

    except Exception as e:
        logger.error(f"❌ ไม่สามารถเริ่มต้นได้: {e}")
        raise

    yield  # ← แอปทำงานอยู่ระหว่างจุดนี้

    # =================== SHUTDOWN ===================
    logger.info("🛑 กำลังปิด SafeTrade Backend...")
    stop_listener()
    logger.info("👋 SafeTrade Backend ปิดเรียบร้อยแล้ว")


# ==========================================
# FastAPI Application
# ==========================================
app = FastAPI(
    title="SafeTrade API",
    description=(
        "Real-time Inference Pipeline สำหรับวิเคราะห์ความเสี่ยง"
        "ในการโกงซื้อขายออนไลน์ ขับเคลื่อนด้วย AI (qwen3:8b) — Supabase Edition"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Middleware — อนุญาตให้ Next.js Frontend เข้าถึง API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Request/Response Models
# ==========================================
class CreateRoomResponse(BaseModel):
    room_id: str


# ==========================================
# API Endpoints
# ==========================================
@app.get("/")
async def root():
    """หน้าแรก — ข้อมูลเบื้องต้นของ API"""
    return {
        "app": "SafeTrade API",
        "version": "2.1.0",
        "database": "Supabase",
        "status": "running",
        "docs_url": "/docs",
    }


def _generate_room_code(length: int = 6) -> str:
    """สุ่มสร้าง Room Code ตัวอักษรพิมพ์ใหญ่ + ตัวเลข (เช่น AB12CD)"""
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))


@app.post("/rooms", response_model=CreateRoomResponse, status_code=201)
async def create_room():
    """
    สร้างห้องแชทใหม่พร้อม Room Code ที่สุ่มขึ้นมา

    - สุ่ม Room Code 6 ตัวอักษร (A-Z + 0-9)
    - สร้าง record ในตาราง chatrooms พร้อมค่าเริ่มต้น
    - คืน room_id กลับไปให้ Frontend นำไปแสดงแก่ผู้ซื้อ
    """
    supabase: Client = app.state.supabase

    # สุ่มจนกว่าจะได้ Room Code ที่ไม่ซ้ำกับที่มีอยู่ (collision ต่ำมาก แต่ safe)
    for _ in range(10):
        room_id = _generate_room_code()
        existing = (
            supabase.table("chatrooms")
            .select("id")
            .eq("id", room_id)
            .execute()
        )
        if not existing.data:
            break  # ได้ room_id ที่ unique แล้ว
    else:
        raise HTTPException(status_code=500, detail="ไม่สามารถสร้าง Room ID ที่ไม่ซ้ำได้")

    # สร้าง record ห้องใหม่ใน chatrooms
    now = datetime.now(timezone.utc).isoformat()
    supabase.table("chatrooms").insert({
        "id": room_id,
        "risk_percentage": -1,
        "reasoning": "ยังไม่มีข้อมูลเพียงพอสำหรับการวิเคราะห์",
        "last_updated": now,
    }).execute()

    logger.info(f"🆕 สร้างห้องแชทใหม่: {room_id}")
    return CreateRoomResponse(room_id=room_id)


@app.get("/health")
async def health_check():
    """
    Health Check — ตรวจสอบสถานะของ dependencies ทั้งหมด

    ตรวจสอบ:
    - Supabase connection
    - Ollama API + qwen3:8b model
    """
    import httpx

    # --- ตรวจสอบ Supabase ---
    supabase_status = "unknown"
    try:
        result = (
            app.state.supabase.table("chatrooms")
            .select("id")
            .limit(1)
            .execute()
        )
        supabase_status = "connected"
    except Exception as e:
        supabase_status = f"error: {str(e)[:80]}"

    # --- ตรวจสอบ Ollama ---
    ollama_status = "unknown"
    ollama_models: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                ollama_models = [m.get("name", "") for m in models]
                has_qwen = any("qwen3" in name for name in ollama_models)
                ollama_status = (
                    "connected"
                    if has_qwen
                    else "connected (⚠️ qwen3:8b not found — run: ollama pull qwen3:8b)"
                )
            else:
                ollama_status = f"error (status: {resp.status_code})"
    except httpx.ConnectError:
        ollama_status = "disconnected (run: ollama serve)"
    except Exception as e:
        ollama_status = f"error: {str(e)[:50]}"

    return {
        "status": "healthy" if supabase_status == "connected" else "degraded",
        "supabase": supabase_status,
        "ollama": ollama_status,
        "ollama_models": ollama_models,
    }
