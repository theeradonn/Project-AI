# 🛡️ SafeTrade — Real-time Fraud Risk Analysis

แอปพลิเคชันแชทพลัง AI ที่วิเคราะห์แนวโน้มและประเมินเปอร์เซ็นต์ความเสี่ยงในการโกงซื้อขายออนไลน์แบบเรียลไทม์

## 🏗️ Architecture

```
Next.js (Chat UI) ↔ Supabase (PostgreSQL + Realtime) ↔ FastAPI (Polling Listener) → Ollama (qwen3:8b) → Risk Score → Supabase → Next.js
```

---

## 📋 Prerequisites

| Tool | Version | ดาวน์โหลด |
|------|---------|-----------|
| Node.js | ≥ 18 | https://nodejs.org/ |
| Python | ≥ 3.10 | https://www.python.org/ |
| Ollama | latest | https://ollama.com/ |
| Supabase Account | Free | https://supabase.com/ |

---

## 🔧 Setup Instructions

### 1. ติดตั้ง Ollama + Pull Model

```bash
# ติดตั้ง Ollama (ดาวน์โหลดจาก https://ollama.com/)
# จากนั้น pull model qwen3:8b
ollama pull qwen3:8b

# ตรวจสอบว่า model พร้อมใช้
ollama list
```

### 2. ตั้งค่า Supabase

1. ไปที่ [supabase.com](https://supabase.com) → **Sign Up** ด้วย GitHub
2. คลิก **New Project** → ตั้งชื่อ + password → รอ ~1 นาที
3. ไปที่ **SQL Editor** → วาง SQL จากไฟล์ `backend/setup.sql` → คลิก **Run**
4. คัดลอก config:
   - **Settings → API → Project URL** → ใช้เป็น `SUPABASE_URL`
   - **Settings → API → `anon` public key** → ใช้เป็น `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - **Settings → API → `service_role` secret key** → ใช้เป็น `SUPABASE_SERVICE_KEY`

### 3. ติดตั้ง Backend (FastAPI)

```bash
cd backend

# สร้าง Virtual Environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# ติดตั้ง Dependencies
pip install -r requirements.txt

# สร้างไฟล์ .env (คัดลอกจาก .env.example แล้วใส่ค่าจาก Supabase)
copy .env.example .env
# แก้ไข SUPABASE_URL และ SUPABASE_SERVICE_KEY ใน .env
```

### 4. ติดตั้ง Frontend (Next.js)

```bash
cd frontend

# ติดตั้ง Dependencies (ถ้ายังไม่ได้ติดตั้ง)
npm install

# สร้างไฟล์ .env.local
copy .env.local.example .env.local
# แก้ไข NEXT_PUBLIC_SUPABASE_URL และ NEXT_PUBLIC_SUPABASE_ANON_KEY
```

---

## 🚀 วิธีรัน (3 Terminals)

### Terminal 1: Ollama
```bash
ollama serve
```

### Terminal 2: Backend
```bash
cd backend
.\venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

### Terminal 3: Frontend
```bash
cd frontend
npm run dev
```

### ทดสอบ

1. เปิด 2 แท็บที่ `http://localhost:3000`
2. แท็บ 1: เลือก **ผู้ซื้อ** → ใส่ชื่อ → เข้าห้อง
3. แท็บ 2: เลือก **ผู้ขาย** → ใส่ชื่อ → เข้าห้อง
4. เริ่มแชทกัน — ดู Risk Score อัปเดตแบบ Real-time!

---

## 📁 Project Structure

```
NewRMproject/
├── backend/
│   ├── main.py                 # FastAPI app + Supabase init
│   ├── ollama_client.py        # Ollama API client + System Prompt
│   ├── supabase_listener.py    # Background polling listener
│   ├── setup.sql               # SQL สำหรับสร้างตารางใน Supabase
│   ├── requirements.txt        # Python dependencies
│   └── .env.example            # Backend config template
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx        # Landing page (role selection)
│   │   │   ├── layout.tsx      # Root layout
│   │   │   └── globals.css     # Global styles + animations
│   │   ├── components/
│   │   │   └── ChatRoom.tsx    # Chat UI + Risk Score display
│   │   └── lib/
│   │       └── supabase.ts     # Supabase client config
│   ├── .env.local.example      # Frontend config template
│   └── package.json
│
└── README.md
```

---

## 🔍 Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "supabase": "connected",
  "ollama": "connected",
  "ollama_models": ["qwen3:8b"]
}
```
