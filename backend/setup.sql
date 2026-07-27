-- ==========================================
-- SafeTrade — Supabase Database Setup
-- รัน SQL นี้ใน Supabase Dashboard → SQL Editor
-- ==========================================

-- >>> สำหรับรันอัปเดตระบบเดิม (Migration): <<<
-- ALTER TABLE messages ADD COLUMN IF NOT EXISTS risk_percentage INTEGER DEFAULT -1;
-- ALTER TABLE messages ADD COLUMN IF NOT EXISTS reasoning TEXT;
-- ALTER TABLE messages ADD COLUMN IF NOT EXISTS detected_flags JSONB DEFAULT '[]'::jsonb;
-- ALTER TABLE messages ADD COLUMN IF NOT EXISTS scam_pattern TEXT;
-- ALTER TABLE messages ADD COLUMN IF NOT EXISTS turn INTEGER;
-- ALTER TABLE chatrooms ADD COLUMN IF NOT EXISTS detected_flags JSONB DEFAULT '[]'::jsonb;
-- ALTER TABLE chatrooms ADD COLUMN IF NOT EXISTS scam_pattern TEXT;

-- 1. ตารางข้อความแชท
CREATE TABLE IF NOT EXISTS messages (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  room_id TEXT NOT NULL DEFAULT 'demo-room',
  text TEXT NOT NULL,
  sender TEXT NOT NULL CHECK (sender IN ('buyer', 'seller')),
  sender_name TEXT NOT NULL,
  processed BOOLEAN DEFAULT FALSE,
  risk_percentage INTEGER DEFAULT -1,
  reasoning TEXT,
  detected_flags JSONB DEFAULT '[]'::jsonb,
  scam_pattern TEXT,
  turn INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. ตารางห้องแชท (เก็บ Risk Score)
CREATE TABLE IF NOT EXISTS chatrooms (
  id TEXT PRIMARY KEY DEFAULT 'demo-room',
  risk_percentage INTEGER DEFAULT -1,
  reasoning TEXT DEFAULT 'ยังไม่มีข้อมูลเพียงพอสำหรับการวิเคราะห์',
  detected_flags JSONB DEFAULT '[]'::jsonb,
  scam_pattern TEXT,
  last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- 3. สร้างห้อง demo
INSERT INTO chatrooms (id) VALUES ('demo-room')
ON CONFLICT (id) DO NOTHING;

-- 4. เปิด Real-time สำหรับ Frontend
DO $$
BEGIN
  BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE messages;
  EXCEPTION WHEN duplicate_object THEN
    RAISE NOTICE 'Table messages is already in publication';
  END;
  
  BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE chatrooms;
  EXCEPTION WHEN duplicate_object THEN
    RAISE NOTICE 'Table chatrooms is already in publication';
  END;
END $$;

-- 5. Row Level Security (RLS) — อนุญาตให้ทุกคนอ่าน/เขียนได้ (สำหรับ dev)
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE chatrooms ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow all on messages" ON messages;
CREATE POLICY "Allow all on messages" ON messages
  FOR ALL USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "Allow all on chatrooms" ON chatrooms;
CREATE POLICY "Allow all on chatrooms" ON chatrooms
  FOR ALL USING (true) WITH CHECK (true);

-- 6. Index สำหรับ query ที่ใช้บ่อย
CREATE INDEX IF NOT EXISTS idx_messages_room_processed
  ON messages (room_id, processed);

CREATE INDEX IF NOT EXISTS idx_messages_room_created
  ON messages (room_id, created_at);
