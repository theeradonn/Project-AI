/**
 * SafeTrade — Supabase Client Configuration
 *
 * ตั้งค่า Supabase สำหรับฝั่ง Frontend (Next.js)
 * ใช้ environment variables จาก .env.local
 *
 * การใช้งาน:
 *   import { supabase } from "@/lib/supabase";
 */

import { createClient } from "@supabase/supabase-js";

const supabaseUrl =
  process.env.NEXT_PUBLIC_SUPABASE_URL || "https://placeholder.supabase.co";
const supabaseAnonKey =
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "placeholder-key";

// แจ้งเตือนถ้ายังไม่ได้ตั้งค่า (จะเห็นใน browser console)
if (
  typeof window !== "undefined" &&
  (!process.env.NEXT_PUBLIC_SUPABASE_URL ||
    !process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY)
) {
  console.warn(
    "⚠️ Supabase ยังไม่ได้ตั้งค่า — สร้างไฟล์ .env.local ก่อน (ดูตัวอย่างจาก .env.local.example)"
  );
}

// Supabase client instance — export ไปใช้ทั่วทั้งแอป
export const supabase = createClient(supabaseUrl, supabaseAnonKey);
