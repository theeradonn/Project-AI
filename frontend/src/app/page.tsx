/**
 * SafeTrade — Main Page
 *
 * หน้าหลักที่ให้ผู้ใช้เลือก Role (ผู้ซื้อ/ผู้ขาย) ก่อนเข้าห้องแชท
 *
 * Flow (Multi-Room):
 * 1. เลือก Role + ใส่ชื่อ
 * 2. Buyer: กดสร้างห้อง → ได้ Room Code → แสดงให้ copy แจ้งผู้ขาย
 * 3. Seller: กรอก Room Code ที่ได้จากผู้ซื้อ → กดเข้าร่วม
 * 4. เข้า ChatRoom component
 */

"use client";

import { useState } from "react";
import ChatRoom from "@/components/ChatRoom";

// ==========================================
// Types
// ==========================================
type UserRole = "buyer" | "seller";

interface UserSession {
  role: UserRole;
  name: string;
  roomId: string;
}

// ==========================================
// Role Selection Landing Page
// ==========================================
function LandingPage({
  onJoin,
}: {
  onJoin: (session: UserSession) => void;
}) {
  const [name, setName] = useState("");
  const [selectedRole, setSelectedRole] = useState<UserRole | null>(null);

  // Buyer flow state
  const [isCreatingRoom, setIsCreatingRoom] = useState(false);
  const [createdRoomCode, setCreatedRoomCode] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Seller flow state
  const [roomCodeInput, setRoomCodeInput] = useState("");
  const [roomCodeError, setRoomCodeError] = useState("");

  // ==========================================
  // Buyer: สร้างห้องใหม่ผ่าน Supabase โดยตรง
  //
  // ทำที่ฝั่ง frontend แทนการเรียก FastAPI เพราะเมื่อ deploy ขึ้น Vercel แล้ว
  // เว็บจะเรียก backend ที่รันบนเครื่องผู้ใช้ (localhost) ไม่ได้
  // logic เหมือน backend/main.py::create_room() ทุกประการ
  // ==========================================

  /** สุ่ม Room Code 6 ตัว จาก A-Z และ 0-9 (เช่น AB12CD) */
  const generateRoomCode = (length = 6): string => {
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    return Array.from(
      { length },
      () => chars[Math.floor(Math.random() * chars.length)]
    ).join("");
  };

  const handleCreateRoom = async () => {
    if (!name.trim()) return;
    setIsCreatingRoom(true);
    try {
      const { supabase } = await import("@/lib/supabase");

      // สุ่มจนกว่าจะได้รหัสที่ไม่ซ้ำ (โอกาสชนต่ำมาก แต่กันไว้เหมือนฝั่ง backend)
      let roomId = "";
      for (let attempt = 0; attempt < 10; attempt++) {
        const candidate = generateRoomCode();
        const { data } = await supabase
          .from("chatrooms")
          .select("id")
          .eq("id", candidate)
          .maybeSingle();

        if (!data) {
          roomId = candidate;
          break;
        }
      }
      if (!roomId) throw new Error("สุ่มรหัสห้องที่ไม่ซ้ำไม่สำเร็จ");

      const { error } = await supabase.from("chatrooms").insert({
        id: roomId,
        risk_percentage: -1,
        reasoning: "ยังไม่มีข้อมูลเพียงพอสำหรับการวิเคราะห์",
        last_updated: new Date().toISOString(),
      });
      if (error) throw error;

      setCreatedRoomCode(roomId);
    } catch (err) {
      console.error(err);
      alert("ไม่สามารถสร้างห้องแชทได้ กรุณาลองใหม่อีกครั้ง");
    } finally {
      setIsCreatingRoom(false);
    }
  };

  // ==========================================
  // Buyer: copy Room Code
  // ==========================================
  const handleCopyCode = () => {
    if (!createdRoomCode) return;
    navigator.clipboard.writeText(createdRoomCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // ==========================================
  // Buyer: เข้าร่วมห้องที่เพิ่งสร้าง
  // ==========================================
  const handleBuyerJoin = () => {
    if (!createdRoomCode || !name.trim()) return;
    onJoin({ role: "buyer", name: name.trim(), roomId: createdRoomCode });
  };

  // ==========================================
  // Seller: เข้าห้องด้วย Room Code
  // ==========================================
  const handleSellerJoin = async () => {
    const code = roomCodeInput.trim().toUpperCase();
    if (!code || !name.trim()) return;

    setRoomCodeError("");
    try {
      // ตรวจสอบว่าห้องนี้มีอยู่จริงใน Supabase ผ่าน backend health check
      // เราใช้ dynamic import supabase client ที่ frontend มีอยู่แล้ว
      // เพื่อ query ห้องโดยตรงจาก Supabase
      const { supabase } = await import("@/lib/supabase");
      const { data } = await supabase
        .from("chatrooms")
        .select("id")
        .eq("id", code)
        .single();

      if (!data) {
        setRoomCodeError("ไม่พบ Room Code นี้ กรุณาตรวจสอบรหัสอีกครั้ง");
        return;
      }

      onJoin({ role: "seller", name: name.trim(), roomId: code });
    } catch {
      setRoomCodeError("ไม่พบ Room Code นี้ กรุณาตรวจสอบรหัสอีกครั้ง");
    }
  };

  return (
    <div className="min-h-[100dvh] bg-gradient-to-br from-[#0a0908] via-[#12100c] to-[#0a0908] flex items-center justify-center p-4">
      {/* Background Decorations */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-amber-500/10 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-yellow-600/10 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-amber-500/5 rounded-full blur-3xl" />
      </div>

      {/* Card */}
      <div className="relative w-full max-w-md">
        <div className="bg-[#12100c]/80 backdrop-blur-xl border border-amber-500/15 rounded-2xl shadow-2xl shadow-black/20 p-6 sm:p-8">
          {/* Logo */}
          <div className="flex flex-col items-center mb-8">
            <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-amber-300 to-yellow-600 shadow-lg shadow-amber-500/20 mb-4">
              <span className="text-3xl">🛡️</span>
            </div>
            <h1 className="text-2xl font-bold text-white tracking-tight">
              SafeTrade
            </h1>
            <p className="text-sm text-stone-400 mt-1 text-center">
              AI-Powered Fraud Risk Analysis for Online Trading
            </p>
          </div>

          {/* Name Input */}
          <div className="mb-6">
            <label className="block text-xs font-medium text-stone-400 mb-2">
              ชื่อของคุณ
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="ใส่ชื่อที่จะแสดงในแชท..."
              className="w-full bg-stone-800/60 border border-amber-500/15 rounded-xl px-4 py-3 text-sm text-white placeholder-stone-500 focus:outline-none focus:ring-2 focus:ring-amber-500/40 focus:border-amber-500/40 transition-all duration-200"
            />
          </div>

          {/* Role Selection */}
          <div className="mb-6">
            <label className="block text-xs font-medium text-stone-400 mb-3">
              เลือกบทบาทของคุณ
            </label>
            <div className="grid grid-cols-2 gap-3">
              {/* Buyer Card */}
              <button
                id="role-buyer-btn"
                onClick={() => {
                  setSelectedRole("buyer");
                  setCreatedRoomCode(null);
                  setRoomCodeInput("");
                  setRoomCodeError("");
                }}
                className={`group relative flex flex-col items-center gap-2 p-4 rounded-xl border transition-all duration-300 ${
                  selectedRole === "buyer"
                    ? "bg-amber-500/15 border-amber-500/50 ring-2 ring-amber-500/30 shadow-lg shadow-amber-500/10"
                    : "bg-stone-800/40 border-amber-500/10 hover:bg-stone-800/60 hover:border-amber-500/15"
                }`}
              >
                <div
                  className={`w-12 h-12 rounded-xl flex items-center justify-center text-xl transition-all duration-300 ${
                    selectedRole === "buyer"
                      ? "bg-gradient-to-br from-amber-300 to-yellow-500 shadow-lg shadow-amber-500/20 scale-110"
                      : "bg-stone-700/60 group-hover:bg-stone-700"
                  }`}
                >
                  🛒
                </div>
                <span
                  className={`text-sm font-semibold transition-colors ${
                    selectedRole === "buyer" ? "text-amber-300" : "text-stone-300"
                  }`}
                >
                  ผู้ซื้อ
                </span>
                <span className="text-[10px] text-stone-500">Buyer</span>
              </button>

              {/* Seller Card */}
              <button
                id="role-seller-btn"
                onClick={() => {
                  setSelectedRole("seller");
                  setCreatedRoomCode(null);
                  setRoomCodeInput("");
                  setRoomCodeError("");
                }}
                className={`group relative flex flex-col items-center gap-2 p-4 rounded-xl border transition-all duration-300 ${
                  selectedRole === "seller"
                    ? "bg-yellow-600/15 border-yellow-600/50 ring-2 ring-yellow-600/30 shadow-lg shadow-yellow-600/10"
                    : "bg-stone-800/40 border-amber-500/10 hover:bg-stone-800/60 hover:border-amber-500/15"
                }`}
              >
                <div
                  className={`w-12 h-12 rounded-xl flex items-center justify-center text-xl transition-all duration-300 ${
                    selectedRole === "seller"
                      ? "bg-gradient-to-br from-stone-600 to-stone-700 shadow-lg shadow-stone-900/40 scale-110"
                      : "bg-stone-700/60 group-hover:bg-stone-700"
                  }`}
                >
                  🏪
                </div>
                <span
                  className={`text-sm font-semibold transition-colors ${
                    selectedRole === "seller"
                      ? "text-yellow-500"
                      : "text-stone-300"
                  }`}
                >
                  ผู้ขาย
                </span>
                <span className="text-[10px] text-stone-500">Seller</span>
              </button>
            </div>
          </div>

          {/* ==================== BUYER FLOW ==================== */}
          {selectedRole === "buyer" && (
            <div className="mb-6 space-y-3">
              {!createdRoomCode ? (
                /* ปุ่มสร้างห้องใหม่ */
                <button
                  id="create-room-btn"
                  onClick={handleCreateRoom}
                  disabled={!name.trim() || isCreatingRoom}
                  className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-amber-300 to-yellow-600 shadow-lg shadow-amber-500/20 hover:shadow-amber-500/40 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100"
                >
                  {isCreatingRoom ? "⏳ กำลังสร้างห้อง..." : "➕ สร้างห้องแชทใหม่"}
                </button>
              ) : (
                /* แสดง Room Code + ปุ่ม copy + ปุ่มเข้าร่วม */
                <div className="space-y-3">
                  <div className="rounded-xl bg-stone-800/60 border border-amber-500/30 p-4">
                    <p className="text-xs text-stone-400 mb-2 text-center">
                      📋 Room Code — แชร์รหัสนี้ให้ผู้ขาย
                    </p>
                    <div className="flex items-center justify-center gap-3">
                      <code className="text-2xl font-bold tracking-[0.3em] text-amber-200 font-mono">
                        {createdRoomCode}
                      </code>
                      <button
                        id="copy-room-code-btn"
                        onClick={handleCopyCode}
                        title="คัดลอก Room Code"
                        className="p-2 rounded-lg bg-stone-700/60 hover:bg-amber-500/20 text-stone-400 hover:text-amber-300 border border-amber-500/10 hover:border-amber-500/30 transition-all duration-200"
                      >
                        {copied ? "✅" : "📋"}
                      </button>
                    </div>
                    {copied && (
                      <p className="text-[10px] text-amber-300 text-center mt-1 animate-in fade-in duration-200">
                        คัดลอกแล้ว!
                      </p>
                    )}
                  </div>

                  <button
                    id="buyer-join-btn"
                    onClick={handleBuyerJoin}
                    className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-amber-300 to-yellow-600 shadow-lg shadow-amber-500/20 hover:shadow-amber-500/40 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200"
                  >
                    เข้าร่วมห้องแชท →
                  </button>
                </div>
              )}
            </div>
          )}

          {/* ==================== SELLER FLOW ==================== */}
          {selectedRole === "seller" && (
            <div className="mb-6 space-y-3">
              <div>
                <label className="block text-xs font-medium text-stone-400 mb-2">
                  Room Code (รับจากผู้ซื้อ)
                </label>
                <input
                  id="room-code-input"
                  type="text"
                  value={roomCodeInput}
                  onChange={(e) => {
                    setRoomCodeInput(e.target.value.toUpperCase());
                    setRoomCodeError("");
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleSellerJoin();
                  }}
                  placeholder="เช่น AB12CD"
                  maxLength={6}
                  className={`w-full bg-stone-800/60 border rounded-xl px-4 py-3 text-sm text-white placeholder-stone-500 font-mono tracking-widest text-center focus:outline-none focus:ring-2 transition-all duration-200 ${
                    roomCodeError
                      ? "border-red-500/50 focus:ring-red-500/30"
                      : "border-amber-500/15 focus:ring-yellow-600/40 focus:border-yellow-600/50"
                  }`}
                />
                {roomCodeError && (
                  <p className="text-xs text-red-400 mt-1.5 flex items-center gap-1">
                    <span>⚠️</span> {roomCodeError}
                  </p>
                )}
              </div>

              <button
                id="seller-join-btn"
                onClick={handleSellerJoin}
                disabled={!roomCodeInput.trim() || !name.trim()}
                className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-stone-600 to-stone-700 shadow-lg shadow-stone-900/40 hover:shadow-yellow-600/30 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100"
              >
                เข้าร่วมห้องแชท →
              </button>
            </div>
          )}

          {/* Footer Note */}
          <p className="text-center text-[10px] text-stone-600 mt-2">
            ระบบจะวิเคราะห์ความเสี่ยงด้วย AI อัตโนมัติขณะที่คุณสนทนา
          </p>
        </div>
      </div>
    </div>
  );
}

// ==========================================
// Main Page Component
// ==========================================
export default function Home() {
  const [session, setSession] = useState<UserSession | null>(null);

  if (!session) {
    return <LandingPage onJoin={setSession} />;
  }

  return (
    <ChatRoom
      roomId={session.roomId}
      userRole={session.role}
      userName={session.name}
    />
  );
}
