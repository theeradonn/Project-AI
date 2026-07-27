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
  // Buyer: สร้างห้องใหม่ผ่าน Backend API
  // ==========================================
  const handleCreateRoom = async () => {
    if (!name.trim()) return;
    setIsCreatingRoom(true);
    try {
      const res = await fetch("http://localhost:8000/rooms", {
        method: "POST",
      });
      if (!res.ok) throw new Error("สร้างห้องไม่สำเร็จ");
      const data = await res.json();
      setCreatedRoomCode(data.room_id);
    } catch (err) {
      console.error(err);
      alert("ไม่สามารถสร้างห้องแชทได้ กรุณาตรวจสอบว่า Backend กำลังทำงานอยู่");
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
    <div className="min-h-[100dvh] bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 flex items-center justify-center p-4">
      {/* Background Decorations */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl" />
      </div>

      {/* Card */}
      <div className="relative w-full max-w-md">
        <div className="bg-slate-900/70 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl shadow-black/20 p-6 sm:p-8">
          {/* Logo */}
          <div className="flex flex-col items-center mb-8">
            <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 shadow-lg shadow-indigo-500/25 mb-4">
              <span className="text-3xl">🛡️</span>
            </div>
            <h1 className="text-2xl font-bold text-white tracking-tight">
              SafeTrade
            </h1>
            <p className="text-sm text-slate-400 mt-1 text-center">
              AI-Powered Fraud Risk Analysis for Online Trading
            </p>
          </div>

          {/* Name Input */}
          <div className="mb-6">
            <label className="block text-xs font-medium text-slate-400 mb-2">
              ชื่อของคุณ
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="ใส่ชื่อที่จะแสดงในแชท..."
              className="w-full bg-slate-800/60 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all duration-200"
            />
          </div>

          {/* Role Selection */}
          <div className="mb-6">
            <label className="block text-xs font-medium text-slate-400 mb-3">
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
                    ? "bg-blue-500/15 border-blue-500/50 ring-2 ring-blue-500/30 shadow-lg shadow-blue-500/10"
                    : "bg-slate-800/40 border-white/5 hover:bg-slate-800/60 hover:border-white/10"
                }`}
              >
                <div
                  className={`w-12 h-12 rounded-xl flex items-center justify-center text-xl transition-all duration-300 ${
                    selectedRole === "buyer"
                      ? "bg-gradient-to-br from-blue-500 to-cyan-500 shadow-lg shadow-blue-500/25 scale-110"
                      : "bg-slate-700/60 group-hover:bg-slate-700"
                  }`}
                >
                  🛒
                </div>
                <span
                  className={`text-sm font-semibold transition-colors ${
                    selectedRole === "buyer" ? "text-blue-400" : "text-slate-300"
                  }`}
                >
                  ผู้ซื้อ
                </span>
                <span className="text-[10px] text-slate-500">Buyer</span>
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
                    ? "bg-violet-500/15 border-violet-500/50 ring-2 ring-violet-500/30 shadow-lg shadow-violet-500/10"
                    : "bg-slate-800/40 border-white/5 hover:bg-slate-800/60 hover:border-white/10"
                }`}
              >
                <div
                  className={`w-12 h-12 rounded-xl flex items-center justify-center text-xl transition-all duration-300 ${
                    selectedRole === "seller"
                      ? "bg-gradient-to-br from-violet-500 to-purple-600 shadow-lg shadow-violet-500/25 scale-110"
                      : "bg-slate-700/60 group-hover:bg-slate-700"
                  }`}
                >
                  🏪
                </div>
                <span
                  className={`text-sm font-semibold transition-colors ${
                    selectedRole === "seller"
                      ? "text-violet-400"
                      : "text-slate-300"
                  }`}
                >
                  ผู้ขาย
                </span>
                <span className="text-[10px] text-slate-500">Seller</span>
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
                  className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-blue-500 to-cyan-600 shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100"
                >
                  {isCreatingRoom ? "⏳ กำลังสร้างห้อง..." : "➕ สร้างห้องแชทใหม่"}
                </button>
              ) : (
                /* แสดง Room Code + ปุ่ม copy + ปุ่มเข้าร่วม */
                <div className="space-y-3">
                  <div className="rounded-xl bg-slate-800/60 border border-blue-500/30 p-4">
                    <p className="text-xs text-slate-400 mb-2 text-center">
                      📋 Room Code — แชร์รหัสนี้ให้ผู้ขาย
                    </p>
                    <div className="flex items-center justify-center gap-3">
                      <code className="text-2xl font-bold tracking-[0.3em] text-blue-300 font-mono">
                        {createdRoomCode}
                      </code>
                      <button
                        id="copy-room-code-btn"
                        onClick={handleCopyCode}
                        title="คัดลอก Room Code"
                        className="p-2 rounded-lg bg-slate-700/60 hover:bg-blue-500/20 text-slate-400 hover:text-blue-400 border border-white/5 hover:border-blue-500/30 transition-all duration-200"
                      >
                        {copied ? "✅" : "📋"}
                      </button>
                    </div>
                    {copied && (
                      <p className="text-[10px] text-blue-400 text-center mt-1 animate-in fade-in duration-200">
                        คัดลอกแล้ว!
                      </p>
                    )}
                  </div>

                  <button
                    id="buyer-join-btn"
                    onClick={handleBuyerJoin}
                    className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-indigo-500 to-purple-600 shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200"
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
                <label className="block text-xs font-medium text-slate-400 mb-2">
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
                  className={`w-full bg-slate-800/60 border rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 font-mono tracking-widest text-center focus:outline-none focus:ring-2 transition-all duration-200 ${
                    roomCodeError
                      ? "border-red-500/50 focus:ring-red-500/30"
                      : "border-white/10 focus:ring-violet-500/50 focus:border-violet-500/50"
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
                className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-violet-500 to-purple-600 shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100"
              >
                เข้าร่วมห้องแชท →
              </button>
            </div>
          )}

          {/* Footer Note */}
          <p className="text-center text-[10px] text-slate-600 mt-2">
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
