/**
 * SafeTrade — ChatRoom Component (Supabase Edition)
 *
 * Client Component สำหรับห้องแชทระหว่างผู้ซื้อและผู้ขาย
 *
 * ฟีเจอร์หลัก:
 * 1. UI แบ่ง 2 ฝั่ง — ข้อความผู้ซื้อ (ซ้าย) / ผู้ขาย (ขวา)
 * 2. Real-time messaging — ใช้ Supabase Realtime (postgres_changes)
 * 3. Risk Score Monitor — แถบเปอร์เซ็นต์ + reasoning ที่อัปเดตอัตโนมัติ
 * 4. Responsive + Animated UI — Premium dark theme
 */

"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { supabase } from "@/lib/supabase";

// ==========================================
// Types
// ==========================================
interface Message {
  id: string;
  room_id: string;
  text: string;
  sender: "buyer" | "seller";
  sender_name: string;
  created_at: string;
  processed: boolean;
  risk_percentage?: number;
  reasoning?: string;
}

interface RiskData {
  risk_percentage: number;
  reasoning: string;
  last_updated: string | null;
}

interface ChatRoomProps {
  roomId: string;
  userRole: "buyer" | "seller";
  userName: string;
}

// ==========================================
// Risk Level Helpers
// ==========================================
function getRiskLevel(percentage: number) {
  if (percentage < 0)
    return { label: "รอวิเคราะห์", color: "gray", emoji: "⏳" };
  if (percentage <= 20)
    return { label: "ปลอดภัย", color: "emerald", emoji: "🟢" };
  if (percentage <= 50)
    return { label: "ระวัง", color: "amber", emoji: "🟡" };
  if (percentage <= 80)
    return { label: "น่าสงสัย", color: "orange", emoji: "🟠" };
  return { label: "เสี่ยงสูงมาก", color: "red", emoji: "🔴" };
}

function getRiskGradient(percentage: number): string {
  if (percentage < 0) return "from-gray-500 to-gray-600";
  if (percentage <= 20) return "from-emerald-400 to-green-500";
  if (percentage <= 50) return "from-amber-400 to-yellow-500";
  if (percentage <= 80) return "from-orange-400 to-orange-600";
  return "from-red-500 to-rose-600";
}

function getRiskBarBg(percentage: number): string {
  if (percentage < 0) return "bg-gray-500/20";
  if (percentage <= 20) return "bg-emerald-500/20";
  if (percentage <= 50) return "bg-amber-500/20";
  if (percentage <= 80) return "bg-orange-500/20";
  return "bg-red-500/20";
}

// ==========================================
// Component
// ==========================================
export default function ChatRoom({
  roomId,
  userRole,
  userName,
}: ChatRoomProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [riskData, setRiskData] = useState<RiskData>({
    risk_percentage: -1,
    reasoning: "ยังไม่มีข้อมูลเพียงพอสำหรับการวิเคราะห์",
    last_updated: null,
  });
  const [newMessage, setNewMessage] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isResetting, setIsResetting] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // ฟังก์ชันรีเซ็ตห้องแชท (ลบข้อความ และตั้งค่าความเสี่ยงเริ่มต้นใหม่)
  const handleResetChat = async () => {
    const isConfirmed = window.confirm(
      "คุณต้องการลบประวัติการสนทนาทั้งหมดในห้องนี้และเริ่มประเมินใหม่ใช่หรือไม่?"
    );
    if (!isConfirmed) return;

    setIsResetting(true);
    try {
      // 1. ลบข้อความทั้งหมดของห้องนี้
      const { error: deleteError } = await supabase
        .from("messages")
        .delete()
        .eq("room_id", roomId);

      if (deleteError) throw deleteError;

      // 2. รีเซ็ตคะแนนความเสี่ยงกลับเป็นเริ่มต้น (-1)
      const { error: resetError } = await supabase
        .from("chatrooms")
        .upsert({
          id: roomId,
          risk_percentage: -1,
          reasoning: "ยังไม่มีข้อมูลเพียงพอสำหรับการวิเคราะห์",
          last_updated: new Date().toISOString(),
        });

      if (resetError) throw resetError;

      // 3. ล้าง state ในระบบ
      setMessages([]);
      setRiskData({
        risk_percentage: -1,
        reasoning: "ยังไม่มีข้อมูลเพียงพอสำหรับการวิเคราะห์",
        last_updated: null,
      });
    } catch (error) {
      console.error("เกิดข้อผิดพลาดในการรีเซ็ตห้องแชท:", error);
      alert("รีเซ็ตแชทไม่สำเร็จ กรุณาลองใหม่อีกครั้ง");
    } finally {
      setIsResetting(false);
    }
  };

  // Auto-scroll ไปข้อความล่าสุด
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // ==========================================
  // Supabase Listeners
  // ==========================================

  // Listener 1: ข้อความแชท (real-time)
  useEffect(() => {
    // ดึงข้อความทั้งหมดตอนเริ่มต้น
    const fetchMessages = async () => {
      const { data } = await supabase
        .from("messages")
        .select("*")
        .eq("room_id", roomId)
        .order("created_at", { ascending: true });

      if (data) setMessages(data as Message[]);
    };
    fetchMessages();

    // Subscribe to new messages (real-time)
    const channel = supabase
      .channel(`messages:${roomId}`)
      .on(
        "postgres_changes",
        {
          event: "INSERT",
          schema: "public",
          table: "messages",
          filter: `room_id=eq.${roomId}`,
        },
        (payload) => {
          const newMsg = payload.new as Message;
          setMessages((prev) => {
            // ป้องกัน duplicate (กรณี optimistic update)
            if (prev.some((m) => m.id === newMsg.id)) return prev;
            return [...prev, newMsg];
          });
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [roomId]);

  // Listener 2: Risk Score (real-time)
  useEffect(() => {
    // ดึง risk score ตอนเริ่มต้น
    const fetchRisk = async () => {
      const { data } = await supabase
        .from("chatrooms")
        .select("*")
        .eq("id", roomId)
        .single();

      if (data) {
        setRiskData({
          risk_percentage: data.risk_percentage ?? -1,
          reasoning: data.reasoning ?? "ยังไม่มีข้อมูลเพียงพอ",
          last_updated: data.last_updated,
        });
      }
    };
    fetchRisk();

    // Subscribe to risk score updates (real-time)
    // ใช้ "*" เพื่อรับทั้ง INSERT และ UPDATE — เพราะ upsert จาก backend
    // อาจ trigger เป็น INSERT หรือ UPDATE ขึ้นอยู่กับสถานะของ record
    const channel = supabase
      .channel(`risk:${roomId}`)
      .on(
        "postgres_changes",
        {
          event: "*",
          schema: "public",
          table: "chatrooms",
          filter: `id=eq.${roomId}`,
        },
        (payload) => {
          const updated = payload.new as Record<string, unknown>;
          setRiskData({
            risk_percentage: (updated.risk_percentage as number) ?? -1,
            reasoning: (updated.reasoning as string) ?? "ไม่มีเหตุผลระบุ",
            last_updated: (updated.last_updated as string) ?? null,
          });
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [roomId]);

  // ==========================================
  // Send Message
  // ==========================================
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();

    const trimmed = newMessage.trim();
    if (!trimmed || isSending) return;

    setIsSending(true);
    try {
      const { error } = await supabase.from("messages").insert({
        room_id: roomId,
        text: trimmed,
        sender: userRole,
        sender_name: userName,
        processed: false,
      });

      if (error) {
        console.error("ส่งข้อความไม่สำเร็จ:", error.message);
      } else {
        setNewMessage("");
        inputRef.current?.focus();
      }
    } catch (error) {
      console.error("ส่งข้อความไม่สำเร็จ:", error);
    } finally {
      setIsSending(false);
    }
  };

  // ==========================================
  // Render Helpers
  // ==========================================
  const riskLevel = getRiskLevel(riskData.risk_percentage);
  const riskGradient = getRiskGradient(riskData.risk_percentage);
  const riskBarBg = getRiskBarBg(riskData.risk_percentage);
  const displayPercentage = Math.max(0, riskData.risk_percentage);

  const formatTime = (isoString: string | null): string => {
    if (!isoString) return "";
    try {
      const date = new Date(isoString);
      return date.toLocaleTimeString("th-TH", {
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return "";
    }
  };

  // ==========================================
  // JSX
  // ==========================================
  return (
    <div className="flex flex-col h-[100dvh] bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* ====== HEADER + RISK MONITOR ====== */}
      <header className="shrink-0 border-b border-white/10 bg-slate-900/80 backdrop-blur-xl">
        {/* Title Bar */}
        <div className="flex items-center justify-between px-4 sm:px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 shadow-lg shadow-indigo-500/25">
              <span className="text-lg">🛡️</span>
            </div>
            <div>
              <h1 className="text-base font-bold text-white tracking-tight">
                SafeTrade
              </h1>
              <div className="flex items-center gap-1.5 mt-0.5">
                <p className="text-xs text-slate-400">AI-Powered Fraud Detection</p>
                <span className="text-slate-600 text-xs">·</span>
                <code className="text-xs text-indigo-400/80 font-mono tracking-wider">{roomId}</code>
              </div>
            </div>
          </div>

          {/* Controls & User Badge */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* Reset Chat Button */}
            <button
              onClick={handleResetChat}
              disabled={isResetting}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-slate-800 hover:bg-red-500/20 text-slate-300 hover:text-red-400 border border-white/10 hover:border-red-500/30 transition-all duration-200 disabled:opacity-50"
              title="ล้างข้อความทั้งหมดและรีเซ็ตการประเมิน"
            >
              <span>{isResetting ? "⏳" : "🔄"}</span>
              <span>รีเซ็ตแชท</span>
            </button>

            {/* User Badge */}
            <div
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold ${
                userRole === "buyer"
                  ? "bg-blue-500/15 text-blue-400 ring-1 ring-blue-500/30"
                  : "bg-violet-500/15 text-violet-400 ring-1 ring-violet-500/30"
              }`}
            >
              <span className="relative flex h-2 w-2">
                <span
                  className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                    userRole === "buyer" ? "bg-blue-400" : "bg-violet-400"
                  }`}
                />
                <span
                  className={`relative inline-flex rounded-full h-2 w-2 ${
                    userRole === "buyer" ? "bg-blue-500" : "bg-violet-500"
                  }`}
                />
              </span>
              {userName} ({userRole === "buyer" ? "ผู้ซื้อ" : "ผู้ขาย"})
            </div>
          </div>
        </div>

        {/* Risk Score Monitor */}
        <div className="px-4 sm:px-6 pb-3">
          <div className="rounded-xl bg-slate-800/60 border border-white/5 p-3 sm:p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-sm">{riskLevel.emoji}</span>
                <span className="text-xs font-medium text-slate-300">
                  Risk Assessment
                </span>
              </div>
              <div className="flex items-baseline gap-1.5">
                <span
                  className={`text-2xl font-bold tracking-tighter bg-gradient-to-r ${riskGradient} bg-clip-text text-transparent`}
                >
                  {riskData.risk_percentage < 0 ? "—" : `${displayPercentage}`}
                </span>
                {riskData.risk_percentage >= 0 && (
                  <span className="text-xs font-medium text-slate-500">%</span>
                )}
              </div>
            </div>

            {/* Progress Bar */}
            <div
              className={`w-full h-2 rounded-full ${riskBarBg} overflow-hidden`}
            >
              <div
                className={`h-full rounded-full bg-gradient-to-r ${riskGradient} transition-all duration-1000 ease-out`}
                style={{
                  width: `${riskData.risk_percentage < 0 ? 0 : displayPercentage}%`,
                }}
              />
            </div>

            {/* Risk Label + Reasoning */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between mt-2 gap-1">
              <span
                className={`text-xs font-semibold px-2 py-0.5 rounded-full w-fit ${
                  riskLevel.color === "gray"
                    ? "bg-gray-500/15 text-gray-400"
                    : riskLevel.color === "emerald"
                      ? "bg-emerald-500/15 text-emerald-400"
                      : riskLevel.color === "amber"
                        ? "bg-amber-500/15 text-amber-400"
                        : riskLevel.color === "orange"
                          ? "bg-orange-500/15 text-orange-400"
                          : "bg-red-500/15 text-red-400"
                }`}
              >
                {riskLevel.label}
              </span>
              <p className="text-xs text-slate-500 truncate max-w-xs sm:max-w-sm">
                {riskData.reasoning}
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* ====== MESSAGES AREA ====== */}
      <main className="flex-1 overflow-y-auto px-4 sm:px-6 py-4 space-y-3 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center gap-2 text-slate-500">
            <span className="text-3xl">💬</span>
            <p className="text-sm">ยังไม่มีข้อความในห้องนี้ — เริ่มทักทายกันได้เลย</p>
          </div>
        ) : (
          messages.map((msg, idx) => {
            const isCurrentUser = msg.sender === userRole;
            const isBuyer = msg.sender === "buyer";
            const showAvatar =
              idx === 0 || messages[idx - 1].sender !== msg.sender;

            return (
              <div
                key={msg.id}
                className={`flex items-end gap-2 ${
                  isCurrentUser ? "flex-row-reverse" : "flex-row"
                }`}
              >
                {/* Avatar */}
                {showAvatar ? (
                  <div
                    className={`shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold shadow-lg ${
                      isBuyer
                        ? "bg-gradient-to-br from-blue-500 to-indigo-600 text-white shadow-blue-500/25"
                        : "bg-gradient-to-br from-violet-500 to-purple-600 text-white shadow-violet-500/25"
                    }`}
                  >
                    {isBuyer ? "ซ" : "ข"}
                  </div>
                ) : (
                  <div className="w-7 shrink-0" />
                )}

                {/* Message Bubble */}
                <div
                  className={`max-w-[75%] sm:max-w-[65%] ${isCurrentUser ? "items-end" : "items-start"}`}
                >
                  {/* Sender Name */}
                  {showAvatar && (
                    <p
                      className={`text-[10px] font-medium mb-1 px-1 ${
                        isCurrentUser ? "text-right" : "text-left"
                      } ${isBuyer ? "text-blue-400/70" : "text-violet-400/70"}`}
                    >
                      {msg.sender_name || (isBuyer ? "ผู้ซื้อ" : "ผู้ขาย")}
                    </p>
                  )}

                  {/* Bubble & Per-Message Risk indicator */}
                  <div className={`flex items-center gap-2 ${isCurrentUser ? "flex-row-reverse" : "flex-row"}`}>
                    {/* Bubble */}
                    <div
                      className={`relative px-3.5 py-2.5 rounded-2xl text-sm leading-relaxed ${
                        isCurrentUser
                          ? isBuyer
                            ? "bg-gradient-to-br from-blue-600 to-blue-700 text-white rounded-br-md"
                            : "bg-gradient-to-br from-violet-600 to-purple-700 text-white rounded-br-md"
                          : "bg-slate-800/80 text-slate-200 border border-white/5 rounded-bl-md"
                      }`}
                    >
                      <p>{msg.text}</p>
                      <p
                        className={`text-[10px] mt-1 ${
                          isCurrentUser
                            ? "text-white/50 text-right"
                            : "text-slate-500 text-left"
                        }`}
                      >
                        {formatTime(msg.created_at)}
                      </p>
                    </div>

                    {/* Per-Message Risk Indicator Badge */}
                    {msg.processed && msg.risk_percentage !== undefined && msg.risk_percentage >= 0 && (
                      <div className="group relative flex items-center justify-center shrink-0">
                        <span
                          className={`cursor-help flex items-center justify-center w-5 h-5 rounded-full text-[9px] font-bold border transition-transform duration-200 hover:scale-110 ${
                            msg.risk_percentage <= 20
                              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                              : msg.risk_percentage <= 50
                                ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                                : msg.risk_percentage <= 80
                                  ? "bg-orange-500/10 text-orange-400 border-orange-500/30"
                                  : "bg-red-500/10 text-red-400 border-red-500/30"
                          }`}
                        >
                          {msg.risk_percentage}%
                        </span>

                        {/* Tooltip on hover */}
                        <div className={`absolute bottom-6 hidden group-hover:block z-50 w-52 p-2.5 rounded-xl bg-slate-900 border border-white/10 shadow-2xl text-[11px] text-slate-300 leading-normal animate-in fade-in duration-200 ${
                          isCurrentUser ? "right-0" : "left-0"
                        }`}>
                          <div className="font-semibold text-white mb-1 flex items-center gap-1.5">
                            <span>{getRiskLevel(msg.risk_percentage).emoji}</span>
                            <span>ความเสี่ยง ณ จุดนี้: {msg.risk_percentage}%</span>
                          </div>
                          <p>{msg.reasoning || "อยู่ระหว่างประมวลผลความเสี่ยง"}</p>
                          <div className={`absolute bottom-[-5px] w-2.5 h-2.5 bg-slate-900 border-r border-b border-white/10 rotate-45 ${
                            isCurrentUser ? "right-2" : "left-2"
                          }`} />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
        <div ref={messagesEndRef} />
      </main>

      {/* ====== INPUT AREA ====== */}
      <footer className="shrink-0 border-t border-white/10 bg-slate-900/80 backdrop-blur-xl p-3 sm:p-4">
        <form
          onSubmit={handleSendMessage}
          className="flex items-center gap-2 sm:gap-3"
        >
          <input
            ref={inputRef}
            type="text"
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            placeholder="พิมพ์ข้อความ..."
            disabled={isSending}
            className="flex-1 bg-slate-800/60 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all duration-200 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!newMessage.trim() || isSending}
            className="shrink-0 flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40 hover:scale-105 active:scale-95 transition-all duration-200 disabled:opacity-40 disabled:hover:scale-100 disabled:hover:shadow-indigo-500/25"
          >
            {isSending ? (
              <svg
                className="animate-spin h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
            ) : (
              <svg
                className="h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"
                />
              </svg>
            )}
          </button>
        </form>
      </footer>
    </div>
  );
}
