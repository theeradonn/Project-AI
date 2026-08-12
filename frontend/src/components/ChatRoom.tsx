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
  /** รายการ pattern ที่ตรวจพบใน turn นี้ เช่น "[B3] ปฏิเสธการเก็บเงินปลายทาง (+20): ..." */
  detected_flags?: string[];
  scam_pattern?: string | null;
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

  /** อีกฝ่ายกำลังพิมพ์อยู่ไหม (มาจาก broadcast ไม่ได้เก็บลง DB) */
  const [isOtherTyping, setIsOtherTyping] = useState(false);
  /** เวลาที่อีกฝ่ายอ่านแชทล่าสุด — ใช้ตัดสินว่าข้อความไหนของเราถูกอ่านแล้ว */
  const [otherLastReadAt, setOtherLastReadAt] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  /** channel สำหรับส่งสัญญาณ "กำลังพิมพ์" */
  const typingChannelRef = useRef<ReturnType<typeof supabase.channel> | null>(null);
  /** timer ซ่อน indicator เมื่ออีกฝ่ายหยุดพิมพ์ */
  const typingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** กันไม่ให้ยิง broadcast ถี่เกินไปตอนพิมพ์รัว */
  const lastTypingSentRef = useRef<number>(0);

  const otherRole = userRole === "buyer" ? "seller" : "buyer";

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
        setOtherLastReadAt(
          (userRole === "buyer" ? data.seller_last_read_at : data.buyer_last_read_at) ?? null
        );
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
          // อีกฝ่ายเพิ่งอ่านแชท → อัปเดตสถานะ "อ่านแล้ว"
          const readKey = userRole === "buyer" ? "seller_last_read_at" : "buyer_last_read_at";
          setOtherLastReadAt((updated[readKey] as string) ?? null);
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [roomId, userRole]);

  // ==========================================
  // Listener 3: "กำลังพิมพ์" — ใช้ Broadcast (ไม่แตะ DB เพราะเป็นข้อมูลชั่วคราว)
  // ==========================================
  useEffect(() => {
    const channel = supabase.channel(`typing:${roomId}`, {
      config: { broadcast: { self: false } }, // ไม่รับ event ของตัวเอง
    });

    channel
      .on("broadcast", { event: "typing" }, (payload) => {
        // สนใจเฉพาะตอนอีกฝ่ายพิมพ์ (ไม่ใช่ตัวเองจากอีกแท็บ)
        if ((payload.payload as { role?: string })?.role !== otherRole) return;

        setIsOtherTyping(true);
        if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
        // ถ้าไม่มีสัญญาณใหม่ภายใน 2.5 วิ ถือว่าหยุดพิมพ์แล้ว
        typingTimeoutRef.current = setTimeout(() => setIsOtherTyping(false), 2500);
      })
      .subscribe();

    typingChannelRef.current = channel;

    return () => {
      if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
      supabase.removeChannel(channel);
      typingChannelRef.current = null;
    };
  }, [roomId, otherRole]);

  /** ส่งสัญญาณว่ากำลังพิมพ์ (ยิงไม่เกิน 1 ครั้ง/วินาที) */
  const broadcastTyping = useCallback(() => {
    const now = Date.now();
    if (now - lastTypingSentRef.current < 1000) return;
    lastTypingSentRef.current = now;

    typingChannelRef.current?.send({
      type: "broadcast",
      event: "typing",
      payload: { role: userRole },
    });
  }, [userRole]);

  // ==========================================
  // Read Receipts — บันทึกว่าเราอ่านถึงเมื่อไหร่
  // ==========================================
  const markAsRead = useCallback(async () => {
    const column = userRole === "buyer" ? "buyer_last_read_at" : "seller_last_read_at";
    const { error } = await supabase
      .from("chatrooms")
      .update({ [column]: new Date().toISOString() })
      .eq("id", roomId);

    // ถ้ายังไม่ได้รัน migration เพิ่มคอลัมน์ จะ error ตรงนี้ — ปล่อยผ่านไม่ให้แอปพัง
    if (error) {
      console.warn(
        "อัปเดตสถานะ 'อ่านแล้ว' ไม่สำเร็จ (อาจยังไม่ได้เพิ่มคอลัมน์ใน Supabase):",
        error.message
      );
    }
  }, [roomId, userRole]);

  // อ่านแล้วเมื่อ: เปิดห้อง / มีข้อความใหม่เข้ามาขณะเปิดหน้าอยู่ / กลับมาโฟกัสหน้าต่าง
  useEffect(() => {
    if (typeof document !== "undefined" && document.visibilityState === "hidden") return;
    markAsRead();
  }, [messages.length, markAsRead]);

  useEffect(() => {
    const onFocus = () => markAsRead();
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onFocus);
    return () => {
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onFocus);
    };
  }, [markAsRead]);

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
  /** id ของข้อความ "ล่าสุดของเรา" ที่อีกฝ่ายอ่านแล้ว — ใช้ติดป้าย "อ่านแล้ว" จุดเดียว */
  const lastReadOwnMessageId = (() => {
    if (!otherLastReadAt) return null;
    const readAt = new Date(otherLastReadAt).getTime();
    const ownRead = messages.filter(
      (m) => m.sender === userRole && new Date(m.created_at).getTime() <= readAt
    );
    return ownRead.length ? ownRead[ownRead.length - 1].id : null;
  })();

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

  /** ป้ายวันที่สำหรับคั่นกลางแชท — วันนี้/เมื่อวาน หรือวันที่เต็ม */
  const formatDateLabel = (isoString: string | null): string => {
    if (!isoString) return "";
    try {
      const date = new Date(isoString);
      const today = new Date();
      const yesterday = new Date(today);
      yesterday.setDate(today.getDate() - 1);

      const sameDay = (a: Date, b: Date) =>
        a.getFullYear() === b.getFullYear() &&
        a.getMonth() === b.getMonth() &&
        a.getDate() === b.getDate();

      if (sameDay(date, today)) return "วันนี้";
      if (sameDay(date, yesterday)) return "เมื่อวาน";
      return date.toLocaleDateString("th-TH", {
        day: "numeric",
        month: "short",
        year: "numeric",
      });
    } catch {
      return "";
    }
  };

  /** เช็คว่าข้อความ 2 อันอยู่คนละวันหรือไม่ (ใช้ตัดสินว่าต้องแทรกป้ายวันที่ไหม) */
  const isDifferentDay = (a: string | null, b: string | null): boolean => {
    if (!a || !b) return false;
    const d1 = new Date(a);
    const d2 = new Date(b);
    return (
      d1.getFullYear() !== d2.getFullYear() ||
      d1.getMonth() !== d2.getMonth() ||
      d1.getDate() !== d2.getDate()
    );
  };

  // ==========================================
  // JSX
  // ==========================================
  return (
    <div className="flex flex-col h-[100dvh] bg-gradient-to-br from-[#0a0908] via-[#12100c] to-[#0a0908]">
      {/* ====== HEADER + RISK MONITOR ====== */}
      <header className="shrink-0 border-b border-amber-500/10 bg-[#12100c]/85 backdrop-blur-xl">
        {/* Title Bar */}
        <div className="flex items-center justify-between px-4 sm:px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-amber-300 to-yellow-600 shadow-lg shadow-amber-500/20">
              <span className="text-lg">🛡️</span>
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight bg-gradient-to-r from-amber-200 to-yellow-500 bg-clip-text text-transparent">
                SafeTrade
              </h1>
              <div className="flex items-center gap-1.5 mt-0.5">
                <p className="text-xs text-stone-400">AI-Powered Fraud Detection</p>
                <span className="text-stone-600 text-xs">·</span>
                <code className="text-xs text-amber-400/80 font-mono tracking-wider">{roomId}</code>
              </div>
            </div>
          </div>

          {/* Controls & User Badge */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* Reset Chat Button */}
            <button
              onClick={handleResetChat}
              disabled={isResetting}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-stone-800/80 hover:bg-red-500/20 text-stone-300 hover:text-red-400 border border-amber-500/15 hover:border-red-500/30 transition-all duration-200 disabled:opacity-50"
              title="ล้างข้อความทั้งหมดและรีเซ็ตการประเมิน"
            >
              <span>{isResetting ? "⏳" : "🔄"}</span>
              <span>รีเซ็ตแชท</span>
            </button>

            {/* User Badge */}
            <div
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold ${
                userRole === "buyer"
                  ? "bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30"
                  : "bg-yellow-600/15 text-yellow-500 ring-1 ring-yellow-600/30"
              }`}
            >
              <span className="relative flex h-2 w-2">
                <span
                  className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                    userRole === "buyer" ? "bg-amber-400" : "bg-yellow-500"
                  }`}
                />
                <span
                  className={`relative inline-flex rounded-full h-2 w-2 ${
                    userRole === "buyer" ? "bg-amber-400" : "bg-yellow-600"
                  }`}
                />
              </span>
              {userName} ({userRole === "buyer" ? "ผู้ซื้อ" : "ผู้ขาย"})
            </div>
          </div>
        </div>

        {/* Risk Score Monitor — แถบเดียวจบ ประหยัดพื้นที่ให้ช่องแชท */}
        <div className="px-4 pb-3 sm:px-6">
          <div className="rounded-2xl border border-amber-500/10 bg-stone-900/60 px-3.5 py-2.5">
            <div className="flex items-center gap-3">
              <span className="text-base">{riskLevel.emoji}</span>

              <div className="min-w-0 flex-1">
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <span
                    className={`text-[11px] font-semibold ${
                      riskLevel.color === "gray"
                        ? "text-stone-400"
                        : riskLevel.color === "emerald"
                          ? "text-emerald-400"
                          : riskLevel.color === "amber"
                            ? "text-amber-400"
                            : riskLevel.color === "orange"
                              ? "text-orange-400"
                              : "text-red-400"
                    }`}
                  >
                    {riskLevel.label}
                  </span>
                  <p className="truncate text-[11px] text-stone-500">
                    {riskData.reasoning}
                  </p>
                </div>

                <div className={`h-1.5 w-full overflow-hidden rounded-full ${riskBarBg}`}>
                  <div
                    className={`h-full rounded-full bg-gradient-to-r ${riskGradient} transition-all duration-1000 ease-out`}
                    style={{
                      width: `${riskData.risk_percentage < 0 ? 0 : displayPercentage}%`,
                    }}
                  />
                </div>
              </div>

              <div className="flex shrink-0 items-baseline gap-0.5">
                <span
                  className={`bg-gradient-to-r text-xl font-bold tracking-tight ${riskGradient} bg-clip-text text-transparent`}
                >
                  {riskData.risk_percentage < 0 ? "—" : displayPercentage}
                </span>
                {riskData.risk_percentage >= 0 && (
                  <span className="text-[10px] font-medium text-stone-500">%</span>
                )}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* ====== MESSAGES AREA ====== */}
      <main className="flex-1 overflow-y-auto px-3 sm:px-5 py-4 scrollbar-thin">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center gap-3 text-stone-500">
            <div className="w-16 h-16 rounded-full bg-stone-800/60 border border-amber-500/10 flex items-center justify-center text-3xl">
              💬
            </div>
            <p className="text-sm">ยังไม่มีข้อความในห้องนี้</p>
            <p className="text-xs text-stone-600">เริ่มทักทายกันได้เลย</p>
          </div>
        ) : (
          messages.map((msg, idx) => {
            const isCurrentUser = msg.sender === userRole;
            const isBuyer = msg.sender === "buyer";

            // จัดกลุ่มข้อความที่ส่งติดกันจากคนเดียวกัน (แบบ Messenger)
            const prev = idx > 0 ? messages[idx - 1] : null;
            const next = idx < messages.length - 1 ? messages[idx + 1] : null;
            const isFirstOfGroup = !prev || prev.sender !== msg.sender;
            const isLastOfGroup = !next || next.sender !== msg.sender;

            // มุมโค้ง: ด้านที่ติดกับข้อความในกลุ่มเดียวกันจะโค้งน้อยลง
            const tail = isCurrentUser ? "r" : "l";
            const radius = [
              "rounded-[18px]",
              !isFirstOfGroup && (tail === "r" ? "rounded-tr-[5px]" : "rounded-tl-[5px]"),
              !isLastOfGroup && (tail === "r" ? "rounded-br-[5px]" : "rounded-bl-[5px]"),
            ]
              .filter(Boolean)
              .join(" ");

            // แสดง badge เฉพาะข้อความที่ "ตรวจพบ pattern จริง" เพื่อลดความรก
            const hasFlags = (msg.detected_flags?.length ?? 0) > 0;

            // แทรกป้ายวันที่เมื่อข้ามวัน (หรือที่ข้อความแรกสุดของแชท)
            const showDateDivider =
              !prev || isDifferentDay(prev.created_at, msg.created_at);

            return (
              <div key={msg.id}>
                {showDateDivider && (
                  <div className="my-5 flex items-center gap-3">
                    <div className="h-px flex-1 bg-amber-500/10" />
                    <span className="text-[10px] font-medium text-stone-500">
                      {formatDateLabel(msg.created_at)}
                    </span>
                    <div className="h-px flex-1 bg-amber-500/10" />
                  </div>
                )}

                <div
                  className={`flex items-end gap-2 ${
                    isFirstOfGroup && !showDateDivider ? "mt-4" : "mt-0.5"
                  } ${isCurrentUser ? "flex-row-reverse" : "flex-row"}`}
                >
                {/* Avatar — โผล่เฉพาะข้อความสุดท้ายของกลุ่ม (แบบ Messenger) */}
                {!isCurrentUser &&
                  (isLastOfGroup ? (
                    <div
                      className={`shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold ${
                        isBuyer
                          ? "bg-gradient-to-br from-amber-300 to-yellow-600 text-stone-900"
                          : "bg-gradient-to-br from-stone-600 to-stone-700 text-amber-200"
                      }`}
                    >
                      {isBuyer ? "ซ" : "ข"}
                    </div>
                  ) : (
                    <div className="w-7 shrink-0" />
                  ))}

                <div className={`flex flex-col max-w-[78%] sm:max-w-[62%] ${isCurrentUser ? "items-end" : "items-start"}`}>
                  {/* ชื่อผู้ส่ง — เฉพาะข้อความแรกของกลุ่ม และเฉพาะฝั่งตรงข้าม */}
                  {isFirstOfGroup && !isCurrentUser && (
                    <p className="text-[11px] font-medium mb-1 px-2 text-stone-500">
                      {msg.sender_name || (isBuyer ? "ผู้ซื้อ" : "ผู้ขาย")}
                    </p>
                  )}

                  <div className={`flex items-center gap-1.5 ${isCurrentUser ? "flex-row-reverse" : "flex-row"}`}>
                    {/* Bubble */}
                    <div
                      className={`group/bubble relative px-3.5 py-2 text-[14px] leading-[1.45] break-words ${radius} ${
                        isCurrentUser
                          ? "bg-gradient-to-br from-amber-300 to-yellow-600 text-stone-900 font-medium"
                          : "bg-stone-800 text-stone-100 border border-amber-500/5"
                      }`}
                    >
                      {msg.text}
                    </div>

                    {/* Risk badge — เฉพาะข้อความที่ตรวจพบ pattern */}
                    {hasFlags && msg.risk_percentage !== undefined && msg.risk_percentage >= 0 && (
                      <div className="group/risk relative flex shrink-0 items-center justify-center">
                        <span
                          className={`flex cursor-help items-center justify-center rounded-full px-1.5 h-[18px] text-[10px] font-bold transition-transform duration-150 hover:scale-110 ${
                            msg.risk_percentage <= 20
                              ? "bg-emerald-500/15 text-emerald-400"
                              : msg.risk_percentage <= 50
                                ? "bg-amber-500/15 text-amber-400"
                                : msg.risk_percentage <= 80
                                  ? "bg-orange-500/15 text-orange-400"
                                  : "bg-red-500/15 text-red-400"
                          }`}
                        >
                          {msg.risk_percentage}
                        </span>

                        {/* Tooltip */}
                        <div
                          className={`invisible absolute bottom-7 z-50 w-60 rounded-xl border border-amber-500/20 bg-stone-900 p-3 text-[11px] leading-relaxed text-stone-300 opacity-0 shadow-2xl transition-all duration-150 group-hover/risk:visible group-hover/risk:opacity-100 ${
                            isCurrentUser ? "right-0" : "left-0"
                          }`}
                        >
                          <div className="mb-1.5 flex items-center gap-1.5 font-semibold text-white">
                            <span>{getRiskLevel(msg.risk_percentage).emoji}</span>
                            <span>ความเสี่ยงสะสม {msg.risk_percentage}%</span>
                          </div>
                          <p className="text-stone-400">
                            {msg.reasoning || "อยู่ระหว่างประมวลผล"}
                          </p>
                          {msg.detected_flags?.map((flag, i) => (
                            <p key={i} className="mt-1.5 border-t border-amber-500/10 pt-1.5 text-[10px] text-stone-500">
                              {flag}
                            </p>
                          ))}
                          <div
                            className={`absolute bottom-[-5px] h-2.5 w-2.5 rotate-45 border-b border-r border-amber-500/20 bg-stone-900 ${
                              isCurrentUser ? "right-3" : "left-3"
                            }`}
                          />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* เวลาที่ส่ง + สถานะอ่านแล้ว */}
                  {(isLastOfGroup || msg.id === lastReadOwnMessageId) && (
                    <span className="mt-1 flex items-center gap-1.5 px-2 text-[10px] tabular-nums text-stone-500">
                      {formatTime(msg.created_at)}
                      {msg.id === lastReadOwnMessageId && (
                        <span className="flex items-center gap-0.5 text-amber-500/80">
                          <svg viewBox="0 0 16 12" className="h-2.5 w-3.5 fill-none stroke-current stroke-2">
                            <path d="M1 6.5 4.5 10 10.5 2" strokeLinecap="round" strokeLinejoin="round" />
                            <path d="M6.5 8.5 8 10 14.5 2" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                          อ่านแล้ว
                        </span>
                      )}
                    </span>
                  )}
                  </div>
                </div>
              </div>
            );
          })
        )}
        {/* กำลังพิมพ์ — จุดกระพริบ 3 จุดแบบ Messenger */}
        {isOtherTyping && (
          <div className="mt-3 flex items-end gap-2">
            <div
              className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[11px] font-bold ${
                otherRole === "buyer"
                  ? "bg-gradient-to-br from-amber-300 to-yellow-600 text-stone-900"
                  : "bg-gradient-to-br from-stone-600 to-stone-700 text-amber-200"
              }`}
            >
              {otherRole === "buyer" ? "ซ" : "ข"}
            </div>
            <div className="flex items-center gap-1 rounded-[18px] border border-amber-500/5 bg-stone-800 px-3.5 py-3">
              {[0, 150, 300].map((delay) => (
                <span
                  key={delay}
                  className="h-1.5 w-1.5 animate-bounce rounded-full bg-stone-500"
                  style={{ animationDelay: `${delay}ms`, animationDuration: "1s" }}
                />
              ))}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </main>

      {/* ====== INPUT AREA ====== */}
      <footer className="shrink-0 border-t border-amber-500/10 bg-[#12100c]/85 px-3 py-3 backdrop-blur-xl sm:px-5">
        <form
          onSubmit={handleSendMessage}
          className="flex items-end gap-2"
        >
          <input
            ref={inputRef}
            type="text"
            value={newMessage}
            onChange={(e) => {
              setNewMessage(e.target.value);
              if (e.target.value.trim()) broadcastTyping();
            }}
            placeholder="Aa"
            disabled={isSending}
            className="min-w-0 flex-1 rounded-full border border-amber-500/10 bg-stone-800 px-4 py-2.5 text-[14px] text-stone-100 placeholder-stone-500 transition-all duration-200 focus:border-amber-500/30 focus:outline-none focus:ring-2 focus:ring-amber-500/30 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!newMessage.trim() || isSending}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-amber-300 to-yellow-600 text-stone-900 transition-all duration-200 hover:brightness-110 active:scale-90 disabled:opacity-40 disabled:hover:brightness-100"
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
