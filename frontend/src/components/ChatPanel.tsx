"use client";

import { useState, useRef, useEffect } from "react";
import { Send } from "lucide-react";
import { sendChat, Citation } from "@/lib/api";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  route?: string;
}

interface ChatPanelProps {
  onCitationsUpdate: (citations: Citation[]) => void;
  onCitationClick: (index: number) => void;
}

export default function ChatPanel({
  onCitationsUpdate,
  onCitationClick,
}: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const query = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setIsLoading(true);

    try {
      const response = await sendChat(query);
      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: response.answer,
        citations: response.citations,
        route: response.query_route,
      };
      setMessages((prev) => [...prev, assistantMsg]);

      if (response.citations.length > 0) {
        onCitationsUpdate(response.citations);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${err instanceof Error ? err.message : "Unknown error"}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[var(--border)]">
        <h2 className="font-semibold text-sm">Chat</h2>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-[var(--text-secondary)] text-sm text-center mt-8">
            Ask a question about your documents
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${
              msg.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[85%] px-3 py-2 rounded-lg text-sm ${
                msg.role === "user"
                  ? "bg-[var(--accent)] text-white"
                  : "bg-[var(--bg-tertiary)] text-[var(--text-primary)]"
              }`}
            >
              {/* Render answer with clickable citation markers */}
              <MessageContent
                content={msg.content}
                citations={msg.citations}
                onCitationClick={onCitationClick}
              />

              {msg.route && (
                <div className="mt-1 text-xs opacity-60">
                  Route: {msg.route}
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-[var(--bg-tertiary)] px-3 py-2 rounded-lg text-sm">
              <span className="animate-pulse">Thinking...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        className="p-3 border-t border-[var(--border)]"
      >
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your documents..."
            className="flex-1 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="p-2 bg-[var(--accent)] rounded-lg hover:bg-[var(--accent-hover)] disabled:opacity-40 transition-colors"
          >
            <Send size={16} />
          </button>
        </div>
      </form>
    </div>
  );
}

/** Renders message content with clickable [N] citation badges. */
function MessageContent({
  content,
  citations,
  onCitationClick,
}: {
  content: string;
  citations?: Citation[];
  onCitationClick: (index: number) => void;
}) {
  if (!citations || citations.length === 0) {
    return <span className="whitespace-pre-wrap">{content}</span>;
  }

  // Split content on [N] markers and interleave with citation badges
  const parts = content.split(/(\[\d+\])/g);

  return (
    <span className="whitespace-pre-wrap">
      {parts.map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (match) {
          const num = parseInt(match[1], 10);
          const citIndex = num - 1; // citations are 1-indexed in text
          if (citIndex >= 0 && citIndex < citations.length) {
            return (
              <button
                key={i}
                onClick={() => onCitationClick(citIndex)}
                className="inline-flex items-center justify-center w-5 h-5 text-xs bg-[var(--accent)] text-white rounded-full mx-0.5 hover:bg-[var(--accent-hover)] transition-colors cursor-pointer"
                title={`${citations[citIndex].source_file} p.${citations[citIndex].page_number}`}
              >
                {num}
              </button>
            );
          }
        }
        return <span key={i}>{part}</span>;
      })}
    </span>
  );
}
