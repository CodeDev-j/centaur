"use client";

import { useRef, useEffect, useCallback } from "react";
import { Send, X, Globe, FileText, Eraser, Check, Loader2, Circle } from "lucide-react";
import { streamChat, ChatMessagePayload } from "@/lib/api";
import { useChatStore, ChatMessage, StreamStep } from "@/stores/useChatStore";
import { useDocStore } from "@/stores/useDocStore";
import { useViewerStore } from "@/stores/useViewerStore";
import { useInspectStore } from "@/stores/useInspectStore";

/** Maximum number of prior messages to send for multi-turn context */
const MAX_HISTORY = 6;

export default function ChatPanel() {
  const messages = useChatStore((s) => s.messages);
  const isThinking = useChatStore((s) => s.isThinking);
  const docScope = useChatStore((s) => s.docScope);
  const citations = useChatStore((s) => s.citations);

  const selectedDocHash = useDocStore((s) => s.selectedDocHash);
  const selectedFilename = useDocStore((s) => s.selectedFilename);

  const inputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const activeStreamRef = useRef<{ cancel: () => void } | null>(null);

  // R2 fix: cancel in-flight stream on unmount
  useEffect(() => () => { activeStreamRef.current?.cancel(); }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleCitationClick = useCallback((index: number) => {
    const cits = useChatStore.getState().citations;
    const citation = cits[index];
    if (!citation) return;

    if (citation.doc_hash && citation.doc_hash !== useDocStore.getState().selectedDocHash) {
      useDocStore.getState().selectDocument(citation.doc_hash);
    }

    const targetPage = citation.bbox?.page_number || citation.page_number;
    if (targetPage && targetPage >= 1) {
      useViewerStore.getState().setCurrentPage(targetPage);
    }

    useChatStore.getState().setActiveCitationIndex(index);

    // Smart default: auto-open Inspect panel when clicking a citation
    useInspectStore.getState().ensurePanelOpen("inspect");
  }, []);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    const input = inputRef.current;
    if (!input) return;
    const query = input.value.trim();
    if (!query || useChatStore.getState().isThinking) return;

    input.value = "";
    const store = useChatStore.getState();

    // Add user message
    store.addMessage({ role: "user", content: query });
    store.setIsThinking(true);

    // Build history for multi-turn (last N messages, excluding dividers/streaming)
    const history: ChatMessagePayload[] = store.messages
      .filter((m) => !m.isStreaming && !m.content.startsWith("---"))
      .slice(-MAX_HISTORY)
      .map((m) => ({ role: m.role, content: m.content }));

    // Determine doc filter
    const docFilter =
      store.docScope === "selected"
        ? useDocStore.getState().selectedDocHash
        : null;

    // Add ghost bubble for streaming progress
    const ghostMessage: ChatMessage = {
      role: "assistant",
      content: "",
      isStreaming: true,
      streamSteps: [
        { label: "Routing query", status: "active" },
        { label: "Retrieving relevant chunks", status: "pending" },
        { label: "Generating answer", status: "pending" },
      ],
    };
    store.addMessage(ghostMessage);

    let finalAnswer = "";
    let finalCitations: typeof citations = [];
    let routeValue = "";

    // R2 fix: cancel any in-flight stream before starting new one
    activeStreamRef.current?.cancel();

    const stream = streamChat(
      query,
      { doc_filter: docFilter, messages: history },
      (event) => {
        const chatStore = useChatStore.getState();

        if (event.type === "route") {
          const content = event.content as Record<string, string>;
          routeValue = content?.query_route || "hybrid";
          chatStore.updateLastAssistant({
            streamSteps: [
              { label: `Routed via ${routeValue}`, status: "done" },
              { label: "Retrieving relevant chunks", status: "active" },
              { label: "Generating answer", status: "pending" },
            ],
          });
        } else if (event.type === "answer") {
          finalAnswer = event.content as string;
          chatStore.updateLastAssistant({
            streamSteps: [
              { label: `Routed via ${routeValue || "hybrid"}`, status: "done" },
              { label: "Chunks retrieved", status: "done" },
              { label: "Generating answer", status: "active" },
            ],
          });
        } else if (event.type === "citations") {
          finalCitations = event.content as typeof citations;
        } else if (event.type === "error") {
          chatStore.updateLastAssistant({
            content: `Error: ${event.content}`,
            isStreaming: false,
            streamSteps: undefined,
          });
          chatStore.setIsThinking(false);
        }
      }
    );
    activeStreamRef.current = stream;

    try {
      await stream.done;
    } catch {
      // AbortError from cancellation — handled by onEvent
    } finally {
      activeStreamRef.current = null;
    }

    // Replace ghost bubble with final answer
    const chatStore = useChatStore.getState();
    if (finalAnswer) {
      chatStore.updateLastAssistant({
        content: finalAnswer,
        citations: finalCitations.length > 0 ? finalCitations : undefined,
        route: routeValue || undefined,
        isStreaming: false,
        streamSteps: undefined,
      });
      if (finalCitations.length > 0) {
        chatStore.setCitations(finalCitations);
      }
    } else if (chatStore.messages[chatStore.messages.length - 1]?.isStreaming) {
      // No answer received — show error
      chatStore.updateLastAssistant({
        content: "No response received from the server.",
        isStreaming: false,
        streamSteps: undefined,
      });
    }
    chatStore.setIsThinking(false);
  }, []); // P4 fix: no deps needed — reads live from getState()

  const handleClear = useCallback(() => {
    useChatStore.getState().clearConversation();
  }, []);

  const toggleScope = useCallback(() => {
    const store = useChatStore.getState();
    store.setDocScope(store.docScope === "all" ? "selected" : "all");
  }, []);

  return (
    <div className="flex flex-col h-full">
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
            className={`flex animate-message-arrive ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] px-3 py-2 rounded-lg text-sm ${
                msg.role === "user"
                  ? "bg-[var(--accent)] text-white"
                  : msg.content.startsWith("---")
                    ? "bg-transparent text-[var(--text-secondary)] text-xs text-center w-full border-t border-[var(--border)] pt-2"
                    : "bg-[var(--bg-tertiary)] text-[var(--text-primary)]"
              }`}
            >
              {/* Streaming progress stepper */}
              {msg.isStreaming && msg.streamSteps ? (
                <StreamStepper steps={msg.streamSteps} />
              ) : msg.content.startsWith("---") ? (
                <span>{msg.content.replace(/^---\s*/, "").replace(/\s*---$/, "")}</span>
              ) : (
                <>
                  <MessageContent
                    content={msg.content}
                    citations={msg.citations}
                    onCitationClick={handleCitationClick}
                  />
                  {msg.route && (
                    <div className="mt-1 text-xs opacity-60">
                      Route: {msg.route}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <form onSubmit={handleSubmit} className="p-3 border-t border-[var(--border)]">
        {/* Doc scope pill + clear button row */}
        <div className="flex items-center gap-2 mb-2">
          <button
            type="button"
            onClick={toggleScope}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-colors border ${
              docScope === "selected" && selectedDocHash
                ? "bg-[var(--accent)]/15 border-[var(--accent)]/40 text-[var(--accent)]"
                : "bg-[var(--bg-tertiary)] border-[var(--border)] text-[var(--text-secondary)]"
            }`}
            title={docScope === "selected" ? "Searching this document only" : "Searching all documents"}
          >
            {docScope === "selected" && selectedDocHash ? (
              <>
                <FileText size={12} />
                <span className="max-w-[140px] truncate">
                  {selectedFilename || "Selected doc"}
                </span>
                <X
                  size={12}
                  className="opacity-60 hover:opacity-100"
                  onClick={(e) => {
                    e.stopPropagation();
                    useChatStore.getState().setDocScope("all");
                  }}
                />
              </>
            ) : (
              <>
                <Globe size={12} />
                <span>All documents</span>
              </>
            )}
          </button>

          {messages.length > 0 && (
            <button
              type="button"
              onClick={handleClear}
              className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]"
              title="Clear conversation"
            >
              <Eraser size={12} />
              New topic
            </button>
          )}
        </div>

        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            placeholder="Ask about your documents..."
            className="flex-1 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
          />
          <button
            type="submit"
            disabled={isThinking}
            className="btn-primary p-2"
          >
            <Send size={16} />
          </button>
        </div>
      </form>
    </div>
  );
}

/** Terminal-style progress stepper shown in ghost bubble */
function StreamStepper({ steps }: { steps: StreamStep[] }) {
  return (
    <div className="space-y-1.5 py-1">
      {steps.map((step, i) => (
        <div key={i} className="flex items-center gap-2 text-xs">
          {step.status === "done" && (
            <Check size={12} className="text-green-400 shrink-0" />
          )}
          {step.status === "active" && (
            <Loader2 size={12} className="text-[var(--accent)] animate-spin shrink-0" />
          )}
          {step.status === "pending" && (
            <Circle size={12} className="text-[var(--text-secondary)] opacity-40 shrink-0" />
          )}
          <span
            className={
              step.status === "done"
                ? "text-[var(--text-primary)]"
                : step.status === "active"
                  ? "text-[var(--accent)]"
                  : "text-[var(--text-secondary)] opacity-50"
            }
          >
            {step.label}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Renders message content with clickable [N] citation badges */
function MessageContent({
  content,
  citations,
  onCitationClick,
}: {
  content: string;
  citations?: { source_file: string; page_number: number; blurb: string }[];
  onCitationClick: (index: number) => void;
}) {
  if (!citations || citations.length === 0) {
    return <span className="whitespace-pre-wrap">{content}</span>;
  }

  const parts = content.split(/(\[\d+\])/g);

  return (
    <span className="whitespace-pre-wrap">
      {parts.map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (match) {
          const num = parseInt(match[1], 10);
          const citIndex = num - 1;
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
