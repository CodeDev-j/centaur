import { create } from "zustand";
import { Citation, VizPayload } from "@/lib/api";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  route?: string;
  /** Set while streaming: shows progress steps before answer arrives */
  isStreaming?: boolean;
  streamSteps?: StreamStep[];
  /** Visualization payload from the visualize node */
  viz?: VizPayload;
}

export interface StreamStep {
  label: string;
  status: "done" | "active" | "pending";
}

interface ChatState {
  messages: ChatMessage[];
  isThinking: boolean;
  docScope: "all" | "selected";
  activeCitationIndex: number | null;
  citations: Citation[];

  addMessage: (msg: ChatMessage) => void;
  updateLastAssistant: (update: Partial<ChatMessage>) => void;
  setIsThinking: (v: boolean) => void;
  setDocScope: (scope: "all" | "selected") => void;
  setActiveCitationIndex: (index: number | null) => void;
  setCitations: (citations: Citation[]) => void;
  clearConversation: () => void;
  insertDivider: (docName: string) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isThinking: false,
  docScope: "all",
  activeCitationIndex: null,
  citations: [],

  addMessage: (msg) =>
    set((s) => ({ messages: [...s.messages, msg] })),

  updateLastAssistant: (update) =>
    set((s) => {
      const msgs = [...s.messages];
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === "assistant") {
          msgs[i] = { ...msgs[i], ...update };
          break;
        }
      }
      return { messages: msgs };
    }),

  setIsThinking: (v) => set({ isThinking: v }),
  setDocScope: (scope) => set({ docScope: scope }),
  setActiveCitationIndex: (index) => set({ activeCitationIndex: index }),
  setCitations: (citations) =>
    set({ citations, activeCitationIndex: null }),
  clearConversation: () =>
    set({ messages: [], citations: [], activeCitationIndex: null }),

  insertDivider: (docName) =>
    set((s) => ({
      messages: [
        ...s.messages,
        {
          role: "assistant" as const,
          content: `--- Context switched to ${docName} ---`,
        },
      ],
    })),
}));
