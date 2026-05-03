import { create } from "zustand";
import {
  createSession,
  deleteSession,
  getHistory,
  listSessions,
  renameSession,
} from "@/lib/api";
import { streamChat } from "@/lib/sse";
import type {
  ChatMessage,
  ChatMode,
  SessionItem,
} from "@/types/api";

const ACTIVE_SESSION_KEY = "rag-chat:active-session-id";
const CHAT_MODE_KEY = "rag-chat:mode";

function loadActiveSession(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(ACTIVE_SESSION_KEY);
  } catch {
    return null;
  }
}

function saveActiveSession(sessionId: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (sessionId) {
      window.localStorage.setItem(ACTIVE_SESSION_KEY, sessionId);
    } else {
      window.localStorage.removeItem(ACTIVE_SESSION_KEY);
    }
  } catch {
    // ignore quota / privacy mode failures
  }
}

function loadChatMode(): ChatMode {
  if (typeof window === "undefined") return "rag";
  try {
    const stored = window.localStorage.getItem(CHAT_MODE_KEY);
    if (stored === "agent") return "agent";
    if (stored === "multi_agent") return "multi_agent";
    return "rag";
  } catch {
    return "rag";
  }
}

function saveChatMode(mode: ChatMode): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(CHAT_MODE_KEY, mode);
  } catch {
    // ignore
  }
}

function makeId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

interface ChatState {
  sessions: SessionItem[];
  currentSessionId: string | null;
  messages: ChatMessage[];
  isStreaming: boolean;
  error: string | null;
  isLoadingSessions: boolean;
  isLoadingHistory: boolean;
  mode: ChatMode;

  initialize: () => Promise<void>;
  refreshSessions: () => Promise<void>;
  startNewChat: () => Promise<SessionItem>;
  selectSession: (sessionId: string) => Promise<void>;
  renameCurrent: (title: string) => Promise<void>;
  renameSessionById: (sessionId: string, title: string) => Promise<void>;
  deleteSessionById: (sessionId: string) => Promise<void>;
  sendMessage: (question: string) => Promise<void>;
  clearError: () => void;
  setMode: (mode: ChatMode) => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  sessions: [],
  currentSessionId: null,
  messages: [],
  isStreaming: false,
  error: null,
  isLoadingSessions: false,
  isLoadingHistory: false,
  mode: loadChatMode(),

  initialize: async () => {
    set({ isLoadingSessions: true, error: null });
    try {
      const sessions = await listSessions();
      const stored = loadActiveSession();
      const fallback = stored && sessions.find((s) => s.session_id === stored)
        ? stored
        : sessions[0]?.session_id ?? null;

      set({ sessions, isLoadingSessions: false });

      if (fallback) {
        await get().selectSession(fallback);
      } else {
        saveActiveSession(null);
      }
    } catch (err) {
      set({
        isLoadingSessions: false,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  },

  refreshSessions: async () => {
    try {
      const sessions = await listSessions();
      set({ sessions });
    } catch (err) {
      set({ error: err instanceof Error ? err.message : String(err) });
    }
  },

  startNewChat: async () => {
    const created = await createSession();
    saveActiveSession(created.session_id);
    set((state) => ({
      sessions: [created, ...state.sessions.filter(
        (s) => s.session_id !== created.session_id,
      )],
      currentSessionId: created.session_id,
      messages: [],
      error: null,
    }));
    return created;
  },

  selectSession: async (sessionId) => {
    if (get().isStreaming) return;
    saveActiveSession(sessionId);
    set({
      currentSessionId: sessionId,
      isLoadingHistory: true,
      messages: [],
      error: null,
    });
    try {
      const history = await getHistory(sessionId);
      const messages: ChatMessage[] = history.messages
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map((m) => ({
          id: makeId(),
          role: m.role as "user" | "assistant",
          content: m.content,
        }));
      set({ messages, isLoadingHistory: false });
    } catch (err) {
      set({
        isLoadingHistory: false,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  },

  renameCurrent: async (title) => {
    const id = get().currentSessionId;
    if (!id) return;
    await get().renameSessionById(id, title);
  },

  renameSessionById: async (sessionId, title) => {
    const trimmed = title.trim();
    if (!trimmed) return;
    try {
      const updated = await renameSession(sessionId, trimmed);
      set((state) => ({
        sessions: state.sessions.map((s) =>
          s.session_id === sessionId ? updated : s,
        ),
      }));
    } catch (err) {
      set({ error: err instanceof Error ? err.message : String(err) });
    }
  },

  deleteSessionById: async (sessionId) => {
    try {
      await deleteSession(sessionId);
    } catch (err) {
      set({ error: err instanceof Error ? err.message : String(err) });
      return;
    }

    const remaining = get().sessions.filter(
      (s) => s.session_id !== sessionId,
    );
    const wasActive = get().currentSessionId === sessionId;

    set({ sessions: remaining });

    if (wasActive) {
      const fallback = remaining[0]?.session_id ?? null;
      if (fallback) {
        await get().selectSession(fallback);
      } else {
        saveActiveSession(null);
        set({ currentSessionId: null, messages: [] });
      }
    }
  },

  sendMessage: async (question) => {
    const trimmed = question.trim();
    if (!trimmed) return;
    if (get().isStreaming) return;

    let sessionId = get().currentSessionId;
    if (!sessionId) {
      const created = await get().startNewChat();
      sessionId = created.session_id;
    }

    const userMessage: ChatMessage = {
      id: makeId(),
      role: "user",
      content: trimmed,
    };
    const assistantId = makeId();
    const assistantPlaceholder: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      isStreaming: true,
    };

    const mode = get().mode;
    set((state) => ({
      messages: [...state.messages, userMessage, assistantPlaceholder],
      isStreaming: true,
      error: null,
    }));

    const updateAssistant = (
      patch: (msg: ChatMessage) => ChatMessage,
    ): void => {
      set((state) => ({
        messages: state.messages.map((m) =>
          m.id === assistantId ? patch(m) : m,
        ),
      }));
    };

    try {
      await streamChat({
        sessionId,
        question: trimmed,
        mode,
        onSources: (event) => {
          updateAssistant((msg) => ({
            ...msg,
            sources: event.sources,
            retrievalQuestion: event.retrieval_question,
          }));
        },
        onToken: (event) => {
          updateAssistant((msg) => ({
            ...msg,
            content: msg.content + event.text,
          }));
        },
        onDone: async (event) => {
          updateAssistant((msg) => ({
            ...msg,
            content: event.full_answer || msg.content,
            trace: event.trace ?? msg.trace,
            agentsInvoked: event.agents_invoked ?? msg.agentsInvoked,
            isStreaming: false,
          }));
          set({ isStreaming: false });
          // Refresh sessions so the (possibly retitled) chat moves to top.
          await get().refreshSessions();
        },
        onError: (event) => {
          updateAssistant((msg) => ({
            ...msg,
            content:
              msg.content || `Sorry, an error occurred: ${event.detail}`,
            isStreaming: false,
          }));
          set({
            isStreaming: false,
            error: event.detail,
          });
        },
      });
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      updateAssistant((msg) => ({
        ...msg,
        content: msg.content || `Sorry, an error occurred: ${detail}`,
        isStreaming: false,
      }));
      set({
        isStreaming: false,
        error: detail,
      });
    }
  },

  clearError: () => set({ error: null }),

  setMode: (mode) => {
    saveChatMode(mode);
    set({ mode });
  },
}));
