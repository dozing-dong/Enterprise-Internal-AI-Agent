// TypeScript counterparts of the backend Pydantic schemas.
// Keep field names in lockstep with backend/api/schemas.py.

export interface SourceItem {
  rank: number;
  content: string;
  metadata: Record<string, unknown>;
}

export interface SessionItem {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface SessionListResponse {
  sessions: SessionItem[];
}

export interface HistoryMessage {
  role: "user" | "assistant" | string;
  content: string;
}

export interface HistoryResponse {
  session_id: string;
  messages: HistoryMessage[];
}

// Chat execution mode. "rag" = fixed LangGraph pipeline, "agent" = ReAct loop,
// "multi_agent" = Supervisor + Policy + ExternalContext + Writer multi-agent flow.
export type ChatMode = "rag" | "agent" | "multi_agent";

// SSE event payloads emitted by POST /chat/stream.
// Stream is intentionally minimal: token / sources / done / error.

export interface SourcesEvent {
  sources: SourceItem[];
  retrieval_question: string;
  original_question: string;
}

export interface TokenEvent {
  text: string;
}

// Unified trace step shown via the trace popover. RAG / Agent / Multi-Agent
// all surface the same shape so the UI can render once. ``agent`` is only
// populated in multi_agent mode (supervisor / policy / external_context / writer).
export interface TraceStep {
  step: number;
  name: string;
  input_summary?: string | null;
  output_summary?: string | null;
  ok?: boolean;
  latency_ms?: number | null;
  error?: string | null;
  agent?: string | null;
}

export interface DoneEvent {
  session_id: string;
  title: string;
  full_answer: string;
  original_question: string;
  retrieval_question: string;
  mode?: ChatMode;
  trace?: TraceStep[];
  // Sub-agents invoked during a multi_agent turn (in order, deduplicated).
  agents_invoked?: string[];
}

export interface ErrorEvent {
  detail: string;
}

// UI-side message model used by the chat store.
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceItem[];
  retrievalQuestion?: string;
  trace?: TraceStep[];
  // Sub-agents involved in this assistant turn (multi_agent mode only).
  agentsInvoked?: string[];
  isStreaming?: boolean;
}
