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

// Chat execution mode. "rag" keeps the legacy fixed pipeline,
// "agent" enables the LLM-driven tool-calling loop on the backend.
export type ChatMode = "rag" | "agent";

// SSE event payloads emitted by POST /chat/stream.
//
// Stages in the agent flow (`deciding`, `tool_running`) replace the
// per-stage RAG indicators (`rewriting`, `retrieving`, ...) but reuse the
// same `progress` event so the existing UI hook stays compatible.
export type StreamStage =
  | "rewriting"
  | "retrieving"
  | "reranking"
  | "generating"
  | "titling"
  | "deciding"
  | "tool_running";

export interface ProgressEvent {
  stage: StreamStage;
  message?: string;
}

export interface SourcesEvent {
  sources: SourceItem[];
  retrieval_question: string;
  original_question: string;
}

export interface TokenEvent {
  text: string;
}

// Emitted when the agent decides to invoke a tool.
// `arguments` is the parsed JSON object the model sent for that tool.
export interface ToolCallEvent {
  step: number;
  name: string;
  arguments: Record<string, unknown>;
  tool_use_id: string;
}

// Emitted right after a tool finishes, regardless of success.
// `summary` contains tool-specific lightweight fields suitable for the UI
// (e.g. `sources_count` for `rag_answer`); the full data stays server-side.
export interface ToolResultEvent {
  step: number;
  name: string;
  ok: boolean;
  error?: string | null;
  latency_ms?: number | null;
  summary?: Record<string, unknown>;
}

// Single entry in the agent decision trace, mirroring AgentStep.to_dict().
export interface DecisionStep {
  index: number;
  thought?: string | null;
  tool_call?: {
    name: string;
    arguments: Record<string, unknown>;
    tool_use_id: string;
  } | null;
  tool_result?: {
    name: string;
    ok: boolean;
    data: Record<string, unknown>;
    error?: string | null;
    latency_ms?: number | null;
  } | null;
  final_answer?: string | null;
  fallback?: boolean;
  reason?: string;
}

export interface DoneEvent {
  session_id: string;
  title: string;
  full_answer: string;
  original_question: string;
  retrieval_question: string;
  // Agent-mode only fields; absent in legacy rag responses.
  mode?: ChatMode;
  decision_trace?: DecisionStep[];
  fallback?: boolean;
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
  isStreaming?: boolean;
}
