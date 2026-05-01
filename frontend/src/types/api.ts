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

// SSE event payloads emitted by POST /chat/stream.
export type StreamStage =
  | "rewriting"
  | "retrieving"
  | "reranking"
  | "generating"
  | "titling";

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

export interface DoneEvent {
  session_id: string;
  title: string;
  full_answer: string;
  original_question: string;
  retrieval_question: string;
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
