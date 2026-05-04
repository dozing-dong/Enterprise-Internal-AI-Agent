"""Shared Pydantic request/response models for the API layer."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# Supported execution modes:
# - rag: fixed LangGraph pipeline (rewrite -> hybrid retrieve -> rerank -> generate)
# - agent: model-driven + tool-calling loop (ReAct)
# - multi_agent: Supervisor + Policy + ExternalContext + Writer multi-agent orchestration
ChatMode = Literal["rag", "agent", "multi_agent"]


class ChatRequest(BaseModel):
    """Chat request body (shared by POST /chat and POST /chat/stream)."""

    question: str = Field(..., min_length=1, description="User input question")
    session_id: str = Field(default="default", description="Session ID")
    mode: ChatMode = Field(
        default="rag",
        description="Execution mode: rag = fixed pipeline; agent = model-driven + tool calling",
    )


class ChatStreamRequest(BaseModel):
    """Streaming chat request body; session_id is required and must be obtained via POST /sessions first."""

    question: str = Field(..., min_length=1, description="User input question")
    session_id: str = Field(..., min_length=1, description="Normalized session ID")
    mode: ChatMode = Field(
        default="rag",
        description="Execution mode: rag = fixed pipeline; agent = model-driven + tool calling",
    )


class SourceItem(BaseModel):
    """A single retrieval source."""

    rank: int
    content: str
    metadata: dict


class TraceStepModel(BaseModel):
    """A single unified call trace step; mirrors ``backend/orchestrator/trace.TraceStep``."""

    step: int
    name: str
    input_summary: str | None = None
    output_summary: str | None = None
    ok: bool = True
    latency_ms: int | None = None
    error: str | None = None
    # In multi_agent mode, marks which sub-agent this step belongs to
    # (supervisor / policy / external_context / writer). None in other modes.
    agent: str | None = None


class ChatResponse(BaseModel):
    """Chat response body."""

    answer: str
    original_question: str
    # The question actually used for retrieval; useful for observing
    # the difference before/after query rewriting.
    retrieval_question: str
    session_id: str
    history_file: str
    sources: list[SourceItem]
    # Unified call trace (same format across RAG / Agent / Multi-Agent modes).
    trace: list[TraceStepModel] = Field(default_factory=list)
    # Lets the frontend confirm which mode this turn actually used.
    mode: ChatMode | None = None
    # In multi_agent mode, the ordered list of sub-agents that were invoked
    # (deduplicated, in order of first appearance). Empty in other modes.
    agents_invoked: list[str] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    """Chat history response body."""

    session_id: str
    messages: list[dict]


class ClearHistoryResponse(BaseModel):
    """Response body for clearing history."""

    message: str
    session_id: str


class ApiInfoResponse(BaseModel):
    """Service info returned by the root route."""

    name: str
    message: str
    execution_mode: str


class SessionItem(BaseModel):
    """A single session metadata entry; used by the sidebar."""

    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    """Session list response body."""

    sessions: list[SessionItem]


class CreateSessionRequest(BaseModel):
    """Optional request body for creating a session."""

    title: str | None = Field(
        default=None,
        max_length=80,
        description="Optional initial title; defaults to 'New Chat' when omitted.",
    )


class RenameSessionRequest(BaseModel):
    """Request body for renaming a session."""

    title: str = Field(..., min_length=1, max_length=80)


class DeleteSessionResponse(BaseModel):
    """Response body for deleting a session."""

    message: str
    session_id: str
