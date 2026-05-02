"""API 层共享的 Pydantic 请求/响应模型。"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# 支持的执行模式：
# - rag：现有固定流水线（默认，保持向后兼容）
# - agent：模型自决 + 工具调用循环
ChatMode = Literal["rag", "agent"]


class ChatRequest(BaseModel):
    """聊天请求体。"""

    question: str = Field(..., min_length=1, description="用户输入的问题")
    session_id: str = Field(default="default", description="会话 ID")
    mode: ChatMode = Field(
        default="rag",
        description="执行模式：rag = 固定流水线；agent = 模型自决 + 工具调用",
    )


class ChatStreamRequest(BaseModel):
    """流式聊天请求体；session_id 必填，由前端先调用 POST /sessions 取得。"""

    question: str = Field(..., min_length=1, description="用户输入的问题")
    session_id: str = Field(..., min_length=1, description="规范化的会话 ID")
    mode: ChatMode = Field(
        default="rag",
        description="执行模式：rag = 固定流水线；agent = 模型自决 + 工具调用",
    )


class SourceItem(BaseModel):
    """单个检索来源。"""

    rank: int
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    """聊天响应体。"""

    answer: str
    original_question: str
    # 实际用于检索的问题，方便观察查询改写前后的差别。
    retrieval_question: str
    session_id: str
    history_file: str
    sources: list[SourceItem]
    # 仅 agent 模式下填充：每一步的决策日志（thought / tool_call / tool_result / final）。
    decision_trace: list[dict] | None = None
    # 仅 agent 模式下填充：表示是否触发了 fallback 到 RAG 流水线。
    fallback: bool | None = None
    # 用于前端确认本轮实际走的是哪种模式（agent 失败回退仍标 agent）。
    mode: ChatMode | None = None


class HistoryResponse(BaseModel):
    """历史记录响应体。"""

    session_id: str
    messages: list[dict]


class ClearHistoryResponse(BaseModel):
    """清空历史的响应体。"""

    message: str
    session_id: str


class ApiInfoResponse(BaseModel):
    """根路由返回的服务信息。"""

    name: str
    message: str
    execution_mode: str


class SessionItem(BaseModel):
    """单条会话元数据，用于侧边栏展示。"""

    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    """会话列表响应体。"""

    sessions: list[SessionItem]


class CreateSessionRequest(BaseModel):
    """创建会话的可选请求体。"""

    title: str | None = Field(
        default=None,
        max_length=80,
        description="可选的初始标题；不传则使用 'New Chat'。",
    )


class RenameSessionRequest(BaseModel):
    """重命名会话的请求体。"""

    title: str = Field(..., min_length=1, max_length=80)


class DeleteSessionResponse(BaseModel):
    """删除会话的响应体。"""

    message: str
    session_id: str
