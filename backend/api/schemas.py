"""API 层共享的 Pydantic 请求/响应模型。"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求体。"""

    question: str = Field(..., min_length=1, description="用户输入的问题")
    session_id: str = Field(default="default", description="会话 ID")


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
