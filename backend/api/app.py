from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import (
    FRONTEND_DIR,
)
from backend.runtime import DemoRuntime, create_demo_runtime
from backend.storage.history import (
    build_history_path,
    clear_session_history,
    read_session_history,
)


class ChatRequest(BaseModel):
    """聊天请求体。"""

    # question 是用户输入的问题内容。
    question: str = Field(..., min_length=1, description="用户输入的问题")

    # session_id 用来区分不同会话。
    session_id: str = Field(default="default", description="会话 ID")


class SourceItem(BaseModel):
    """单个检索来源。"""

    # rank 表示这个片段在检索结果中的顺序。
    rank: int

    # content 是实际检索到的文本内容。
    content: str

    # metadata 保留原始文档的来源信息。
    metadata: dict


class ChatResponse(BaseModel):
    """聊天响应体。"""

    # answer 是模型最终返回的文本答案。
    answer: str

    # original_question 是用户原始问题。
    original_question: str

    # retrieval_question 是实际用于检索的问题。
    # 这样你在学习阶段就能直接观察“改写前”和“改写后”的差别。
    retrieval_question: str

    # session_id 原样返回，方便前端继续复用这个会话。
    session_id: str

    # history_file 返回历史文件路径，便于学习阶段直接查看。
    history_file: str

    # sources 返回这次回答实际参考过的检索片段。
    sources: list[SourceItem]


class HistoryResponse(BaseModel):
    """历史记录响应体。"""

    session_id: str
    messages: list[dict]


class ClearHistoryResponse(BaseModel):
    """清空历史的响应体。"""

    message: str
    session_id: str


app = FastAPI(
    title="TEST01 RAG Demo API",
    version="0.4.0",
    description="最小 RAG 后端服务，包含查询改写、两阶段检索和静态前端页面。",
)

demo_runtime: DemoRuntime | None = None


def get_demo_runtime() -> DemoRuntime:
    """获取已经初始化好的运行时对象。"""
    global demo_runtime
    if demo_runtime is None:
        demo_runtime = create_demo_runtime(execution_mode="langgraph")
    return demo_runtime


@app.on_event("startup")
def startup_event() -> None:
    """应用启动时提前初始化 RAG 所需对象。"""
    get_demo_runtime()
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
def home_page():
    """返回前端首页。"""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api")
def api_info() -> dict:
    """提供一个最简单的 API 信息接口。"""
    return {
        "name": "TEST01 RAG Demo API",
        "message": "服务已经启动，可以访问 /docs 查看接口文档。",
        "execution_mode": "langgraph",
    }


@app.get("/health")
def health() -> dict:
    """健康检查接口。"""
    runtime = get_demo_runtime()

    return {
        "status": "ok",
        "vector_document_count": runtime.vector_document_count,
        "raw_document_count": len(runtime.documents),
        "execution_mode": runtime.execution_mode,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """最小聊天接口。"""
    runtime = get_demo_runtime()

    try:
        result = runtime.chat_executor(request.question, request.session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    history_file = build_history_path(request.session_id)

    return ChatResponse(
        answer=result["answer"],
        original_question=result["original_question"],
        retrieval_question=result["retrieval_question"],
        session_id=request.session_id,
        history_file=str(history_file),
        sources=result["sources"],
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str) -> HistoryResponse:
    """读取指定会话的历史记录。"""
    messages = read_session_history(session_id)

    return HistoryResponse(
        session_id=session_id,
        messages=messages,
    )


@app.delete("/history/{session_id}", response_model=ClearHistoryResponse)
def delete_history(session_id: str) -> ClearHistoryResponse:
    """清空指定会话的历史记录。"""
    clear_session_history(session_id)

    return ClearHistoryResponse(
        message="会话历史已清空。",
        session_id=session_id,
    )
