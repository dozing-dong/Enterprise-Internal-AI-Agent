"""FastAPI 应用工厂。

职责：
- 通过 lifespan 管理运行时的启动与关闭。
- 注册全局异常处理器。
- 挂载各业务路由模块。
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.dependencies import get_runtime, init_runtime
from backend.api.exceptions import RagException
from backend.api.routes.chat import router as chat_router
from backend.api.routes.history import router as history_router
from backend.api.routes.sessions import router as sessions_router
from backend.api.schemas import ApiInfoResponse
from backend.runtime import DemoRuntime


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用生命周期：启动时预热运行时，关闭时可做清理。"""
    init_runtime()
    yield
    # 关闭阶段：如有需要可在此处释放数据库连接池等资源。


app = FastAPI(
    title="RAG Demo API",
    version="0.5.0",
    description="企业内部知识库问答系统后端，包含查询改写、混合检索和多轮对话。",
    lifespan=lifespan,
)

# 允许本地 Vite 开发服务器跨域访问。生产部署时应换成实际前端域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
    expose_headers=["Content-Type"],
)


@app.exception_handler(RagException)
async def rag_exception_handler(request: Request, exc: RagException) -> JSONResponse:
    """将 RagException 映射为结构化的 JSON 错误响应。"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    """捕获未被路由层处理的 RuntimeError，统一返回 503。

    常见场景：运行时未初始化、pgvector 连接失败、Bedrock 调用超时。
    """
    return JSONResponse(
        status_code=503,
        content={"detail": f"服务暂时不可用：{exc}"},
    )


app.include_router(chat_router)
app.include_router(history_router)
app.include_router(sessions_router)


@app.get("/", response_model=ApiInfoResponse)
def api_info(runtime: Annotated[DemoRuntime, Depends(get_runtime)]) -> ApiInfoResponse:
    """返回服务基本信息，也用于确认服务已启动。"""
    return ApiInfoResponse(
        name="RAG Demo API",
        message="服务已启动，可访问 /docs 查看接口文档。",
        execution_mode=runtime.execution_mode,
    )


@app.get("/health")
def health(runtime: Annotated[DemoRuntime, Depends(get_runtime)]) -> dict:
    """健康检查接口，返回向量库文档数量等运行状态。"""
    return {
        "status": "ok",
        "vector_document_count": runtime.vector_document_count,
        "raw_document_count": len(runtime.documents),
        "execution_mode": runtime.execution_mode,
    }
