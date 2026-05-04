"""FastAPI application factory.

Responsibilities:
- Manage runtime startup and shutdown via lifespan.
- Register global exception handlers.
- Mount business route modules.
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
    """Manage application lifecycle: warm up runtime on startup, clean up on shutdown."""
    init_runtime()
    yield
    # Shutdown stage: release database connection pools or similar resources here if needed.


app = FastAPI(
    title="RAG Demo API",
    version="0.5.0",
    description="Enterprise internal knowledge base Q&A backend with query rewriting, hybrid retrieval, and multi-turn dialogue.",
    lifespan=lifespan,
)

# Allow CORS from the local Vite dev server. Replace with the actual frontend domain in production.
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
    """Map RagException to a structured JSON error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    """Catch RuntimeErrors not handled by routes and uniformly return 503.

    Common scenarios: runtime not initialized, pgvector connection failure,
    Bedrock call timeout.
    """
    return JSONResponse(
        status_code=503,
        content={"detail": f"Service temporarily unavailable: {exc}"},
    )


app.include_router(chat_router)
app.include_router(history_router)
app.include_router(sessions_router)


@app.get("/", response_model=ApiInfoResponse)
def api_info(runtime: Annotated[DemoRuntime, Depends(get_runtime)]) -> ApiInfoResponse:
    """Return basic service info; also used to confirm the service has started."""
    return ApiInfoResponse(
        name="RAG Demo API",
        message="Service is running. Visit /docs for API documentation.",
        execution_mode=runtime.execution_mode,
    )


@app.get("/health")
def health(runtime: Annotated[DemoRuntime, Depends(get_runtime)]) -> dict:
    """Health check endpoint; returns runtime status such as vector store document count."""
    return {
        "status": "ok",
        "vector_document_count": runtime.vector_document_count,
        "raw_document_count": len(runtime.documents),
        "execution_mode": runtime.execution_mode,
    }
