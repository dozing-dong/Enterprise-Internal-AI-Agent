"""文档重排序模块。

把召回阶段拿到的候选文档，按 query 相关度做精排后只保留 top_k。
对外提供统一的 ``Reranker`` 抽象，当前内置 Bedrock Rerank 实现。

调用方在 reranker 不可用时（API 异常 / 候选为空）会自动回退到截断后的原始顺序，
不会让整条 RAG 链路因为重排失败而中断。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from backend.config import BEDROCK_RERANK_MODEL_ID
from backend.llm import bedrock_rerank
from backend.types import RagDocument


logger = logging.getLogger(__name__)


RerankFn = Callable[[str, list[RagDocument], int], list[RagDocument]]


@dataclass(slots=True)
class Reranker:
    """轻量重排器协议，统一不同后端实现的调用入口。"""

    rerank_fn: RerankFn

    def invoke(
        self,
        query: str,
        docs: list[RagDocument],
        top_k: int,
    ) -> list[RagDocument]:
        if not docs or top_k <= 0:
            return list(docs)
        return self.rerank_fn(query, docs, top_k)


def _attach_rerank_score(doc: RagDocument, score: float, rank: int) -> RagDocument:
    metadata = dict(doc.metadata)
    metadata["rerank_score"] = float(score)
    metadata["rerank_rank"] = rank
    return RagDocument(page_content=doc.page_content, metadata=metadata)


def _fallback_truncate(docs: list[RagDocument], top_k: int) -> list[RagDocument]:
    """重排失败时的兜底：保持原顺序截断到 top_k。"""
    return list(docs[:top_k])


def build_bedrock_reranker(
    model_id: str = BEDROCK_RERANK_MODEL_ID,
) -> Reranker:
    """构建一个调用 Bedrock Rerank API 的重排器。"""

    def rerank_fn(query: str, docs: list[RagDocument], top_k: int) -> list[RagDocument]:
        documents = [doc.page_content for doc in docs]

        try:
            scored = bedrock_rerank(
                query,
                documents,
                model_id=model_id,
                top_k=top_k,
            )
        except Exception as exc:
            logger.warning(
                "Bedrock rerank 调用失败，回退到原始顺序截断: %s", exc,
            )
            return _fallback_truncate(docs, top_k)

        if not scored:
            logger.warning("Bedrock rerank 返回空结果，回退到原始顺序截断。")
            return _fallback_truncate(docs, top_k)

        reranked: list[RagDocument] = []
        for new_rank, (index, score) in enumerate(scored, start=1):
            reranked.append(_attach_rerank_score(docs[index], score, new_rank))
        return reranked

    return Reranker(rerank_fn=rerank_fn)


def build_reranker(backend: str) -> Reranker:
    """根据配置选择重排实现。

    目前仅支持 ``bedrock``；后续要扩展本地 cross-encoder / LLM rerank 时，
    在这里加分支即可，调用侧无需感知差异。
    """
    backend_normalized = (backend or "").strip().lower()
    if backend_normalized == "bedrock":
        return build_bedrock_reranker()
    raise ValueError(f"未知的 RERANK_BACKEND: {backend!r}")
