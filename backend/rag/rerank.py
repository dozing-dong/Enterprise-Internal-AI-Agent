"""Document reranking module.

Reranks the candidate documents from the recall stage by query
relevance and keeps only the top_k. Provides a unified ``Reranker``
abstraction; currently has a built-in Bedrock Rerank implementation.

When the reranker is unavailable (API exception / no candidates), the
caller automatically falls back to the truncated original order, so the
RAG pipeline does not break because rerank failed.
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
    """Lightweight reranker protocol unifying the entry point across backends."""

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
    """Fallback for rerank failures: keep the original order, truncated to top_k."""
    return list(docs[:top_k])


def build_bedrock_reranker(
    model_id: str = BEDROCK_RERANK_MODEL_ID,
) -> Reranker:
    """Build a reranker that calls the Bedrock Rerank API."""

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
                "Bedrock rerank call failed; falling back to original-order truncation: %s", exc,
            )
            return _fallback_truncate(docs, top_k)

        if not scored:
            logger.warning("Bedrock rerank returned an empty result; falling back to original-order truncation.")
            return _fallback_truncate(docs, top_k)

        reranked: list[RagDocument] = []
        for new_rank, (index, score) in enumerate(scored, start=1):
            reranked.append(_attach_rerank_score(docs[index], score, new_rank))
        return reranked

    return Reranker(rerank_fn=rerank_fn)


def build_reranker(backend: str) -> Reranker:
    """Pick a reranker implementation based on configuration.

    Currently only ``bedrock`` is supported; to extend with a local
    cross-encoder / LLM rerank, add a branch here-callers do not need
    to be aware of the difference.
    """
    backend_normalized = (backend or "").strip().lower()
    if backend_normalized == "bedrock":
        return build_bedrock_reranker()
    raise ValueError(f"Unknown RERANK_BACKEND: {backend!r}")
