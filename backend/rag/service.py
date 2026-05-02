"""RAG 统一入口：非流式走 LangGraph 编译链，流式走与历史一致的手写流水线。

``run`` 等价于 ``chat_executor``；``stream`` 产出与路由层约定的 ``(event_type, data)``
元组（仅 progress / sources / token），历史写入在 token 结束后于本模块内完成，
``touch_session`` / 标题 / 最终 ``done`` 由 orchestrator 统一处理。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from backend.config import RETRIEVER_CANDIDATE_K, RERANK_TOP_K, RETRIEVER_TOP_K
from backend.data.processing import convert_docs_to_sources, format_docs
from backend.rag.models import chat_completion_stream
from backend.rag.retrievers import SearchRetriever, fuse_retrieval_results
from backend.rag.rewrite import rewrite_question_for_retrieval
from backend.storage.history import append_session_messages

# 与 backend/rag/chain.py 中 generate_answer 节点保持一致。
GENERATION_SYSTEM_PROMPT = (
    "You are an assistant that answers questions based on retrieval results. "
    "Prefer to rely on the provided knowledge base snippets and the conversation history. "
    "If the reference content is not sufficient to support a conclusion, "
    "clearly say that you do not know and do not fabricate an answer."
)


def build_user_message(question: str, context: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"Question: {question}\n\n"
            f"Reference knowledge:\n{context}\n\n"
            "Please answer the question based on the reference knowledge above."
        ),
    }


ChatExecutor = Callable[[str, str], dict[str, Any]]


class RagService:
    """外层统一调用的 RAG 门面：``run`` 委托编译链；``stream`` 手写检索 + 流式生成。"""

    def __init__(
        self,
        *,
        chat_executor: ChatExecutor,
        vector_retriever: SearchRetriever,
        keyword_retriever: SearchRetriever,
        rewrite_chain: Any | None,
        reranker: Any | None,
    ) -> None:
        self._chat_executor = chat_executor
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        self._rewrite_chain = rewrite_chain
        self._reranker = reranker

    def run(self, question: str, session_id: str) -> dict[str, Any]:
        return self._chat_executor(question, session_id)

    def stream(
        self,
        question: str,
        session_id: str,
        *,
        existing_history: list[dict],
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """流式 RAG：yield (event_type, data)，event_type 为 progress | sources | token。"""
        yield (
            "progress",
            {"stage": "rewriting", "message": "Refining your query..."},
        )
        retrieval_question = rewrite_question_for_retrieval(
            question,
            self._rewrite_chain,
        )

        yield (
            "progress",
            {"stage": "retrieving", "message": "Searching the knowledge base..."},
        )
        vector_docs = self._vector_retriever.invoke(retrieval_question)
        keyword_docs = self._keyword_retriever.invoke(retrieval_question)
        candidate_top_k = (
            max(RETRIEVER_CANDIDATE_K, RERANK_TOP_K)
            if self._reranker is not None
            else RETRIEVER_TOP_K
        )
        fused_docs = fuse_retrieval_results(
            vector_docs,
            keyword_docs,
            top_k=candidate_top_k,
        )

        final_docs = fused_docs
        if self._reranker is not None and fused_docs:
            yield (
                "progress",
                {
                    "stage": "reranking",
                    "message": "Reranking retrieved snippets...",
                },
            )
            final_docs = self._reranker.invoke(
                retrieval_question,
                fused_docs,
                RERANK_TOP_K,
            )

        sources = convert_docs_to_sources(final_docs)
        context = format_docs(final_docs)

        yield (
            "sources",
            {
                "sources": sources,
                "retrieval_question": retrieval_question,
                "original_question": question,
            },
        )

        yield (
            "progress",
            {"stage": "generating", "message": "Generating answer..."},
        )

        messages = list(existing_history)
        messages.append(build_user_message(question, context))

        answer_parts: list[str] = []
        for token in chat_completion_stream(
            messages,
            system_prompt=GENERATION_SYSTEM_PROMPT,
            temperature=0.0,
        ):
            answer_parts.append(token)
            yield ("token", {"text": token})

        full_answer = "".join(answer_parts).strip() or "No answer generated."

        append_session_messages(
            session_id,
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": full_answer},
            ],
        )
