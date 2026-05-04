"""RAG LangGraph: rewrite -> hybrid retrieve -> rerank -> generate -> persist.

Design notes:
- The ``generate_answer`` node uses ``ChatBedrockConverse.invoke``; the
  outer caller can capture ``AIMessageChunk`` automatically under
  ``stream_mode=["messages","updates"]`` without writing custom streaming logic.
- Node completion events (``updates``) carry per-step ``tool_trace``
  increments that the orchestrator aggregates into a unified trace list.
- The entry function is ``build_rag_graph`` and returns the compiled LangGraph application.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from backend.config import EMPLOYEE_LOOKUP_TOP_K
from backend.data.processing import convert_docs_to_sources, format_docs
from backend.llm import get_chat_model
from backend.rag.employee_retriever import (
    EmployeeStore,
    employee_records_to_documents,
    safe_search_employees,
)
from backend.rag.rerank import Reranker
from backend.rag.retrievers import SearchRetriever, fuse_retrieval_results
from backend.rag.rewrite import rewrite_question_for_retrieval
from backend.storage.history import append_session_messages, read_session_history
from backend.types import RagDocument


GENERATION_SYSTEM_PROMPT = (
    "You are an assistant that answers questions based on retrieval results. "
    "Prefer to rely on the provided knowledge base snippets and the conversation history. "
    "When a reference snippet describes an employee profile (e.g. starts with "
    "'Employee E####:' or contains explicit department/title/email fields), "
    "treat it as authoritative structured data: only use the names, "
    "departments, titles, and emails that actually appear in the snippets. "
    "If no employee snippet was retrieved for an employee-related question, "
    "tell the user the directory has no matching record instead of guessing. "
    "If the reference content is not sufficient to support a conclusion, "
    "clearly say that you do not know and do not fabricate an answer."
)


class GraphState(TypedDict, total=False):
    question: str
    session_id: str
    original_question: str
    retrieval_question: str
    retrieved_docs: list
    vector_docs: list
    keyword_docs: list
    employee_docs: list
    context: str
    sources: list[dict]
    history: list[dict]
    answer: str
    tool_trace: Annotated[list[dict], operator.add]
    iteration: int
    should_continue: bool


def _doc_signature(document: RagDocument) -> tuple[str, str]:
    """Dedup key matching ``retrievers._doc_key``; defined here to avoid a circular import."""
    context_id = str(document.metadata.get("context_id", ""))
    return context_id, document.page_content


def _merge_unique_docs(
    primary: list[RagDocument],
    secondary: list[RagDocument],
) -> list[RagDocument]:
    """Deduplicate by ``_doc_signature``: ``primary`` first, then ``secondary``."""
    seen: set[tuple[str, str]] = set()
    merged: list[RagDocument] = []
    for source in (primary, secondary):
        for document in source:
            key = _doc_signature(document)
            if key in seen:
                continue
            seen.add(key)
            merged.append(document)
    return merged


def build_rag_graph(
    vector_retriever: SearchRetriever,
    keyword_retriever: SearchRetriever,
    rewrite_chain: Callable[[str], str] | None,
    max_iterations: int,
    min_sources: int,
    top_k: int,
    reranker: Reranker | None = None,
    rerank_top_k: int | None = None,
    employee_store: EmployeeStore | None = None,
    employee_top_k: int = EMPLOYEE_LOOKUP_TOP_K,
):
    """Build and compile the RAG LangGraph.

    The returned object supports both ``.invoke({...})`` and
    ``.stream/.astream(stream_mode=...)``, used by ``RagAnswerTool`` for
    synchronous calls and by the orchestrator for streaming.

    If ``employee_store`` is injected, each turn runs a structured
    employee lookup once (in parallel with vector and keyword retrieval);
    matched results are merged into ``retrieved_docs`` and forcibly
    preserved after rerank. A miss does not affect the main flow.
    """
    chat_model = get_chat_model(temperature=0.0)
    graph = StateGraph(GraphState)

    def route_intent(state: GraphState) -> dict:
        return {
            "retrieval_question": state["question"],
            "iteration": state.get("iteration", 0),
        }

    def load_history(state: GraphState) -> dict:
        history = read_session_history(state["session_id"])
        return {"history": history}

    def rewrite_query(state: GraphState) -> dict:
        retrieval_question = rewrite_question_for_retrieval(
            state["question"],
            rewrite_chain,
        )
        return {
            "retrieval_question": retrieval_question,
            "tool_trace": [
                {
                    "tool": "rewrite_query",
                    "input": state["question"],
                    "output": retrieval_question,
                }
            ],
        }

    def vector_retrieve(state: GraphState) -> dict:
        retrieval_question = state.get("retrieval_question", state["question"])
        docs = vector_retriever.invoke(retrieval_question)
        return {
            "vector_docs": docs,
            "tool_trace": [
                {
                    "tool": "vector_retrieve",
                    "input": retrieval_question,
                    "output_count": len(docs),
                }
            ],
        }

    def keyword_retrieve(state: GraphState) -> dict:
        retrieval_question = state.get("retrieval_question", state["question"])
        docs = keyword_retriever.invoke(retrieval_question)
        return {
            "keyword_docs": docs,
            "tool_trace": [
                {
                    "tool": "keyword_retrieve",
                    "input": retrieval_question,
                    "output_count": len(docs),
                }
            ],
        }

    def employee_retrieve(state: GraphState) -> dict:
        """Structured employee lookup that runs every turn.

        - When ``employee_store`` is not injected, returns immediately;
          the node is a no-op.
        - Any error is swallowed by ``safe_search_employees`` into an
          empty result, avoiding interruption of the main pipeline; a
          miss is still considered normal.
        """
        if employee_store is None:
            return {"employee_docs": []}

        original_question = state["question"]
        retrieval_question = state.get("retrieval_question", original_question)

        # Employee identification preferentially looks at the user's
        # original question (it usually contains "my name is xxx");
        # if no hit, fall back to the rewritten query.
        records = safe_search_employees(
            employee_store,
            original_question,
            limit=employee_top_k,
        )
        used_query = original_question
        if not records and retrieval_question != original_question:
            records = safe_search_employees(
                employee_store,
                retrieval_question,
                limit=employee_top_k,
            )
            used_query = retrieval_question

        docs = employee_records_to_documents(records, query=used_query)
        return {
            "employee_docs": docs,
            "tool_trace": [
                {
                    "tool": "employee_retrieve",
                    "input": used_query,
                    "output_count": len(docs),
                }
            ],
        }

    def fuse_docs(state: GraphState) -> dict:
        vector_docs = state.get("vector_docs", [])
        keyword_docs = state.get("keyword_docs", [])
        employee_docs = state.get("employee_docs", [])
        fused = fuse_retrieval_results(
            vector_docs,
            keyword_docs,
            top_k=top_k,
        )
        # Structured employee results are "must-include evidence"; force
        # them to the top and dedup. When there is no hit, this step is
        # equivalent to ``fused`` unchanged, matching the
        # "ignore-on-miss" semantic.
        docs = _merge_unique_docs(employee_docs, fused)
        return {
            "retrieved_docs": docs,
            "context": format_docs(docs),
            "sources": convert_docs_to_sources(docs),
            "tool_trace": [
                {
                    "tool": "fuse_docs",
                    "input_vector_count": len(vector_docs),
                    "input_keyword_count": len(keyword_docs),
                    "input_employee_count": len(employee_docs),
                    "output_count": len(docs),
                }
            ],
            "iteration": state.get("iteration", 0) + 1,
        }

    def rerank_docs(state: GraphState) -> dict:
        candidates = state.get("retrieved_docs", [])
        if reranker is None or not candidates:
            return {}

        final_k = rerank_top_k if rerank_top_k is not None else top_k
        retrieval_question = state.get("retrieval_question", state["question"])
        reranked = reranker.invoke(retrieval_question, candidates, final_k)

        # Treat structured employee hits as "must-include evidence";
        # force them to be preserved and pinned to the top after rerank,
        # even if their relevance score is low.
        employee_docs = state.get("employee_docs", [])
        final_docs = _merge_unique_docs(employee_docs, reranked)

        return {
            "retrieved_docs": final_docs,
            "context": format_docs(final_docs),
            "sources": convert_docs_to_sources(final_docs),
            "tool_trace": [
                {
                    "tool": "rerank_docs",
                    "input_count": len(candidates),
                    "output_count": len(final_docs),
                    "preserved_employee_count": len(employee_docs),
                }
            ],
        }

    def quality_gate(state: GraphState) -> dict:
        sources = state.get("sources", [])
        iteration = state.get("iteration", 0)
        should_continue = len(sources) < min_sources and iteration < max_iterations
        return {"should_continue": should_continue}

    def generate_answer(state: GraphState) -> dict:
        context = state.get("context", "No relevant knowledge snippets were retrieved.")
        history = state.get("history", [])

        chat_messages = [SystemMessage(GENERATION_SYSTEM_PROMPT)]
        for item in history:
            role = item.get("role")
            content = item.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            if role == "assistant":
                from langchain_core.messages import AIMessage

                chat_messages.append(AIMessage(content))
            elif role == "user":
                chat_messages.append(HumanMessage(content))

        chat_messages.append(
            HumanMessage(
                f"Question: {state['question']}\n\n"
                f"Reference knowledge:\n{context}\n\n"
                "Please answer the question based on the reference knowledge above."
            )
        )

        ai_msg = chat_model.invoke(chat_messages)
        text = _coerce_to_text(ai_msg.content)

        return {
            "answer": text or "Failed to generate a usable answer.",
            "tool_trace": [
                {
                    "tool": "generate_answer",
                    "output_chars": len(text),
                }
            ],
        }

    def save_history(state: GraphState) -> dict:
        append_session_messages(
            state["session_id"],
            [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": state.get("answer", "")},
            ],
        )
        return {}

    def finalize(state: GraphState) -> dict:
        return {
            "answer": state.get("answer", "Failed to generate a usable answer."),
            "sources": state.get("sources", []),
            "original_question": state["question"],
            "retrieval_question": state.get("retrieval_question", state["question"]),
        }

    graph.add_node("route_intent", route_intent)
    graph.add_node("load_history", load_history)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("vector_retrieve", vector_retrieve)
    graph.add_node("keyword_retrieve", keyword_retrieve)
    graph.add_node("employee_retrieve", employee_retrieve)
    graph.add_node("fuse_docs", fuse_docs)
    graph.add_node("rerank_docs", rerank_docs)
    graph.add_node("quality_gate", quality_gate)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("save_history", save_history)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("route_intent")
    graph.add_edge("route_intent", "load_history")
    graph.add_edge("load_history", "rewrite_query")
    graph.add_edge("rewrite_query", "vector_retrieve")
    graph.add_edge("rewrite_query", "keyword_retrieve")
    graph.add_edge("rewrite_query", "employee_retrieve")
    graph.add_edge("vector_retrieve", "fuse_docs")
    graph.add_edge("keyword_retrieve", "fuse_docs")
    graph.add_edge("employee_retrieve", "fuse_docs")
    graph.add_edge("fuse_docs", "rerank_docs")
    graph.add_edge("rerank_docs", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        lambda state: "rewrite_query" if state.get("should_continue") else "generate_answer",
        {
            "rewrite_query": "rewrite_query",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("generate_answer", "save_history")
    graph.add_edge("save_history", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


def _coerce_to_text(content) -> str:
    """Bedrock Converse ``content`` may be a string or a list of blocks."""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                value = block.get("text", "")
                parts.append(str(value))
            else:
                parts.append(str(block))
        return "".join(parts).strip()
    return str(content or "").strip()
