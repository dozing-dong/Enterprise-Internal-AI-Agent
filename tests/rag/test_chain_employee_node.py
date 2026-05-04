"""Verify the semantics of the always-run ``employee_retrieve`` node in the RAG chain.

Strategy:
- No Bedrock / real Postgres: assemble ``build_rag_graph`` with fake
  retrievers, a fake EmployeeStore and a fake ChatModel, then ``invoke``
  the full pipeline once and assert that ``retrieved_docs`` contains the
  structured employee entries, and that the main flow still works when
  there are no employee hits.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from backend.rag.chain import build_rag_graph
from backend.rag.employee_retriever import EmployeeRecord
from backend.rag.retrievers import SearchRetriever
from backend.types import RagDocument


class _FakeChatModel:
    def __init__(self, content: str = "fake-answer"):
        self._content = content

    def bind_tools(self, *_a, **_kw):  # pragma: no cover - chain not using tools
        return self

    def with_config(self, **_kw):  # pragma: no cover
        return self

    def invoke(self, *_a, **_kw):
        return AIMessage(self._content)


class _FakeStore:
    def __init__(self, records):
        self._records = records

    def search(self, query, *, department=None, title=None, limit=5):
        return list(self._records)


def _build_graph(monkeypatch, *, employee_records, vector_docs=(), keyword_docs=()):
    from backend.storage import history as history_module

    history_module.set_history_store(history_module.MemoryHistoryStore())

    monkeypatch.setattr(
        "backend.rag.chain.get_chat_model",
        lambda **_: _FakeChatModel(),
    )

    vector_retriever = SearchRetriever(invoke_fn=lambda _q: list(vector_docs))
    keyword_retriever = SearchRetriever(invoke_fn=lambda _q: list(keyword_docs))
    store = _FakeStore(employee_records)

    return build_rag_graph(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        rewrite_chain=None,
        max_iterations=1,
        min_sources=1,
        top_k=5,
        reranker=None,
        rerank_top_k=None,
        employee_store=store,
        employee_top_k=3,
    )


def test_employee_node_injects_records_into_retrieved_docs(monkeypatch):
    records = [
        EmployeeRecord(
            employee_id="E10",
            name="Alice",
            department="Engineering",
            title="SWE",
            email="alice@x",
        )
    ]
    vector_docs = [
        RagDocument(
            page_content="generic kb snippet",
            metadata={"context_id": "kb_1"},
        )
    ]
    graph = _build_graph(
        monkeypatch,
        employee_records=records,
        vector_docs=vector_docs,
        keyword_docs=[],
    )
    state = graph.invoke({"question": "who is Alice", "session_id": "s-emp-1"})
    sources = state["sources"]
    # At least one source must come from the structured employee hit
    # (marked by document_role).
    employee_sources = [
        s for s in sources if s["metadata"].get("document_role") == "employee_structured"
    ]
    assert employee_sources, "employee snippet must appear in sources"
    assert employee_sources[0]["metadata"]["employee_id"] == "E10"


def test_employee_node_no_match_does_not_break_main_flow(monkeypatch):
    """When no employee is found, the main flow must not error and must keep using the regular retrieval results."""
    vector_docs = [
        RagDocument(
            page_content="kb only",
            metadata={"context_id": "kb_only"},
        )
    ]
    graph = _build_graph(
        monkeypatch,
        employee_records=[],
        vector_docs=vector_docs,
        keyword_docs=[],
    )
    state = graph.invoke({"question": "policy on overtime", "session_id": "s-emp-2"})
    sources = state["sources"]
    # There must be no employee sources, but the pipeline must complete
    # normally and return content from vector_docs.
    assert all(
        s["metadata"].get("document_role") != "employee_structured" for s in sources
    )
    assert any(s["metadata"].get("context_id") == "kb_only" for s in sources)
    assert isinstance(state.get("answer"), str)
