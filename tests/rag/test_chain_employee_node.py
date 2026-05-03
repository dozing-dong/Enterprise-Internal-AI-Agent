"""验证 RAG chain 中 ``employee_retrieve`` 必查节点的语义。

策略：
- 不接 Bedrock / 真实 PG：用 fake retrievers + fake EmployeeStore + fake
  ChatModel 装配 ``build_rag_graph``，直接 ``invoke`` 一次完整流水线，
  断言 ``retrieved_docs`` 中包含员工结构化条目，且查不到时主流程不挂。
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
    # 至少一条 sources 来自员工结构化命中（document_role 标记）。
    employee_sources = [
        s for s in sources if s["metadata"].get("document_role") == "employee_structured"
    ]
    assert employee_sources, "employee snippet must appear in sources"
    assert employee_sources[0]["metadata"]["employee_id"] == "E10"


def test_employee_node_no_match_does_not_break_main_flow(monkeypatch):
    """员工查不到时，主流程不报错且沿用普通检索结果。"""
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
    # 不应该有员工 source，但流水线必须正常完成并返回 vector_docs 里的内容。
    assert all(
        s["metadata"].get("document_role") != "employee_structured" for s in sources
    )
    assert any(s["metadata"].get("context_id") == "kb_only" for s in sources)
    assert isinstance(state.get("answer"), str)
