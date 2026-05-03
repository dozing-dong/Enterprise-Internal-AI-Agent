"""项目运行时装配。

把所有依赖一次性构造好暴露给 CLI / API：
- ``rag_graph``：编译好的 RAG LangGraph，``.invoke()`` 与 ``.stream()`` 通用。
- ``agent_graph``：编译好的 Agent ReAct LangGraph；其工具集中包含
  ``rag_answer``，由 ``rag_graph`` 注入；不再走任何 fallback。
"""

from typing import Any

from backend.agent import build_agent_graph, build_rag_answer_tool, current_time
from backend.config import (
    LANGGRAPH_MAX_ITERATIONS,
    LANGGRAPH_MIN_SOURCES,
    QUERY_REWRITE_ENABLED,
    RERANK_BACKEND,
    RERANK_ENABLED,
    RERANK_TOP_K,
    RETRIEVER_CANDIDATE_K,
    RETRIEVER_TOP_K,
)
from backend.data.knowledge_base import build_documents
from backend.data.processing import split_documents
from backend.rag.chain import build_rag_graph
from backend.rag.rerank import Reranker, build_reranker
from backend.rag.retrievers import (
    SearchRetriever,
    VectorStoreClient,
    build_bm25_retriever,
    build_hybrid_retriever,
    build_vector_retriever,
    get_vector_document_count,
    load_vectorstore,
    rebuild_vectorstore,
)
from backend.rag.rewrite import build_query_rewrite_chain
from backend.types import RagDocument


_UNSET: Any = object()
EXECUTION_MODE_NAME = "langgraph"


class DemoRuntime:
    """把 demo 运行时需要的对象放到一起，方便 CLI 和 API 复用。"""

    def __init__(
        self,
        documents: list[RagDocument],
        split_documents_list: list[RagDocument],
        vectorstore: VectorStoreClient,
        vector_retriever: SearchRetriever,
        keyword_retriever: SearchRetriever,
        retriever: SearchRetriever,
        rewrite_chain: Any | None,
        rag_graph: Any,
        agent_graph: Any,
        vector_document_count: int,
        reranker: Reranker | None = None,
    ) -> None:
        self.documents = documents
        self.split_documents_list = split_documents_list
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.rewrite_chain = rewrite_chain

        # 两个核心可流式 LangGraph 应用。
        self.rag_graph = rag_graph
        self.agent_graph = agent_graph

        self.execution_mode = EXECUTION_MODE_NAME
        self.vector_document_count = vector_document_count
        self.reranker = reranker


def prepare_documents_for_rag() -> tuple[list[RagDocument], list[RagDocument]]:
    """准备原始文档和切分后的片段。"""
    documents = build_documents()
    split_documents_list = split_documents(documents)

    return documents, split_documents_list


def rebuild_demo_index() -> dict:
    """手动重建项目使用的向量索引。"""
    documents, split_documents_list = prepare_documents_for_rag()
    vectorstore = rebuild_vectorstore(split_documents_list)
    vector_document_count = get_vector_document_count(
        vectorstore,
        fallback_count=len(split_documents_list),
    )

    return {
        "raw_document_count": len(documents),
        "split_document_count": len(split_documents_list),
        "vector_document_count": vector_document_count,
    }


def build_demo_retriever(
    split_documents_list: list[RagDocument],
    vectorstore: VectorStoreClient,
) -> SearchRetriever:
    """构建在线服务默认使用的检索器。"""
    return build_hybrid_retriever(split_documents_list, vectorstore)


def build_demo_rewrite_chain() -> Any | None:
    """根据配置决定是否启用查询改写。"""
    if not QUERY_REWRITE_ENABLED:
        return None
    return build_query_rewrite_chain()


def build_demo_reranker() -> Reranker | None:
    """根据配置决定是否启用重排。"""
    if not RERANK_ENABLED:
        return None
    return build_reranker(RERANK_BACKEND)


def _resolve_retrieval_top_k() -> int:
    """根据是否启用重排，返回召回阶段实际使用的候选数。"""
    if RERANK_ENABLED:
        return max(RETRIEVER_CANDIDATE_K, RERANK_TOP_K)
    return RETRIEVER_TOP_K


def build_default_agent_tools(rag_graph: Any) -> list:
    """构造默认工具集：rag_answer + current_time。"""
    return [build_rag_answer_tool(rag_graph), current_time]


def create_demo_runtime(
    *,
    documents: list[RagDocument] | None = None,
    split_documents_list: list[RagDocument] | None = None,
    vectorstore: VectorStoreClient | None = None,
    vector_retriever: SearchRetriever | None = None,
    keyword_retriever: SearchRetriever | None = None,
    retriever: SearchRetriever | None = None,
    rewrite_chain: Any = _UNSET,
    reranker: Any = _UNSET,
    vector_document_count: int | None = None,
    rag_graph: Any | None = None,
    agent_graph: Any | None = None,
) -> DemoRuntime:
    """创建在线服务和 CLI 需要的运行时对象。"""
    if documents is None or split_documents_list is None:
        default_docs, default_splits = prepare_documents_for_rag()
        documents = documents if documents is not None else default_docs
        split_documents_list = (
            split_documents_list if split_documents_list is not None else default_splits
        )

    if vectorstore is None:
        vectorstore = load_vectorstore()

    if vector_document_count is None:
        vector_document_count = get_vector_document_count(
            vectorstore,
            fallback_count=len(split_documents_list),
        )

    if vector_document_count == 0:
        raise RuntimeError(
            "未找到可用的向量索引。请先运行 `python build_index.py` 构建索引。"
        )

    retrieval_top_k = _resolve_retrieval_top_k()

    if vector_retriever is None:
        vector_retriever = build_vector_retriever(vectorstore, top_k=retrieval_top_k)

    if keyword_retriever is None:
        keyword_retriever = build_bm25_retriever(
            split_documents_list, top_k=retrieval_top_k
        )

    if retriever is None:
        retriever = build_demo_retriever(split_documents_list, vectorstore)

    if rewrite_chain is _UNSET:
        rewrite_chain = build_demo_rewrite_chain()

    if reranker is _UNSET:
        reranker = build_demo_reranker()

    if rag_graph is None:
        rag_graph = build_rag_graph(
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            rewrite_chain=rewrite_chain,
            max_iterations=LANGGRAPH_MAX_ITERATIONS,
            min_sources=LANGGRAPH_MIN_SOURCES,
            top_k=retrieval_top_k,
            reranker=reranker,
            rerank_top_k=RERANK_TOP_K if reranker is not None else None,
        )

    if agent_graph is None:
        agent_graph = build_agent_graph(build_default_agent_tools(rag_graph))

    return DemoRuntime(
        documents,
        split_documents_list,
        vectorstore,
        vector_retriever,
        keyword_retriever,
        retriever,
        rewrite_chain,
        rag_graph=rag_graph,
        agent_graph=agent_graph,
        vector_document_count=vector_document_count,
        reranker=reranker,
    )
