"""项目运行时装配。

把所有依赖一次性构造好暴露给 CLI / API：
- ``rag_graph``：编译好的 RAG LangGraph，``.invoke()`` 与 ``.stream()`` 通用。
- ``agent_graph``：编译好的单 Agent ReAct LangGraph；其工具集中包含
  ``rag_answer``，由 ``rag_graph`` 注入；不再走任何 fallback。
- ``multi_agent_graph``：编译好的多 Agent 编排 LangGraph（Supervisor +
  Policy + External + Writer 四个层级 subgraph）；外部工具来自 MCP。
  装配失败（如 langchain-mcp-adapters 不可用、Node 未安装、env 缺失）时
  降级为 ``None``，``mode=multi_agent`` 的请求会返回 503，rag/agent 不受影响。
"""

from typing import Any

import logging

from backend.agent import (
    build_agent_graph,
    build_employee_lookup_tool,
    build_rag_answer_tool,
    current_time,
)
from backend.config import (
    EMPLOYEE_RAG_MANDATORY,
    EMPLOYEE_SEED_ON_STARTUP,
    LANGGRAPH_MAX_ITERATIONS,
    LANGGRAPH_MIN_SOURCES,
    MULTI_AGENT_ENABLED,
    QUERY_REWRITE_ENABLED,
    RERANK_BACKEND,
    RERANK_ENABLED,
    RERANK_TOP_K,
    RETRIEVER_CANDIDATE_K,
    RETRIEVER_TOP_K,
)
from backend.mcp import MCPLoadResult, load_external_mcp_tools
from backend.multi_agent import build_multi_agent_graph
from backend.data.knowledge_base import build_documents
from backend.data.processing import split_documents
from backend.rag.chain import build_rag_graph
from backend.rag.employee_retriever import EmployeeStore, seed_default_employees
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


logger = logging.getLogger(__name__)


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
        multi_agent_graph: Any | None = None,
        mcp_load_result: MCPLoadResult | None = None,
    ) -> None:
        self.documents = documents
        self.split_documents_list = split_documents_list
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.rewrite_chain = rewrite_chain

        # 三个核心可流式 LangGraph 应用。
        self.rag_graph = rag_graph
        self.agent_graph = agent_graph
        # multi_agent_graph 可能为 None（依赖未装 / MCP 启动失败）。
        self.multi_agent_graph = multi_agent_graph
        # 持有 MCP 客户端 / 加载结果的引用，避免被 GC 关闭底层 stdio 连接。
        self.mcp_load_result = mcp_load_result

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


def build_default_agent_tools(
    rag_graph: Any,
    *,
    employee_store: EmployeeStore | None = None,
) -> list:
    """构造默认工具集：rag_answer + current_time + employee_lookup。"""
    return [
        build_rag_answer_tool(rag_graph),
        build_employee_lookup_tool(employee_store),
        current_time,
    ]


def _maybe_seed_employee_directory(store: EmployeeStore) -> None:
    """启动期把 demo 员工写入 PG，便于直接体验。

    任何异常都吞掉，避免 PG 不可用时阻塞主服务启动；上层只把它当成
    “最尽力”的初始化。
    """
    if not EMPLOYEE_SEED_ON_STARTUP:
        return
    try:
        if store.count() > 0:
            return
        inserted = seed_default_employees(store)
        if inserted:
            logger.info("seeded %s demo employees", inserted)
    except Exception:
        logger.exception("employee directory seeding skipped due to error")


def _build_multi_agent_graph_safely(
    *,
    rag_graph: Any,
    employee_store: EmployeeStore | None,
) -> tuple[Any | None, MCPLoadResult | None]:
    """启动期"尽力"装配多 Agent 图。

    任何环节失败（MCP 工具加载失败、subgraph 编译失败）都返回 ``(None, None)``。
    rag / agent 模式不受影响。
    """
    if not MULTI_AGENT_ENABLED:
        logger.info("MULTI_AGENT_ENABLED=false，跳过多 Agent 图装配。")
        return None, None

    try:
        load_result = load_external_mcp_tools()
    except Exception:  # noqa: BLE001
        logger.exception("加载 MCP 工具时出错，多 Agent 图降级为无外部工具。")
        load_result = MCPLoadResult()

    try:
        graph = build_multi_agent_graph(
            rag_graph=rag_graph,
            mcp_tools=load_result.tools,
            employee_store=employee_store,
        )
    except Exception:  # noqa: BLE001
        logger.exception("多 Agent 图编译失败，已降级为不可用。")
        return None, load_result

    if load_result.failed_servers:
        logger.warning(
            "部分 MCP server 加载失败：%s",
            ", ".join(load_result.failed_servers),
        )
    return graph, load_result


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
    multi_agent_graph: Any = _UNSET,
    employee_store: EmployeeStore | None = None,
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

    if employee_store is None:
        employee_store = EmployeeStore()
        _maybe_seed_employee_directory(employee_store)

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
            employee_store=employee_store if EMPLOYEE_RAG_MANDATORY else None,
        )

    if agent_graph is None:
        agent_graph = build_agent_graph(
            build_default_agent_tools(rag_graph, employee_store=employee_store)
        )

    mcp_load_result: MCPLoadResult | None = None
    if multi_agent_graph is _UNSET:
        multi_agent_graph, mcp_load_result = _build_multi_agent_graph_safely(
            rag_graph=rag_graph,
            employee_store=employee_store,
        )

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
        multi_agent_graph=multi_agent_graph,
        mcp_load_result=mcp_load_result,
    )
