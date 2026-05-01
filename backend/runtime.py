from typing import Any, Callable

from backend.config import (
    LANGGRAPH_MAX_ITERATIONS,
    LANGGRAPH_MIN_SOURCES,
    QUERY_REWRITE_ENABLED,
    RETRIEVER_TOP_K,
)
from backend.data.knowledge_base import build_documents
from backend.data.processing import split_documents
from backend.rag.chain import build_langgraph_executor
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
        chat_executor: Callable[[str, str], dict],
        execution_mode: str,
        vector_document_count: int,
    ) -> None:
        # 保存原始文档。
        self.documents = documents

        # 保存切分后的片段。
        self.split_documents_list = split_documents_list

        # 保存向量库对象。
        self.vectorstore = vectorstore

        # 保存检索器。
        self.retriever = retriever

        # 保存向量检索器。
        self.vector_retriever = vector_retriever

        # 保存关键词检索器。
        self.keyword_retriever = keyword_retriever

        # 保存查询改写链。
        self.rewrite_chain = rewrite_chain

        # 保存统一的聊天执行器。
        self.chat_executor = chat_executor

        # 保存当前执行模式。
        self.execution_mode = execution_mode

        # 保存向量库文档数量。
        self.vector_document_count = vector_document_count


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


def create_demo_runtime(execution_mode: str | None = None) -> DemoRuntime:
    """创建在线服务和 CLI 需要的运行时对象。"""
    selected_execution_mode = execution_mode or "langgraph"
    documents, split_documents_list = prepare_documents_for_rag()
    vectorstore = load_vectorstore()
    vector_document_count = get_vector_document_count(
        vectorstore,
        fallback_count=len(split_documents_list),
    )

    if vector_document_count == 0:
        raise RuntimeError(
            "未找到可用的向量索引。请先运行 `python build_index.py` 构建索引。"
        )

    vector_retriever = build_vector_retriever(vectorstore, top_k=RETRIEVER_TOP_K)
    keyword_retriever = build_bm25_retriever(split_documents_list, top_k=RETRIEVER_TOP_K)
    retriever = build_demo_retriever(split_documents_list, vectorstore)
    rewrite_chain = build_demo_rewrite_chain()
    chat_executor = _build_chat_executor(
        vector_retriever,
        keyword_retriever,
        rewrite_chain,
        execution_mode=selected_execution_mode,
    )

    return DemoRuntime(
        documents,
        split_documents_list,
        vectorstore,
        vector_retriever,
        keyword_retriever,
        retriever,
        rewrite_chain,
        chat_executor=chat_executor,
        execution_mode=selected_execution_mode,
        vector_document_count=vector_document_count,
    )


def _build_chat_executor(
    vector_retriever: SearchRetriever,
    keyword_retriever: SearchRetriever,
    rewrite_chain: Any | None,
    execution_mode: str,
) -> Callable[[str, str], dict]:
    if execution_mode != "langgraph":
        raise ValueError("当前版本仅支持 langgraph 执行模式。")

    langgraph_executor = build_langgraph_executor(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        rewrite_chain=rewrite_chain,
        max_iterations=LANGGRAPH_MAX_ITERATIONS,
        min_sources=LANGGRAPH_MIN_SOURCES,
        top_k=RETRIEVER_TOP_K,
    )
    return lambda question, session_id: langgraph_executor.invoke(
        {"question": question, "session_id": session_id}
    )
