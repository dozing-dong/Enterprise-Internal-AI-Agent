from __future__ import annotations

import re
from typing import Any

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from backend.config import (
    BM25_WEIGHT,
    PGVECTOR_COLLECTION_NAME,
    PGVECTOR_CONNECTION,
    PGVECTOR_DISTANCE_STRATEGY,
    PGVECTOR_PRE_DELETE_COLLECTION,
    RETRIEVER_TOP_K,
    VECTOR_WEIGHT,
)
from backend.rag.models import create_embedding_model


def get_vector_document_count(vectorstore: Any, fallback_count: int = 0) -> int:
    """统一读取不同向量库实现中的文档总数。"""
    if hasattr(vectorstore, "_collection"):
        return vectorstore._collection.count()

    return fallback_count


def _build_pgvector_store(
    split_documents_list: list[Document],
    *,
    pre_delete_collection: bool,
):
    try:
        from langchain_postgres import PGVector
    except ImportError as exc:
        raise ImportError("缺少 langchain-postgres 依赖，无法使用 pgvector。") from exc

    embedding_model = create_embedding_model()
    return PGVector.from_documents(
        documents=split_documents_list,
        embedding=embedding_model,
        connection=PGVECTOR_CONNECTION,
        collection_name=PGVECTOR_COLLECTION_NAME,
        distance_strategy=PGVECTOR_DISTANCE_STRATEGY,
        pre_delete_collection=pre_delete_collection,
    )


def _load_pgvector_store():
    try:
        from langchain_postgres import PGVector
    except ImportError as exc:
        raise ImportError("缺少 langchain-postgres 依赖，无法使用 pgvector。") from exc

    embedding_model = create_embedding_model()
    return PGVector(
        embeddings=embedding_model,
        connection=PGVECTOR_CONNECTION,
        collection_name=PGVECTOR_COLLECTION_NAME,
        distance_strategy=PGVECTOR_DISTANCE_STRATEGY,
    )


def rebuild_vectorstore(split_documents_list: list[Document]) -> Any:
    """重建并持久化向量库。"""
    return _build_pgvector_store(
        split_documents_list,
        pre_delete_collection=PGVECTOR_PRE_DELETE_COLLECTION,
    )


def load_vectorstore() -> Any:
    """加载已经持久化到磁盘的向量库。"""
    return _load_pgvector_store()


def build_evaluation_vectorstore(
    split_documents_list: list[Document],
    collection_name: str,
) -> Any:
    """为评测脚本构建一个不落盘的临时向量库。"""
    _ = collection_name
    return _build_pgvector_store(
        split_documents_list,
        pre_delete_collection=True,
    )


def tokenize_for_bm25(text: str) -> list[str]:
    """给 BM25 做一个简单、可读的中文友好分词。"""
    normalized = text.lower()
    chunks = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", normalized)

    tokens: list[str] = []

    for chunk in chunks:
        # 英文和数字直接保留。
        if re.fullmatch(r"[a-z0-9_]+", chunk):
            tokens.append(chunk)
            continue

        # 单字中文直接保留。
        if len(chunk) == 1:
            tokens.append(chunk)
            continue

        # 多字中文做最简单的 2-gram。
        tokens.extend(chunk[index:index + 2] for index in range(len(chunk) - 1))

    # 如果上面的规则没有切出任何 token，就退回原始字符列表。
    return tokens or list(normalized)


def build_vector_retriever(
    vectorstore: Any,
    top_k: int = RETRIEVER_TOP_K,
) -> BaseRetriever:
    """把向量库包装成一个统一接口的检索器。"""
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def build_bm25_retriever(
    split_documents_list: list[Document],
    top_k: int = RETRIEVER_TOP_K,
) -> BM25Retriever:
    """基于切分后的文档构建 BM25 检索器。"""
    bm25_retriever = BM25Retriever.from_documents(
        split_documents_list,
        preprocess_func=tokenize_for_bm25,
    )

    bm25_retriever.k = top_k

    return bm25_retriever


def build_hybrid_retriever(
    split_documents_list: list[Document],
    vectorstore: Any,
    top_k: int = RETRIEVER_TOP_K,
) -> EnsembleRetriever:
    """构建混合检索器：向量检索 + BM25。"""
    # 这里允许传入 top_k，是为了后面给 rerank 留出更多候选文档。
    # 如果还只拿 3 个候选，rerank 能做的事情会非常有限。
    vector_retriever = build_vector_retriever(vectorstore, top_k=top_k)
    bm25_retriever = build_bm25_retriever(split_documents_list, top_k=top_k)

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[VECTOR_WEIGHT, BM25_WEIGHT],
    )
