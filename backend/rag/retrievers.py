from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from rank_bm25 import BM25Okapi

from backend.config import (
    BM25_WEIGHT,
    BM25_TOKENIZER_NGRAM,
    PGVECTOR_COLLECTION_NAME,
    PGVECTOR_COLLECTIONS_TABLE,
    PGVECTOR_CONNECTION,
    PGVECTOR_EMBEDDINGS_TABLE,
    PGVECTOR_PRE_DELETE_COLLECTION,
    RETRIEVER_TOP_K,
    RRF_K,
    VECTOR_WEIGHT,
)
from backend.llm import embed_texts
from backend.types import RagDocument


def _normalize_connection_string(connection: str) -> str:
    """把 SQLAlchemy 风格的连接串转为 psycopg 可用格式。"""
    return connection.replace("postgresql+psycopg://", "postgresql://", 1)


def _vector_to_sql(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"


def _ensure_extension(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def _ensure_tables(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {PGVECTOR_COLLECTIONS_TABLE} (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            );
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {PGVECTOR_EMBEDDINGS_TABLE} (
                id BIGSERIAL PRIMARY KEY,
                collection_id INTEGER NOT NULL REFERENCES {PGVECTOR_COLLECTIONS_TABLE}(id) ON DELETE CASCADE,
                document TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}',
                embedding VECTOR NOT NULL
            );
            """
        )
        cursor.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{PGVECTOR_EMBEDDINGS_TABLE}_collection_id
            ON {PGVECTOR_EMBEDDINGS_TABLE} (collection_id);
            """
        )


def _get_collection_id(connection, collection_name: str, *, create: bool) -> int | None:
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT id FROM {PGVECTOR_COLLECTIONS_TABLE} WHERE name = %s;",
            (collection_name,),
        )
        row = cursor.fetchone()
        if row:
            return int(row[0])

        if not create:
            return None

        cursor.execute(
            f"INSERT INTO {PGVECTOR_COLLECTIONS_TABLE}(name) VALUES (%s) RETURNING id;",
            (collection_name,),
        )
        return int(cursor.fetchone()[0])


def _delete_collection_documents(connection, collection_id: int) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            f"DELETE FROM {PGVECTOR_EMBEDDINGS_TABLE} WHERE collection_id = %s;",
            (collection_id,),
        )


def _connect():
    try:
        import psycopg
    except ImportError as exc:
        raise ImportError("缺少 psycopg 依赖，无法使用 pgvector。") from exc
    return psycopg.connect(_normalize_connection_string(PGVECTOR_CONNECTION))


@dataclass(slots=True)
class VectorStoreClient:
    collection_name: str

    def rebuild(self, documents: list[RagDocument], *, pre_delete_collection: bool) -> None:
        with _connect() as connection:
            _ensure_extension(connection)
            _ensure_tables(connection)
            collection_id = _get_collection_id(connection, self.collection_name, create=True)
            if collection_id is None:
                raise RuntimeError("无法创建 pgvector collection。")
            if pre_delete_collection:
                _delete_collection_documents(connection, collection_id)

            texts = [document.page_content for document in documents]
            vectors = embed_texts(texts)
            payloads = [
                (
                    collection_id,
                    document.page_content,
                    json.dumps(document.metadata, ensure_ascii=False),
                    _vector_to_sql(vector),
                )
                for document, vector in zip(documents, vectors, strict=True)
            ]

            with connection.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO {PGVECTOR_EMBEDDINGS_TABLE}(collection_id, document, metadata, embedding)
                    VALUES (%s, %s, %s::jsonb, %s::vector);
                    """,
                    payloads,
                )
            connection.commit()

    def count(self) -> int:
        with _connect() as connection:
            _ensure_extension(connection)
            _ensure_tables(connection)
            collection_id = _get_collection_id(connection, self.collection_name, create=False)
            if collection_id is None:
                return 0
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {PGVECTOR_EMBEDDINGS_TABLE} WHERE collection_id = %s;",
                    (collection_id,),
                )
                return int(cursor.fetchone()[0])

    def similarity_search(self, query: str, top_k: int) -> list[RagDocument]:
        with _connect() as connection:
            _ensure_extension(connection)
            _ensure_tables(connection)
            collection_id = _get_collection_id(connection, self.collection_name, create=False)
            if collection_id is None:
                return []
            query_vector = embed_texts([query])[0]
            vector_sql = _vector_to_sql(query_vector)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT document, metadata, (embedding <=> %s::vector) AS distance
                    FROM {PGVECTOR_EMBEDDINGS_TABLE}
                    WHERE collection_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (vector_sql, collection_id, vector_sql, top_k),
                )
                rows = cursor.fetchall()

        docs: list[RagDocument] = []
        for row in rows:
            metadata = row[1]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            if not isinstance(metadata, dict):
                metadata = {}
            metadata = dict(metadata)
            metadata["distance"] = float(row[2])
            docs.append(RagDocument(page_content=row[0], metadata=metadata))
        return docs


@dataclass(slots=True)
class SearchRetriever:
    """轻量检索器协议，保留 invoke 接口便于兼容原调用方。"""

    invoke_fn: Callable[[str], list[RagDocument]]

    def invoke(self, query: str) -> list[RagDocument]:
        return self.invoke_fn(query)


@dataclass(slots=True)
class Bm25Index:
    model: BM25Okapi
    documents: list[RagDocument]
    tokenized_docs: list[list[str]]


def get_vector_document_count(vectorstore: Any, fallback_count: int = 0) -> int:
    """统一读取不同向量库实现中的文档总数。"""
    if hasattr(vectorstore, "count"):
        return vectorstore.count()
    return int(fallback_count)


def _build_pgvector_store(
    split_documents_list: list[RagDocument],
    *,
    pre_delete_collection: bool,
    collection_name: str,
) -> VectorStoreClient:
    vectorstore = VectorStoreClient(collection_name=collection_name)
    vectorstore.rebuild(
        split_documents_list,
        pre_delete_collection=pre_delete_collection,
    )
    return vectorstore


def _load_pgvector_store(collection_name: str = PGVECTOR_COLLECTION_NAME) -> VectorStoreClient:
    return VectorStoreClient(collection_name=collection_name)


def rebuild_vectorstore(split_documents_list: list[RagDocument]) -> VectorStoreClient:
    """重建并持久化向量库。"""
    return _build_pgvector_store(
        split_documents_list,
        pre_delete_collection=PGVECTOR_PRE_DELETE_COLLECTION,
        collection_name=PGVECTOR_COLLECTION_NAME,
    )


def load_vectorstore() -> VectorStoreClient:
    """加载已经持久化到磁盘的向量库。"""
    return _load_pgvector_store()


def build_evaluation_vectorstore(
    split_documents_list: list[RagDocument],
    collection_name: str,
) -> VectorStoreClient:
    """为评测脚本构建一个不落盘的临时向量库。"""
    return _build_pgvector_store(
        split_documents_list,
        pre_delete_collection=True,
        collection_name=collection_name,
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
        if len(chunk) <= BM25_TOKENIZER_NGRAM - 1:
            tokens.append(chunk)
            continue

        # 多字中文做简单的 n-gram。
        tokens.extend(
            chunk[index:index + BM25_TOKENIZER_NGRAM]
            for index in range(len(chunk) - BM25_TOKENIZER_NGRAM + 1)
        )

    # 如果上面的规则没有切出任何 token，就退回原始字符列表。
    return tokens or list(normalized)


def vector_search(
    vectorstore: VectorStoreClient,
    query: str,
    *,
    top_k: int,
) -> list[RagDocument]:
    """执行向量检索。"""
    return vectorstore.similarity_search(query, top_k=top_k)


def build_vector_retriever(
    vectorstore: VectorStoreClient,
    top_k: int = RETRIEVER_TOP_K,
) -> SearchRetriever:
    """把向量库包装成一个统一接口的检索器。"""
    return SearchRetriever(invoke_fn=lambda query: vector_search(vectorstore, query, top_k=top_k))


def build_bm25_index(
    split_documents_list: list[RagDocument],
) -> Bm25Index:
    tokenized_docs = [tokenize_for_bm25(document.page_content) for document in split_documents_list]
    return Bm25Index(
        model=BM25Okapi(tokenized_docs),
        documents=split_documents_list,
        tokenized_docs=tokenized_docs,
    )


def keyword_search(
    bm25_index: Bm25Index,
    query: str,
    *,
    top_k: int,
) -> list[RagDocument]:
    query_tokens = tokenize_for_bm25(query)
    scores = bm25_index.model.get_scores(query_tokens)
    indexed_scores = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    docs: list[RagDocument] = []
    for index, score in indexed_scores:
        metadata = dict(bm25_index.documents[index].metadata)
        metadata["bm25_score"] = float(score)
        docs.append(
            RagDocument(
                page_content=bm25_index.documents[index].page_content,
                metadata=metadata,
            )
        )
    return docs


def build_bm25_retriever(
    split_documents_list: list[RagDocument],
    top_k: int = RETRIEVER_TOP_K,
) -> SearchRetriever:
    """基于切分后的文档构建 BM25 检索器。"""
    bm25_index = build_bm25_index(split_documents_list)
    return SearchRetriever(
        invoke_fn=lambda query: keyword_search(bm25_index, query, top_k=top_k),
    )


def _doc_key(document: RagDocument) -> tuple[str, str]:
    context_id = str(document.metadata.get("context_id", ""))
    return context_id, document.page_content


def fuse_retrieval_results(
    vector_docs: list[RagDocument],
    bm25_docs: list[RagDocument],
    *,
    top_k: int,
    vector_weight: float = VECTOR_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
) -> list[RagDocument]:
    """通过 RRF 融合两个检索列表。"""
    doc_map: dict[tuple[str, str], RagDocument] = {}
    scored: dict[tuple[str, str], float] = {}

    for rank, document in enumerate(vector_docs, start=1):
        key = _doc_key(document)
        doc_map[key] = document
        scored[key] = scored.get(key, 0.0) + vector_weight / (RRF_K + rank)

    for rank, document in enumerate(bm25_docs, start=1):
        key = _doc_key(document)
        if key not in doc_map:
            doc_map[key] = document
        scored[key] = scored.get(key, 0.0) + bm25_weight / (RRF_K + rank)

    ranked_keys = sorted(scored, key=scored.get, reverse=True)[:top_k]
    fused_docs: list[RagDocument] = []
    for key in ranked_keys:
        base_doc = doc_map[key]
        metadata = dict(base_doc.metadata)
        metadata["fusion_score"] = scored[key]
        fused_docs.append(RagDocument(page_content=base_doc.page_content, metadata=metadata))
    return fused_docs


def build_hybrid_retriever(
    split_documents_list: list[RagDocument],
    vectorstore: VectorStoreClient,
    top_k: int = RETRIEVER_TOP_K,
) -> SearchRetriever:
    """构建混合检索器：向量检索 + BM25。"""
    vector_retriever = build_vector_retriever(vectorstore, top_k=top_k)
    bm25_retriever = build_bm25_retriever(split_documents_list, top_k=top_k)

    def invoke(query: str) -> list[RagDocument]:
        vector_docs = vector_retriever.invoke(query)
        bm25_docs = bm25_retriever.invoke(query)
        return fuse_retrieval_results(
            vector_docs,
            bm25_docs,
            top_k=top_k,
        )

    return SearchRetriever(invoke_fn=invoke)
