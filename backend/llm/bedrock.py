"""底层 Bedrock 调用封装。

只负责两件事：
- 创建并缓存 boto3 client（chat / agent runtime 各一个连接池）。
- 提供 ``embed_texts`` / ``bedrock_rerank`` 两个非 chat 类的纯 API 调用。

chat 与工具调用一律走 ``chat_models.get_chat_model``，本模块不再封装。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import lru_cache

from backend.config import (
    AWS_REGION,
    BEDROCK_EMBEDDING_MODEL_ID,
    BEDROCK_RERANK_REGION,
)


@lru_cache(maxsize=1)
def _get_bedrock_client():
    """创建 Bedrock Runtime 客户端。

    使用 lru_cache 复用底层 boto3 client（连接池、签名器开销较高）。
    """
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("缺少 boto3 依赖，无法使用 Bedrock。") from exc

    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
    )


@lru_cache(maxsize=1)
def _get_bedrock_agent_runtime_client():
    """创建 Bedrock Agent Runtime 客户端。

    Bedrock Rerank API 走的是 bedrock-agent-runtime，
    与 embedding 用的 bedrock-runtime 不是同一个 service。
    单独缓存以便在 rerank 不可用 region 时通过 BEDROCK_RERANK_REGION 切换。
    """
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("缺少 boto3 依赖，无法使用 Bedrock。") from exc

    return boto3.client(
        "bedrock-agent-runtime",
        region_name=BEDROCK_RERANK_REGION or AWS_REGION,
    )


def reset_bedrock_client() -> None:
    """显式释放 Bedrock 客户端缓存，下次调用时重新创建。"""
    _get_bedrock_client.cache_clear()
    _get_bedrock_agent_runtime_client.cache_clear()


def bedrock_rerank(
    query: str,
    documents: Sequence[str],
    *,
    model_id: str,
    top_k: int,
) -> list[tuple[int, float]]:
    """调用 Bedrock Agent Runtime 的 Rerank API。

    返回 [(原始索引, 相关度分数), ...]，按分数降序，长度 <= top_k。
    任何异常都向上抛出，由调用方决定降级策略。
    """
    if not documents:
        return []

    client = _get_bedrock_agent_runtime_client()

    region = BEDROCK_RERANK_REGION or AWS_REGION
    model_arn = f"arn:aws:bedrock:{region}::foundation-model/{model_id}"

    request = {
        "queries": [{"type": "TEXT", "textQuery": {"text": query}}],
        "sources": [
            {
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "TEXT",
                    "textDocument": {"text": text},
                },
            }
            for text in documents
        ],
        "rerankingConfiguration": {
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "numberOfResults": min(top_k, len(documents)),
                "modelConfiguration": {"modelArn": model_arn},
            },
        },
    }

    response = client.rerank(**request)
    results = response.get("results", []) or []

    parsed: list[tuple[int, float]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        score = item.get("relevanceScore")
        if not isinstance(index, int) or not isinstance(score, (int, float)):
            continue
        if index < 0 or index >= len(documents):
            continue
        parsed.append((index, float(score)))
    return parsed


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """调用 Bedrock embedding 模型生成向量。"""
    client = _get_bedrock_client()
    embeddings: list[list[float]] = []

    for text in texts:
        response = client.invoke_model(
            modelId=BEDROCK_EMBEDDING_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text}),
        )
        payload = json.loads(response["body"].read())
        vector = payload.get("embedding")
        if not isinstance(vector, list):
            raise ValueError("embedding 响应格式异常。")
        embeddings.append([float(value) for value in vector])

    return embeddings
