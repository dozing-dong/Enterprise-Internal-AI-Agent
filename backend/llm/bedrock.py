"""Low-level Bedrock call wrappers.

Responsible for two things only:
- Create and cache boto3 clients (one connection pool each for chat / agent runtime).
- Provide ``embed_texts`` / ``bedrock_rerank``: pure non-chat API calls.

Chat and tool-calling go exclusively through ``chat_models.get_chat_model``;
this module no longer wraps them.
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
    """Create a Bedrock Runtime client.

    Uses lru_cache to reuse the underlying boto3 client (the connection
    pool and signer are expensive to recreate).
    """
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("boto3 is required to use Bedrock.") from exc

    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
    )


@lru_cache(maxsize=1)
def _get_bedrock_agent_runtime_client():
    """Create a Bedrock Agent Runtime client.

    The Bedrock Rerank API uses bedrock-agent-runtime, which is a separate
    service from the bedrock-runtime used for embedding. Cached separately
    so it can be switched via BEDROCK_RERANK_REGION when rerank is not
    available in the primary region.
    """
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("boto3 is required to use Bedrock.") from exc

    return boto3.client(
        "bedrock-agent-runtime",
        region_name=BEDROCK_RERANK_REGION or AWS_REGION,
    )


def reset_bedrock_client() -> None:
    """Explicitly drop cached Bedrock clients so the next call rebuilds them."""
    _get_bedrock_client.cache_clear()
    _get_bedrock_agent_runtime_client.cache_clear()


def bedrock_rerank(
    query: str,
    documents: Sequence[str],
    *,
    model_id: str,
    top_k: int,
) -> list[tuple[int, float]]:
    """Call the Rerank API on Bedrock Agent Runtime.

    Returns [(original_index, relevance_score), ...] sorted descending by
    score, with length <= top_k. Any exception is propagated to the
    caller, which decides on a degradation strategy.
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
    """Call the Bedrock embedding model to generate vectors."""
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
            raise ValueError("Unexpected embedding response format.")
        embeddings.append([float(value) for value in vector])

    return embeddings
