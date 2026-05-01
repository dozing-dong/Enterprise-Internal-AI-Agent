from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from functools import lru_cache
from typing import Any

from backend.config import (
    AWS_REGION,
    BEDROCK_CHAT_MODEL_ID,
    BEDROCK_EMBEDDING_MODEL_ID,
    BEDROCK_RERANK_REGION,
)


@lru_cache(maxsize=1)
def _get_bedrock_client():
    """创建 Bedrock Runtime 客户端。

    使用 lru_cache 复用底层 boto3 client（连接池、签名器开销较高）。
    该缓存仅为性能优化，可在测试或凭证刷新场景下显式清空。
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
    与 chat / embedding 用的 bedrock-runtime 不是同一个 service。
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


def _extract_text_from_converse_response(response: dict[str, Any]) -> str:
    content_blocks = (
        response.get("output", {})
        .get("message", {})
        .get("content", [])
    )
    text_blocks: list[str] = []
    for block in content_blocks:
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                text_blocks.append(text)
    return "\n".join(text_blocks).strip()


def chat_completion(
    messages: Sequence[dict[str, str]],
    *,
    system_prompt: str | None = None,
    temperature: float = 0.0,
) -> str:
    """调用 Bedrock Converse API 生成文本。"""
    client = _get_bedrock_client()
    request = _build_converse_request(
        messages,
        system_prompt=system_prompt,
        temperature=temperature,
    )
    response = client.converse(**request)
    answer = _extract_text_from_converse_response(response)
    if not answer:
        return "没有生成可用回答。"
    return answer


def _build_converse_request(
    messages: Sequence[dict[str, str]],
    *,
    system_prompt: str | None,
    temperature: float,
) -> dict[str, Any]:
    """组装 Bedrock Converse / ConverseStream 共用的请求体。"""
    conversation_messages: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        conversation_messages.append(
            {
                "role": role,
                "content": [{"text": content}],
            }
        )

    if not conversation_messages:
        raise ValueError("messages 不能为空。")

    request: dict[str, Any] = {
        "modelId": BEDROCK_CHAT_MODEL_ID,
        "messages": conversation_messages,
        "inferenceConfig": {"temperature": temperature},
    }

    if system_prompt:
        request["system"] = [{"text": system_prompt}]

    return request


def chat_completion_stream(
    messages: Sequence[dict[str, str]],
    *,
    system_prompt: str | None = None,
    temperature: float = 0.0,
) -> Iterator[str]:
    """以流式方式调用 Bedrock Converse Stream，逐 delta 产出文本。

    - 仅 yield text delta；其他事件（messageStart / contentBlockStart / metadata 等）忽略。
    - 调用方负责拼接完整答案。
    """
    client = _get_bedrock_client()
    request = _build_converse_request(
        messages,
        system_prompt=system_prompt,
        temperature=temperature,
    )

    response = client.converse_stream(**request)
    stream = response.get("stream")
    if stream is None:
        return

    for event in stream:
        delta_event = event.get("contentBlockDelta")
        if not delta_event:
            continue
        delta = delta_event.get("delta") or {}
        text = delta.get("text")
        if isinstance(text, str) and text:
            yield text


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
