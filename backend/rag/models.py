from __future__ import annotations

import json
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from backend.config import (
    AWS_REGION,
    BEDROCK_CHAT_MODEL_ID,
    BEDROCK_EMBEDDING_MODEL_ID,
)


@lru_cache(maxsize=1)
def _get_bedrock_client():
    """创建 Bedrock Runtime 客户端。"""
    try:
        import boto3
    except ImportError as exc:
        raise ImportError("缺少 boto3 依赖，无法使用 Bedrock。") from exc

    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
    )


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

    response = client.converse(**request)
    answer = _extract_text_from_converse_response(response)
    if not answer:
        return "没有生成可用回答。"
    return answer


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
