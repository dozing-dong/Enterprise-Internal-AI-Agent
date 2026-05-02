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
    messages: Sequence[dict[str, Any]],
    *,
    system_prompt: str | None,
    temperature: float,
    tool_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """组装 Bedrock Converse / ConverseStream 共用的请求体。

    支持两种 ``content`` 形式：
    - ``str``：普通文本消息（兼容现有调用方）。
    - ``list[dict]``：富内容块（toolUse / toolResult / 多段 text 等），
      调用方需自行符合 Bedrock 块结构，函数原样透传。

    ``tool_config`` 不为 None 时，注入到顶层 ``toolConfig`` 字段，
    用来声明本次 Converse 调用可用的工具集。
    """
    conversation_messages: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, str):
            blocks: list[dict[str, Any]] = [{"text": content}]
        elif isinstance(content, list):
            # 富内容形式：调用方必须保证每个 block 是 Bedrock 接受的结构
            # （toolUse / toolResult / text / image 等）。这里不做深拷贝，
            # 调用方不应再修改这些 dict。
            blocks = [block for block in content if isinstance(block, dict)]
            if not blocks:
                continue
        else:
            continue

        conversation_messages.append({"role": role, "content": blocks})

    if not conversation_messages:
        raise ValueError("messages 不能为空。")

    request: dict[str, Any] = {
        "modelId": BEDROCK_CHAT_MODEL_ID,
        "messages": conversation_messages,
        "inferenceConfig": {"temperature": temperature},
    }

    if system_prompt:
        request["system"] = [{"text": system_prompt}]

    if tool_config:
        request["toolConfig"] = tool_config

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


def _parse_converse_tool_response(response: dict[str, Any]) -> dict[str, Any]:
    """从 Bedrock Converse 的非流式响应里抽出 text + toolUse。"""
    content_blocks = (
        response.get("output", {}).get("message", {}).get("content", []) or []
    )

    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if isinstance(block.get("text"), str):
            text_parts.append(block["text"])
            continue
        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict):
            tool_uses.append(
                {
                    "tool_use_id": tool_use.get("toolUseId", ""),
                    "name": tool_use.get("name", ""),
                    "input": tool_use.get("input") or {},
                }
            )

    return {
        "stop_reason": response.get("stopReason", ""),
        "text": "\n".join(text_parts).strip(),
        "tool_uses": tool_uses,
        "raw_content": content_blocks,
    }


def chat_completion_with_tools(
    messages: Sequence[dict[str, Any]],
    tool_config: dict[str, Any],
    *,
    system_prompt: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """带工具声明的 Converse 调用（非流式）。

    返回:
        {
            "stop_reason": str,            # "end_turn" / "tool_use" / "max_tokens" / ...
            "text": str,                   # 模型文本输出（可能为空，尤其在 tool_use 时）
            "tool_uses": [                 # 模型本轮请求调用的工具
                {"tool_use_id": str, "name": str, "input": dict}
            ],
            "raw_content": list[dict],     # 原始 content blocks，便于回灌到下一轮 messages
        }
    """
    client = _get_bedrock_client()
    request = _build_converse_request(
        messages,
        system_prompt=system_prompt,
        temperature=temperature,
        tool_config=tool_config,
    )
    response = client.converse(**request)
    return _parse_converse_tool_response(response)


def chat_completion_with_tools_stream(
    messages: Sequence[dict[str, Any]],
    tool_config: dict[str, Any],
    *,
    system_prompt: str | None = None,
    temperature: float = 0.0,
) -> Iterator[dict[str, Any]]:
    """带工具声明的 ConverseStream 调用。

    向调用方逐步 yield 语义化事件：

    - ``{"type": "text_delta", "text": "..."}``: 模型文本片段，可直接转 SSE token。
    - ``{"type": "tool_use", "tool_use_id": ..., "name": ..., "input": {...}}``:
      模型一次完整的工具调用决策（已合并并解析过 input JSON）。
    - ``{"type": "stop", "stop_reason": "..."}``: 单次 Converse 调用结束。

    设计要点：
    - Bedrock 把 toolUse 的 input 以 JSON 字符串增量下发；这里负责按
      ``contentBlockIndex`` 累积，到 ``contentBlockStop`` 时一次性解析。
    - text_delta 直接透传，调用方决定是否流给前端。
    """
    client = _get_bedrock_client()
    request = _build_converse_request(
        messages,
        system_prompt=system_prompt,
        temperature=temperature,
        tool_config=tool_config,
    )

    response = client.converse_stream(**request)
    stream = response.get("stream")
    if stream is None:
        return

    # contentBlockIndex -> {"type": "tool_use"|"text", "tool_use_id"?, "name"?, "input_buffer"?}
    blocks: dict[int, dict[str, Any]] = {}

    for event in stream:
        if "contentBlockStart" in event:
            start_event = event["contentBlockStart"]
            index = start_event.get("contentBlockIndex", -1)
            start = start_event.get("start") or {}
            tool_use_start = start.get("toolUse")
            if isinstance(tool_use_start, dict):
                blocks[index] = {
                    "type": "tool_use",
                    "tool_use_id": tool_use_start.get("toolUseId", ""),
                    "name": tool_use_start.get("name", ""),
                    "input_buffer": "",
                }
            continue

        if "contentBlockDelta" in event:
            delta_event = event["contentBlockDelta"]
            index = delta_event.get("contentBlockIndex", -1)
            delta = delta_event.get("delta") or {}

            text = delta.get("text")
            if isinstance(text, str) and text:
                yield {"type": "text_delta", "text": text}
                continue

            tool_use_delta = delta.get("toolUse")
            if isinstance(tool_use_delta, dict):
                input_fragment = tool_use_delta.get("input")
                if isinstance(input_fragment, str):
                    block = blocks.setdefault(
                        index,
                        {
                            "type": "tool_use",
                            "tool_use_id": "",
                            "name": "",
                            "input_buffer": "",
                        },
                    )
                    block["input_buffer"] = (
                        block.get("input_buffer", "") + input_fragment
                    )
            continue

        if "contentBlockStop" in event:
            stop_event = event["contentBlockStop"]
            index = stop_event.get("contentBlockIndex", -1)
            block = blocks.pop(index, None)
            if block and block.get("type") == "tool_use":
                raw_input = block.get("input_buffer", "")
                try:
                    parsed_input = json.loads(raw_input) if raw_input else {}
                except json.JSONDecodeError:
                    parsed_input = {"_raw": raw_input}
                yield {
                    "type": "tool_use",
                    "tool_use_id": block.get("tool_use_id", ""),
                    "name": block.get("name", ""),
                    "input": parsed_input,
                }
            continue

        if "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "")
            yield {"type": "stop", "stop_reason": stop_reason}
            continue

        # 其他事件（messageStart / metadata / 异常）此处忽略，
        # 业务层不需要这些细节；如未来要算 token 成本可在此扩展。


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
