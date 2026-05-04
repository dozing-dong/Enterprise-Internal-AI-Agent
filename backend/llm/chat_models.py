"""ChatBedrockConverse factory.

LangGraph nodes obtain a LangChain ChatModel uniformly via ``get_chat_model()``:
- Synchronous ``.invoke(messages)`` for normal generation and tool decisions.
- Under ``graph.astream(stream_mode=["messages","updates"])``, calling
  ``.invoke`` inside a node is automatically captured as ``AIMessageChunk``
  events, so no streaming logic has to be hand-written.
- ``.bind_tools(tools)`` automatically serializes per the Bedrock Converse tool protocol.
"""

from __future__ import annotations

from langchain_aws import ChatBedrockConverse

from backend.config import AWS_REGION, BEDROCK_CHAT_MODEL_ID


def get_chat_model(*, temperature: float = 0.0) -> ChatBedrockConverse:
    """Return a ChatBedrockConverse instance.

    Each call constructs a new object so different nodes can use different
    temperatures; the underlying boto3 client is managed by langchain-aws.
    """
    return ChatBedrockConverse(
        model_id=BEDROCK_CHAT_MODEL_ID,
        region_name=AWS_REGION,
        temperature=temperature,
    )
