"""LLM integration layer.

Centralizes all low-level Bedrock calls in this package:
- ``bedrock``: boto3 client, embedding, rerank, and other non-chat APIs.
- ``chat_models``: chat-model factory based on
  ``langchain-aws.ChatBedrockConverse``, used by the LangGraph nodes of
  RAG / Agent.

The goal is to let ``backend.rag`` and ``backend.agent`` depend only on
``backend.llm`` instead of on each other.
"""

from backend.llm.bedrock import (
    bedrock_rerank,
    embed_texts,
    reset_bedrock_client,
)
from backend.llm.chat_models import get_chat_model

__all__ = [
    "bedrock_rerank",
    "embed_texts",
    "get_chat_model",
    "reset_bedrock_client",
]
