"""LLM 接入层。

把所有底层 Bedrock 调用集中到本包：
- ``bedrock``：boto3 客户端、embedding、rerank 等不属于 chat 模型范畴的 API。
- ``chat_models``：基于 ``langchain-aws.ChatBedrockConverse`` 的 chat 模型工厂，
  供 RAG / Agent 各自的 LangGraph 节点使用。

目的是让 ``backend.rag`` 与 ``backend.agent`` 都仅依赖 ``backend.llm``，
而不互相耦合。
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
