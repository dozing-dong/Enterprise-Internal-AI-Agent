from typing import Any

from langchain_core.embeddings import Embeddings

from backend.config import (
    AWS_REGION,
    BEDROCK_CHAT_MODEL_ID,
    BEDROCK_EMBEDDING_MODEL_ID,
)


def create_embedding_model() -> Embeddings:
    """Create the embedding backend used by the vector store."""
    try:
        from langchain_aws import BedrockEmbeddings
    except ImportError as exc:
        raise ImportError("缺少 langchain-aws 依赖，无法使用 Bedrock embedding。") from exc

    return BedrockEmbeddings(
        model_id=BEDROCK_EMBEDDING_MODEL_ID,
        region_name=AWS_REGION,
    )


def create_chat_model() -> Any:
    """Create the chat model."""
    try:
        from langchain_aws import ChatBedrockConverse
    except ImportError as exc:
        raise ImportError("缺少 langchain-aws 依赖，无法使用 Bedrock chat。") from exc

    return ChatBedrockConverse(
        model=BEDROCK_CHAT_MODEL_ID,
        region_name=AWS_REGION,
        temperature=0,
    )
