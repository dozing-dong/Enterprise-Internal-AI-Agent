"""ChatBedrockConverse 工厂。

LangGraph 节点统一通过 ``get_chat_model()`` 拿到一个 LangChain ChatModel：
- 同步 ``.invoke(messages)`` 用于普通生成与工具决策。
- 在 ``graph.astream(stream_mode=["messages","updates"])`` 下，节点内部调用
  ``.invoke`` 即可被自动捕获为 ``AIMessageChunk``，无需手写流式逻辑。
- ``.bind_tools(tools)`` 自动按 Bedrock Converse 工具协议序列化。
"""

from __future__ import annotations

from langchain_aws import ChatBedrockConverse

from backend.config import AWS_REGION, BEDROCK_CHAT_MODEL_ID


def get_chat_model(*, temperature: float = 0.0) -> ChatBedrockConverse:
    """返回一个 ChatBedrockConverse 实例。

    每次调用都会构造一个新对象，便于不同节点设置不同 temperature；
    底层 boto3 client 由 langchain-aws 自行管理。
    """
    return ChatBedrockConverse(
        model_id=BEDROCK_CHAT_MODEL_ID,
        region_name=AWS_REGION,
        temperature=temperature,
    )
