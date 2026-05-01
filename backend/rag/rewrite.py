from collections.abc import Callable

from backend.rag.models import chat_completion


REWRITE_SYSTEM_PROMPT = (
    "你是一个检索问题改写助手。"
    "你的任务是把用户原问题改写成更适合知识库检索的短查询。"
    "请保留原问题里的时间、人物、地点、对象、朝代、因果关系和关键词。"
    "不要回答问题，不要解释原因，只输出一条改写后的检索问题。"
)


def build_rewrite_messages(question: str) -> list[dict[str, str]]:
    """构建改写阶段要发送的消息。"""
    return [
        {
            "role": "user",
            "content": (
                f"原问题：{question}\n\n"
                "请输出更适合检索的改写问题。"
            ),
        }
    ]


def build_query_rewrite_chain() -> Callable[[str], str]:
    """构建一个最小查询改写函数。"""

    def rewrite_question(question: str) -> str:
        messages = build_rewrite_messages(question)
        return chat_completion(messages, system_prompt=REWRITE_SYSTEM_PROMPT)

    return rewrite_question


def normalize_rewritten_question(rewritten_question: str, original_question: str) -> str:
    """清理模型输出，尽量只保留检索问题本身。"""
    cleaned_question = rewritten_question.strip()

    # 有些模型会带上“改写后问题：”这种前缀。
    # 这里用最简单、可读的方式去掉常见标签。
    prefixes = [
        "改写后问题：",
        "改写后的问题：",
        "检索问题：",
        "改写问题：",
        "问题：",
    ]

    for prefix in prefixes:
        if cleaned_question.startswith(prefix):
            cleaned_question = cleaned_question[len(prefix):].strip()

    # 如果模型没有认真按要求输出，导致清理后变成空字符串，
    # 就直接退回原问题，避免后面的检索链断掉。
    if not cleaned_question:
        return original_question

    return cleaned_question


def rewrite_question_for_retrieval(
    question: str,
    rewrite_chain: Callable[[str], str] | None,
) -> str:
    """把原问题改写成更适合检索的问题。"""
    # 如果当前没有启用查询改写，就直接返回原问题。
    if rewrite_chain is None:
        return question

    rewritten_question = rewrite_chain(question)
    return normalize_rewritten_question(rewritten_question, question)
