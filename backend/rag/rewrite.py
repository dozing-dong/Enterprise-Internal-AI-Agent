from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from backend.llm import get_chat_model


REWRITE_SYSTEM_PROMPT = (
    "You are a query rewriting assistant for retrieval. "
    "Your task is to rewrite the user's original question into a concise query "
    "that is better suited for knowledge base retrieval. "
    "Preserve the time, people, places, objects, dynasties, causal relations, "
    "and any key terms mentioned in the original question. "
    "Do not answer the question, do not explain your reasoning, "
    "and output only a single rewritten retrieval query."
)


def _build_user_text(question: str) -> str:
    return (
        f"Original question: {question}\n\n"
        "Please output a single rewritten query that is better suited for retrieval."
    )


def build_query_rewrite_chain() -> Callable[[str], str]:
    """构建一个最小查询改写函数。"""

    chat_model = get_chat_model(temperature=0.0)

    def rewrite_question(question: str) -> str:
        ai_msg = chat_model.invoke(
            [
                SystemMessage(REWRITE_SYSTEM_PROMPT),
                HumanMessage(_build_user_text(question)),
            ]
        )
        content = ai_msg.content
        if isinstance(content, list):
            text_parts = [
                str(block.get("text", "")) if isinstance(block, dict) else str(block)
                for block in content
            ]
            content = "".join(text_parts)
        return str(content or "").strip() or "没有生成可用回答。"

    return rewrite_question


def normalize_rewritten_question(rewritten_question: str, original_question: str) -> str:
    """清理模型输出，尽量只保留检索问题本身。"""
    cleaned_question = rewritten_question.strip()

    # 有些模型会带上“改写后问题：”或 "Rewritten question:" 这类前缀。
    # 这里用最简单、可读的方式去掉常见标签，英文与中文前缀同时兼容。
    prefixes = [
        "Rewritten question:",
        "Rewritten query:",
        "Retrieval query:",
        "Query:",
        "Question:",
        "改写后问题：",
        "改写后的问题：",
        "检索问题：",
        "改写问题：",
        "问题：",
    ]

    for prefix in prefixes:
        if cleaned_question.lower().startswith(prefix.lower()):
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
