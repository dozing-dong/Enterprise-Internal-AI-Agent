"""会话标题生成。

通过 LLM 从首轮 user/assistant 内容生成一个 3-6 个英文单词的简短标题。
失败时回退到截断后的原始问题，保证总能拿到一个可读的 title。
"""

from __future__ import annotations

from backend.rag.models import chat_completion


TITLE_SYSTEM_PROMPT = (
    "You generate concise English titles for chat conversations. "
    "Produce a title of 3 to 6 words that summarizes the topic. "
    "Use Title Case, no surrounding quotes, no trailing punctuation, "
    "no emojis, no leading labels such as 'Title:'. "
    "Output only the title itself."
)

MAX_TITLE_LENGTH = 60
FALLBACK_LENGTH = 40


def _clean_title(raw: str) -> str:
    """剥掉常见装饰字符，保证 title 干净可显示。"""
    cleaned = raw.strip()

    if cleaned.lower().startswith("title:"):
        cleaned = cleaned[len("title:") :].strip()

    cleaned = cleaned.strip("\"'`“”‘’ ")
    cleaned = cleaned.splitlines()[0].strip() if cleaned else cleaned
    cleaned = cleaned.rstrip(".。！!？?")

    if len(cleaned) > MAX_TITLE_LENGTH:
        cleaned = cleaned[:MAX_TITLE_LENGTH].rstrip()

    return cleaned


def _fallback_title(question: str) -> str:
    snippet = question.strip().splitlines()[0] if question.strip() else "New Chat"
    if len(snippet) > FALLBACK_LENGTH:
        snippet = snippet[:FALLBACK_LENGTH].rstrip() + "…"
    return snippet or "New Chat"


def generate_session_title(question: str, answer: str) -> str:
    """根据首轮对话生成一个简短标题。

    出错时回退到原问题截断版本。
    """
    user_excerpt = question.strip()
    assistant_excerpt = answer.strip()
    if len(user_excerpt) > 400:
        user_excerpt = user_excerpt[:400].rstrip() + "…"
    if len(assistant_excerpt) > 400:
        assistant_excerpt = assistant_excerpt[:400].rstrip() + "…"

    messages = [
        {
            "role": "user",
            "content": (
                "Conversation to summarize:\n\n"
                f"User: {user_excerpt}\n\n"
                f"Assistant: {assistant_excerpt}\n\n"
                "Return a 3-6 word English title for this conversation."
            ),
        }
    ]

    try:
        raw = chat_completion(
            messages,
            system_prompt=TITLE_SYSTEM_PROMPT,
            temperature=0.2,
        )
    except Exception:
        return _fallback_title(question)

    cleaned = _clean_title(raw)
    if not cleaned:
        return _fallback_title(question)
    return cleaned
