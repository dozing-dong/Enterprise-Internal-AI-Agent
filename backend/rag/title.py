"""Session title generation.

Use the LLM to generate a 3-6 word English title from the first
user/assistant exchange. On failure, falls back to a truncated version
of the original question, ensuring there is always a readable title.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from backend.llm import get_chat_model


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
    """Strip common decorative characters to keep the title clean and presentable."""
    cleaned = raw.strip()

    if cleaned.lower().startswith("title:"):
        cleaned = cleaned[len("title:"):].strip()

    cleaned = cleaned.strip("\"'`\u201c\u201d\u2018\u2019 ")
    cleaned = cleaned.splitlines()[0].strip() if cleaned else cleaned
    cleaned = cleaned.rstrip(".\u3002\uff01!\uff1f?")

    if len(cleaned) > MAX_TITLE_LENGTH:
        cleaned = cleaned[:MAX_TITLE_LENGTH].rstrip()

    return cleaned


def _fallback_title(question: str) -> str:
    snippet = question.strip().splitlines()[0] if question.strip() else "New Chat"
    if len(snippet) > FALLBACK_LENGTH:
        snippet = snippet[:FALLBACK_LENGTH].rstrip() + "\u2026"
    return snippet or "New Chat"


def _coerce_to_text(content) -> str:
    if isinstance(content, list):
        parts = [
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        ]
        return "".join(parts)
    return str(content or "")


def generate_session_title(question: str, answer: str) -> str:
    """Generate a short title from the first conversation exchange.

    Falls back to a truncated original question on error.
    """
    user_excerpt = question.strip()
    assistant_excerpt = answer.strip()
    if len(user_excerpt) > 400:
        user_excerpt = user_excerpt[:400].rstrip() + "\u2026"
    if len(assistant_excerpt) > 400:
        assistant_excerpt = assistant_excerpt[:400].rstrip() + "\u2026"

    user_message = HumanMessage(
        "Conversation to summarize:\n\n"
        f"User: {user_excerpt}\n\n"
        f"Assistant: {assistant_excerpt}\n\n"
        "Return a 3-6 word English title for this conversation."
    )

    try:
        chat_model = get_chat_model(temperature=0.2)
        ai_msg = chat_model.invoke([SystemMessage(TITLE_SYSTEM_PROMPT), user_message])
        raw = _coerce_to_text(ai_msg.content)
    except Exception:
        return _fallback_title(question)

    cleaned = _clean_title(raw)
    if not cleaned:
        return _fallback_title(question)
    return cleaned
