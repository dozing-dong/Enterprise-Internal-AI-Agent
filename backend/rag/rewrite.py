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
    """Build a minimal query rewriting function."""

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
        return str(content or "").strip() or "Failed to generate a usable answer."

    return rewrite_question


def normalize_rewritten_question(rewritten_question: str, original_question: str) -> str:
    """Clean up the model's output, keeping only the retrieval query itself when possible."""
    cleaned_question = rewritten_question.strip()

    # Some models prepend labels like "Rewritten question:" or "Rewritten query:".
    # Strip the most common prefixes the simple, readable way; keep both
    # English and Chinese prefixes for compatibility with bilingual outputs.
    prefixes = [
        "Rewritten question:",
        "Rewritten query:",
        "Retrieval query:",
        "Query:",
        "Question:",
        # Chinese counterparts kept as Unicode escapes so the source stays ASCII.
        "\u6539\u5199\u540e\u95ee\u9898\uff1a",
        "\u6539\u5199\u540e\u7684\u95ee\u9898\uff1a",
        "\u68c0\u7d22\u95ee\u9898\uff1a",
        "\u6539\u5199\u95ee\u9898\uff1a",
        "\u95ee\u9898\uff1a",
    ]

    for prefix in prefixes:
        if cleaned_question.lower().startswith(prefix.lower()):
            cleaned_question = cleaned_question[len(prefix):].strip()

    # If the model did not follow the instructions and the cleaned text is
    # empty, fall back to the original question to keep the retrieval chain
    # from breaking.
    if not cleaned_question:
        return original_question

    return cleaned_question


def _expand_retrieval_hints(query: str) -> str:
    """Add retrieval hints for cross-language scenarios so English policy docs are easier to hit."""
    text = (query or "").strip()
    if not text:
        return text

    hints: list[str] = []

    def add_hint(value: str) -> None:
        if value not in hints:
            hints.append(value)

    # Chinese phrasings -> English policy keywords; mainly covers the
    # "employee business travel / reimbursement" scenarios. Tokens are
    # written as Unicode escapes to keep the source ASCII.
    if any(token in text for token in ("\u51fa\u5dee", "\u5dee\u65c5", "\u5916\u5730")):
        add_hint("business travel policy")
        add_hint("travel request")
        add_hint("hotel limit")
    if any(token in text for token in ("\u62a5\u9500", "\u8d39\u7528", "\u53d1\u7968", "reimbursement")):
        add_hint("expense reimbursement policy")
        add_hint("reimbursable scope")
        add_hint("approval rules")
    if any(token in text for token in ("\u6d41\u7a0b", "\u5ba1\u6279")):
        add_hint("approval workflow")
    if "policy" in text.lower() or "\u653f\u7b56" in text:
        add_hint("company policy")

    if not hints:
        return text
    return f"{text} | " + " | ".join(hints)


def rewrite_question_for_retrieval(
    question: str,
    rewrite_chain: Callable[[str], str] | None,
) -> str:
    """Rewrite the original question into a more retrieval-friendly form."""
    # If query rewriting is not enabled, just return the original question.
    if rewrite_chain is None:
        return _expand_retrieval_hints(question)

    rewritten_question = rewrite_chain(question)
    normalized = normalize_rewritten_question(rewritten_question, question)
    return _expand_retrieval_hints(normalized)
