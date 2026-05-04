import re

from backend.types import RagDocument

from backend.config import (
    CHUNK_PROFILES,
    DEFAULT_CHUNK_PROFILE_NAME,
)


def get_chunk_profile(profile_name: str | None = None) -> dict:
    """Read the specified chunk profile configuration."""
    # Use the project default profile when the caller doesn't pass one.
    selected_profile_name = profile_name or DEFAULT_CHUNK_PROFILE_NAME

    if selected_profile_name not in CHUNK_PROFILES:
        raise ValueError(
            f"Unknown chunk profile: {selected_profile_name}. "
            f"Available options: {list(CHUNK_PROFILES.keys())}"
        )

    return CHUNK_PROFILES[selected_profile_name]


def split_text_by_window(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text into overlapping windows."""
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and less than chunk_size.")

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


STRUCTURE_HEADING_PATTERNS = [
    # e.g. "Chapter 8" / "Article 12" in Chinese: "\u7b2c8\u7ae0" / "\u7b2c12\u6761"
    re.compile(r"^\u7b2c[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u4e070-9]+[\u7ae0\u8282\u6761\u6b3e\u5219\u76ee]"),
    # e.g. "(\u4e00)" / "(2)" / "1.2"
    re.compile(r"^[\uff08(]?[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u53410-9]+[\uff09).\u3001]\s*"),
    # e.g. "1. xxx" / "1.2 xxx"
    re.compile(r"^\d+(?:\.\d+)*[\s\u3001.]"),
]


def _is_structure_heading(line: str) -> bool:
    return any(pattern.match(line) for pattern in STRUCTURE_HEADING_PATTERNS)


def split_text_by_structure_then_window(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Variant B splitting: split by structure first, then window-split overlong blocks."""
    if not text:
        return []

    lines = text.splitlines()
    structure_blocks: list[str] = []
    current_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current_lines:
                structure_blocks.append("\n".join(current_lines).strip())
                current_lines = []
            continue

        if _is_structure_heading(line) and current_lines:
            structure_blocks.append("\n".join(current_lines).strip())
            current_lines = [line]
            continue

        current_lines.append(line)

    if current_lines:
        structure_blocks.append("\n".join(current_lines).strip())

    if not structure_blocks:
        structure_blocks = [text.strip()]

    chunks: list[str] = []
    for block in structure_blocks:
        if len(block) <= chunk_size:
            chunks.append(block)
            continue
        chunks.extend(
            split_text_by_window(
                block,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks


def split_documents(
    documents: list[RagDocument],
    profile_name: str | None = None,
) -> list[RagDocument]:
    """Split long text into smaller, retrieval-friendly chunks."""
    profile = get_chunk_profile(profile_name)
    split_docs: list[RagDocument] = []

    for document in documents:
        chunks = split_text_by_structure_then_window(
            document.page_content,
            chunk_size=profile["chunk_size"],
            chunk_overlap=profile["chunk_overlap"],
        )
        for chunk_index, chunk in enumerate(chunks, start=1):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = chunk_index
            split_docs.append(RagDocument(page_content=chunk, metadata=metadata))

    return split_docs


def format_docs(docs: list[RagDocument]) -> str:
    """Concatenate retrieval results into a context string usable directly in a prompt."""
    if not docs:
        return "No relevant knowledge snippets were retrieved."

    context_parts = []

    for index, doc in enumerate(docs, start=1):
        context_parts.append(f"[Snippet {index}]\n{doc.page_content}")

    return "\n\n".join(context_parts)


def convert_docs_to_sources(docs: list[RagDocument]) -> list[dict]:
    """Convert internal RagDocument objects into API-friendly source records."""
    sources = []

    for index, doc in enumerate(docs, start=1):
        sources.append(
            {
                "rank": index,
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return sources
