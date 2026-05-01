import re

from backend.types import RagDocument

from backend.config import (
    CHUNK_PROFILES,
    DEFAULT_CHUNK_PROFILE_NAME,
)

#获取切分策略
def get_chunk_profile(profile_name: str | None = None) -> dict:
    """读取指定的 chunk 策略配置。"""
    # 如果调用方没有显式传入策略名，就使用项目默认策略。
    selected_profile_name = profile_name or DEFAULT_CHUNK_PROFILE_NAME

    if selected_profile_name not in CHUNK_PROFILES:
        raise ValueError(
            f"未知的 chunk 策略: {selected_profile_name}。"
            f"可选值有: {list(CHUNK_PROFILES.keys())}"
        )

    return CHUNK_PROFILES[selected_profile_name]

#执行按窗口切分文本
def split_text_by_window(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """按窗口切分文本。"""
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0。")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须 >= 0 且小于 chunk_size。")

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
    # 例如：第八章、第12条
    re.compile(r"^第[一二三四五六七八九十百千万0-9]+[章节条款则目]"),
    # 例如：（一）、(2)、1.2
    re.compile(r"^[（(]?[一二三四五六七八九十0-9]+[）).、]\s*"),
    # 例如：1. xxx / 1.2 xxx
    re.compile(r"^\d+(?:\.\d+)*[\s、.]"),
]


def _is_structure_heading(line: str) -> bool:
    return any(pattern.match(line) for pattern in STRUCTURE_HEADING_PATTERNS)


def split_text_by_structure_then_window(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """版本B切分：先结构分块，再对超长块窗口切分。"""
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
    """把长文本切成适合检索的小片段。"""
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
    """把检索结果拼成 prompt 可直接使用的上下文字符串。"""
    if not docs:
        return "未检索到可用知识片段。"

    context_parts = []

    # 这里把每个片段都带上简单编号。
    for index, doc in enumerate(docs, start=1):
        context_parts.append(f"[片段 {index}]\n{doc.page_content}")

    return "\n\n".join(context_parts)


def convert_docs_to_sources(docs: list[RagDocument]) -> list[dict]:
    """把项目内 RagDocument 转成更适合 API 返回的 sources。"""
    sources = []

    # 每个检索结果都保留排序、正文和 metadata。
    for index, doc in enumerate(docs, start=1):
        sources.append(
            {
                "rank": index,
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return sources
