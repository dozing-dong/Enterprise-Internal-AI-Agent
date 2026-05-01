from backend.types import RagDocument

from backend.config import (
    CHUNK_PROFILES,
    DEFAULT_CHUNK_PROFILE_NAME,
)


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


def split_documents(
    documents: list[RagDocument],
    profile_name: str | None = None,
) -> list[RagDocument]:
    """把长文本切成适合检索的小片段。"""
    profile = get_chunk_profile(profile_name)
    split_docs: list[RagDocument] = []

    for document in documents:
        chunks = split_text_by_window(
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
    """把 LangChain Document 转成更适合 API 返回的 sources。"""
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
