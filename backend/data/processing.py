from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import (
    CHUNK_PROFILES,
    CHUNK_SEPARATORS,
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


def build_text_splitter(profile_name: str | None = None) -> RecursiveCharacterTextSplitter:
    """根据 chunk 策略配置创建文本切分器。"""
    profile = get_chunk_profile(profile_name)

    # 这里仍然使用最容易理解的字符级切分器。
    # 当前任务的重点不是换一种更复杂的切分器，
    # 而是先观察 chunk_size 和 chunk_overlap 变化本身会带来什么影响。
    return RecursiveCharacterTextSplitter(
        chunk_size=profile["chunk_size"],
        chunk_overlap=profile["chunk_overlap"],
        length_function=len,
        separators=CHUNK_SEPARATORS,
    )


def split_documents(
    documents: list[Document],
    profile_name: str | None = None,
) -> list[Document]:
    """把长文本切成适合检索的小片段。"""
    text_splitter = build_text_splitter(profile_name)
    return text_splitter.split_documents(documents)


def format_docs(docs: list[Document]) -> str:
    """把检索结果拼成 prompt 可直接使用的上下文字符串。"""
    if not docs:
        return "未检索到可用知识片段。"

    context_parts = []

    # 这里把每个片段都带上简单编号。
    for index, doc in enumerate(docs, start=1):
        context_parts.append(f"[片段 {index}]\n{doc.page_content}")

    return "\n\n".join(context_parts)


def convert_docs_to_sources(docs: list[Document]) -> list[dict]:
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
