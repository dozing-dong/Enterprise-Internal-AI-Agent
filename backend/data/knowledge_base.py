import json
from functools import lru_cache
from pathlib import Path

from backend.config import PROJECT_ROOT
from backend.types import RagDocument


LOCAL_EVAL_DIR = PROJECT_ROOT / "backend" / "data" / "data" / "local_eval"
LOCAL_EVAL_DOCUMENTS_PATH = LOCAL_EVAL_DIR / "documents.json"
LOCAL_EVAL_CASES_PATH = LOCAL_EVAL_DIR / "eval_cases.json"


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def read_json_file(file_path: Path) -> list[dict]:
    for encoding in ["utf-8", "utf-8-sig", "gbk"]:
        try:
            with file_path.open("r", encoding=encoding) as file:
                return json.load(file)
        except UnicodeDecodeError:
            continue

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)

#根据本地原始文档构建适合RAG的文档列表
@lru_cache(maxsize=1)
def build_documents() -> list[RagDocument]:
    rows = read_json_file(LOCAL_EVAL_DOCUMENTS_PATH)
    documents: list[RagDocument] = []

    for index, row in enumerate(rows, start=1):
        documents.append(
            RagDocument(
                page_content=normalize_text(row["content"]),
                metadata={
                    "source": "local_eval",
                    "context_id": row["context_id"],
                    "document_role": row.get("document_role", "reference_context"),
                    "title": row.get("title", ""),
                    "paragraph_index": index,
                },
            )
        )

    return documents

#根据本地原始评测样本构建适合RAG的评测样本列表
@lru_cache(maxsize=1)
def build_eval_cases() -> list[dict]:
    rows = read_json_file(LOCAL_EVAL_CASES_PATH)
    eval_cases: list[dict] = []

    for index, row in enumerate(rows, start=1):
        eval_cases.append(
            {
                "case_id": row.get("case_id", f"local_{index:03d}"),
                "category": row.get("category", "local_eval"),
                "difficulty": row.get("difficulty", "synthetic"),
                "question": row["question"],
                "reference": row["reference"],
                "reference_context_ids": row["reference_context_ids"],
                "note": row.get("note", "项目内置自构造评测样本。"),
            }
        )

    return eval_cases


def clear_document_caches() -> None:
    """显式清空文档加载相关的 lru_cache。

    设计约束：
    - 这些缓存仅是性能优化，不承担业务状态。
    - 测试或热重载时调用以避免跨用例污染。
    """
    build_documents.cache_clear()
    build_eval_cases.cache_clear()
