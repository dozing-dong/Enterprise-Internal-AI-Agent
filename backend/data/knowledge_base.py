import json
from functools import lru_cache
from pathlib import Path

from backend.config import PROJECT_ROOT
from backend.types import RagDocument


LOCAL_EVAL_DIR = PROJECT_ROOT / "backend" / "data" / "data" / "local_eval"
LOCAL_EVAL_DOCUMENTS_PATH = LOCAL_EVAL_DIR / "documents.json"
LOCAL_EVAL_CASES_PATH = LOCAL_EVAL_DIR / "eval_cases.json"


def normalize_text(text: str) -> str:
    # Normalize whitespace within each line but preserve line breaks so that
    # the downstream structure-aware splitter can split on paragraph/article
    # boundaries rather than falling back to character-level slicing.
    lines = text.splitlines()
    normalized_lines = [" ".join(line.split()) for line in lines]
    return "\n".join(normalized_lines).strip()


def read_json_file(file_path: Path) -> list[dict]:
    for encoding in ["utf-8", "utf-8-sig", "gbk"]:
        try:
            with file_path.open("r", encoding=encoding) as file:
                return json.load(file)
        except UnicodeDecodeError:
            continue

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)

# Build a RAG-friendly list of documents from the local raw documents.
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

# Build a RAG-friendly list of eval cases from the local raw eval samples.
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
                "note": row.get("note", "Built-in self-constructed eval sample."),
            }
        )

    return eval_cases


def clear_document_caches() -> None:
    """Explicitly clear the lru_cache entries used by document loading.

    Design constraints:
    - These caches are pure performance optimizations and hold no business state.
    - Call this in tests or on hot reload to avoid cross-case pollution.
    """
    build_documents.cache_clear()
    build_eval_cases.cache_clear()
