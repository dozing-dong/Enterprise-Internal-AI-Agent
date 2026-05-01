import json
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from backend.config import PROJECT_ROOT


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


@lru_cache(maxsize=1)
def build_documents() -> list[Document]:
    rows = read_json_file(LOCAL_EVAL_DOCUMENTS_PATH)
    documents: list[Document] = []

    for index, row in enumerate(rows, start=1):
        documents.append(
            Document(
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
