"""Generate eval_cases.json from the policy documents in documents.json by calling an LLM.

For each document, 3 questions are generated covering easy / medium / hard
difficulty. Both questions and reference answers are produced in English to
match the English source documents. ``reference_context_ids`` points to the
``context_id`` of the source document.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage

from backend.llm import get_chat_model

DOCUMENTS_PATH = PROJECT_ROOT / "backend/data/data/local_eval/documents.json"
OUTPUT_PATH = PROJECT_ROOT / "backend/data/data/local_eval/eval_cases.json"

CATEGORY_MAP = {
    "expense": "expense",
    "leave": "leave",
    "attendance": "attendance",
    "travel": "travel",
    "procurement": "procurement",
    "contract": "contract",
    "security_account": "security",
    "data_classification": "security",
    "access_control": "security",
    "release": "it_ops",
    "incident": "it_ops",
    "meeting": "admin",
    "overtime": "hr",
    "asset": "admin",
    "vendor": "procurement",
    "reception": "admin",
    "invoice": "finance",
    "training": "hr",
    "remote_work": "hr",
    "archive": "admin",
    "employee_directory": "hr",
    "employee_qa": "hr",
    "org_structure": "hr",
}


PROMPT_TEMPLATE = """\
Below is an internal company policy document (English original):

---
Title: {title}
Content:
{content}
---

Based on the document above, generate 3 question-answer cases in English,
covering different difficulty levels (easy / medium / hard).

Requirements:
1. Each question must have a clear answer derivable from the document; do not rely on outside information.
2. The reference answer should be concise and accurate, kept within 1-3 sentences.
3. Use the same terminology, article numbers, and amounts that appear in the source document.
4. Output STRICTLY as the following JSON array, with no extra text:

[
  {{
    "question": "...",
    "reference": "...",
    "difficulty": "easy"
  }},
  {{
    "question": "...",
    "reference": "...",
    "difficulty": "medium"
  }},
  {{
    "question": "...",
    "reference": "...",
    "difficulty": "hard"
  }}
]
"""


def get_category(context_id: str) -> str:
    for key, category in CATEGORY_MAP.items():
        if key in context_id:
            return category
    return "general"


def generate_cases_for_doc(model, doc: dict, doc_index: int) -> list[dict]:
    context_id = doc["context_id"]
    title = doc["title"]
    content = doc["content"]
    category = get_category(context_id)

    prompt = PROMPT_TEMPLATE.format(title=title, content=content)
    response = model.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )

    items = json.loads(raw)

    cases = []
    for i, item in enumerate(items, start=1):
        case_id = f"policy_{doc_index:03d}_{i:02d}"
        cases.append(
            {
                "case_id": case_id,
                "category": category,
                "difficulty": item["difficulty"],
                "question": item["question"],
                "reference": item["reference"],
                "reference_context_ids": [context_id],
                "note": f"Auto-generated: {title}",
            }
        )

    return cases


def main() -> None:
    docs = json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8"))
    model = get_chat_model(temperature=0.3)

    all_cases: list[dict] = []
    for doc_index, doc in enumerate(docs, start=1):
        print(f"[{doc_index}/{len(docs)}] generating cases: {doc['context_id']} - {doc['title']}")
        try:
            cases = generate_cases_for_doc(model, doc, doc_index)
            all_cases.extend(cases)
            print(f"  -> generated {len(cases)} cases")
        except Exception as exc:
            print(f"  ! failed: {exc}")

    OUTPUT_PATH.write_text(
        json.dumps(all_cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone. Wrote {len(all_cases)} cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
