# API Baseline Checklist

This checklist captures the expected behavior before and after the pure LangGraph migration.

## `/chat`

- Request body keeps `question` and `session_id`.
- Response keeps these fields:
  - `answer`
  - `original_question`
  - `retrieval_question`
  - `session_id`
  - `history_file`
  - `sources`
- Error path still returns HTTP 500 with readable message.

## `/history/{session_id}`

- Returns `session_id`.
- Returns `messages` as JSON array.
- Works when history file is missing (returns empty list).

## `/health`

- Returns:
  - `status`
  - `vector_document_count`
  - `raw_document_count`
  - `execution_mode`

## Refactor Comment Baseline

- No unfinished refactor markers in `backend/**/*.py`:
  - `TODO`, `FIXME`, `HACK`, `XXX`, `WIP`, `REFACTOR`
  - `待重构`, `待迁移`, `后续优化`
- Existing `临时` wording in retriever docstring is an intentional evaluation design note, not migration debt.

## DI Acceptance

- Routes consume runtime only via `Depends(get_runtime)` (no module-level
  reach-around).
- `create_demo_runtime` accepts optional injection of every component while
  defaulting to the original wiring.
- `set_runtime_factory` / `reset_runtime` exist for swapping or resetting the
  composition root (used by tests).
- Performance caches in `backend/data/knowledge_base.py` and
  `backend/rag/models.py` provide explicit clear helpers
  (`clear_document_caches`, `reset_bedrock_client`).
- `tests/api/test_dependency_injection.py` covers `/`, `/health`, `/chat`
  with a fake runtime (no pgvector / Bedrock calls).
