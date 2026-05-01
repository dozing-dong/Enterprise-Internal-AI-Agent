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
