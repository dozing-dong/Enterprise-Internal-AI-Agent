# FastAPI Development Entry

This project now uses FastAPI as the standard backend service entry.

## Start service

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Build vector index once (or whenever source docs change):

```bash
python build_index.py
```

3. Start FastAPI app:

```bash
python run_api.py
```

You can override runtime mode with environment variable:

```bash
set EXECUTION_MODE=langgraph
```

4. Open docs:

- OpenAPI UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## Supported runtime entries

- `python run_api.py`: standard local development and integration entry.
- `python backend/cli.py`: local debugging helper only; not the primary service entry.

## API modules in source control

FastAPI modules under `backend/api/` are part of the main backend codebase and should be tracked with normal pull request workflow:

- `backend/api/app.py`
- `backend/api/dependencies.py`
- `backend/api/exceptions.py`
- `backend/api/schemas.py`
- `backend/api/routes/chat.py`
- `backend/api/routes/history.py`

## Dependency injection model

Routes only depend on `Depends(get_runtime)`; concrete construction lives in the
composition root at [backend/api/dependencies.py](../backend/api/dependencies.py):

- `init_runtime()` builds the singleton via the configured factory.
- `set_runtime_factory(factory)` replaces the factory before lifespan runs.
- `reset_runtime()` clears the singleton (mainly for tests).
- `create_demo_runtime(...)` accepts optional injection of every component, so a
  partially-mocked runtime can be assembled without re-doing the full pipeline.

For tests, prefer `set_runtime_factory(lambda: fake_runtime)` plus FastAPI's
`TestClient` so lifespan picks up the fake object without touching pgvector or
Bedrock. See [tests/api/test_dependency_injection.py](../tests/api/test_dependency_injection.py)
for a working example.

Install dev dependencies and run tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## Cache control points

Performance caches are explicit and resettable, not hidden state:

- `backend/data/knowledge_base.py` exposes `clear_document_caches()`.
- `backend/rag/models.py` exposes `reset_bedrock_client()`.

Use them in tests or when reloading underlying data/credentials.
