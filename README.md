# Enterprise Internal Knowledge Base — Q&A System

A production-ready RAG + Multi-Agent Q&A platform for enterprise internal knowledge bases. Built with **LangGraph**, **LangChain**, **AWS Bedrock**, **pgvector**, and **FastAPI** on the backend, and **React + Vite** on the frontend.

---

## Key Features

| Capability | Details |
|---|---|
| **Three execution modes** | `rag` — fixed LangGraph pipeline; `agent` — ReAct loop with tool calling; `multi_agent` — Supervisor-orchestrated multi-agent |
| **Hybrid retrieval** | Reciprocal Rank Fusion (RRF) over dense (pgvector cosine) and sparse (BM25) results |
| **Query rewriting** | LLM-powered rewrite + bilingual expansion hints |
| **Reranking** | AWS Bedrock Rerank API (`amazon.rerank-v1:0`) |
| **Employee lookup** | Structured PostgreSQL employee directory, integrated into both RAG chain and Agent tool-calling |
| **Multi-Agent orchestration** | Supervisor → (PolicyAgent ‖ ExternalContextAgent) → WriterAgent; all streaming |
| **MCP external tools** | Weather (Open-Meteo, no key), Brave Search (key required), Business Calendar (project-local) |
| **Session management** | PostgreSQL-backed chat history + session metadata; sidebar for multi-turn conversations |
| **Streaming** | Server-Sent Events (SSE) across all three modes; token, sources, trace, done events |
| **Observability** | Unified call trace (latency, input/output summaries, per-agent attribution) |

---

## Architecture Overview

```
Frontend (React + Vite)
    │
    │  REST / SSE
    ▼
FastAPI  ─── ChatOrchestrator
    │               │
    │        ┌──────┴──────────────┐
    │        │                     │
    │   RAG Graph              Agent Graph
    │   (LangGraph)            (LangGraph)
    │        │                     │
    │        └──────┬──────────────┘
    │               │
    │     Multi-Agent Graph (LangGraph)
    │     ┌──────────────────────────┐
    │     │  Supervisor              │
    │     │  ├─ PolicyAgent          │
    │     │  │    └─ rag_answer tool │
    │     │  ├─ ExternalContextAgent │
    │     │  │    └─ MCP tools       │
    │     │  └─ WriterAgent          │
    │     └──────────────────────────┘
    │
    ├── pgvector  (vector store + chat history + employee directory)
    └── AWS Bedrock  (chat, embeddings, rerank)
```

---

## Repository Structure

```
.
├── backend/
│   ├── agent/          # ReAct agent (LangGraph) + tool definitions
│   ├── api/            # FastAPI app, routes, schemas, DI
│   ├── data/           # Document loading & chunking
│   ├── llm/            # Bedrock chat model + embedding factories
│   ├── mcp/            # MCP client loader (MultiServerMCPClient)
│   ├── mcp_servers/    # Project-local MCP servers (weather, business_calendar)
│   ├── multi_agent/    # Multi-Agent orchestration (Supervisor + 3 sub-agents)
│   ├── orchestrator/   # ChatOrchestrator: mode dispatch + SSE streaming
│   ├── rag/            # RAG graph (rewrite → retrieve → rerank → generate)
│   ├── storage/        # Chat history + session metadata (PostgreSQL / in-memory)
│   ├── config.py       # All runtime configuration (env-var driven)
│   ├── runtime.py      # DemoRuntime assembly (graphs, stores, MCP)
│   └── types.py        # Shared types (RagDocument)
├── frontend/           # React + Vite + Tailwind + shadcn/ui
├── tests/              # pytest unit & integration tests (fully mocked, no real cloud)
├── evals/              # Offline evaluation scripts (RAGAS, generation quality)
├── scripts/            # Utility scripts (eval case generation / verification)
├── build_index.py      # One-shot CLI: ingest documents and build the pgvector index
├── run_api.py          # FastAPI entry point
├── requirements.txt    # Python runtime dependencies
├── requirements-dev.txt# Dev / test dependencies
└── .env.example        # Template for environment variables (copy to .env)
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Tested on 3.11 |
| Node 18+ | Frontend dev / `npx` for Brave Search MCP |
| PostgreSQL 15+ with pgvector | `CREATE EXTENSION pgvector;` in your DB |
| AWS credentials | Bedrock access required (chat, embeddings, rerank) |
| Brave Search API key | Optional — only needed for web-search in multi-agent mode |

---

## Quick Start

### 1. Clone and set up Python environment

```bash
git clone <repo-url>
cd Enterprise-Internal-Knowledge-Base-Question-Answering-System

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt   # for tests
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your actual values
```

Key variables:

| Variable | Description |
|---|---|
| `AWS_REGION` | AWS region for Bedrock (e.g. `us-east-1`) |
| `BEDROCK_CHAT_MODEL_ID` | Claude model ID via Bedrock cross-region inference |
| `BEDROCK_EMBEDDING_MODEL_ID` | Titan embedding model ID |
| `BEDROCK_RERANK_MODEL_ID` | Bedrock rerank model ID |
| `BEDROCK_RERANK_REGION` | Region for the rerank API (may differ from chat region) |
| `PGVECTOR_CONNECTION` | PostgreSQL connection string (psycopg v3 format) |
| `BRAVE_API_KEY` | Brave Search API key (multi-agent web-search tool) |

### 3. Set up PostgreSQL

```sql
-- Connect to your Postgres instance, then:
CREATE DATABASE rag_demo;
\c rag_demo
CREATE EXTENSION IF NOT EXISTS vector;
```

Tables (`rag_embeddings`, `rag_collections`, `rag_chat_history`, `rag_employees`) are created automatically on first startup.

### 4. Build the knowledge-base index

Place your source documents in `docs/` (PDF, TXT, MD supported), then:

```bash
python build_index.py
```

### 5. Start the backend

```bash
python run_api.py
# API available at http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
# UI available at http://localhost:5173
```

---

## API Reference (key endpoints)

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check with document counts |
| `POST` | `/chat` | Blocking Q&A (RAG / Agent / Multi-Agent) |
| `POST` | `/chat/stream` | Streaming Q&A via SSE |
| `GET` | `/history/{session_id}` | Fetch session message history |
| `DELETE` | `/history/{session_id}` | Clear session history |
| `GET` | `/sessions` | List all sessions |
| `POST` | `/sessions` | Create a new session |
| `PATCH` | `/sessions/{session_id}` | Rename a session |
| `DELETE` | `/sessions/{session_id}` | Delete a session |

Full interactive docs: `http://localhost:8000/docs`

---

## Execution Modes

### `rag` (default)
Fixed LangGraph pipeline: **query rewrite → hybrid retrieve (vector + BM25) → RRF fusion → rerank → LLM generate → persist history**.

### `agent`
ReAct loop: the model decides which tools to call (`rag_answer`, `employee_lookup`, `current_time`) and iterates until it produces a final answer.

### `multi_agent`
Supervisor routes to specialized sub-agents in parallel:
- **PolicyAgent** — retrieves internal travel / expense / approval policies via the RAG sub-graph.
- **ExternalContextAgent** — calls MCP tools (weather, business calendar, Brave Search) for external context.
- **WriterAgent** — synthesizes all gathered context into the final user-facing answer.

---

## MCP External Tools

| Tool | Transport | Requires |
|---|---|---|
| Weather (Open-Meteo) | stdio — project-local Python | Nothing (free, no key) |
| Business Calendar | stdio — project-local Python | Nothing (`holidays` library) |
| Brave Search | stdio — `npx @modelcontextprotocol/server-brave-search` | `BRAVE_API_KEY` + Node.js |

Disable any tool by setting `MCP_<NAME>_ENABLED=false` in `.env`.

---

## Running Tests

```bash
pytest tests/ -v
```

All tests are fully mocked — no real PostgreSQL, Bedrock, or MCP server is needed.

---

## Evaluation

Offline evaluation scripts are in `evals/`:

```bash
# RAG retrieval quality (RAGAS)
python evals/ragas_retrieval_eval.py

# Generation quality
python evals/generation_eval.py

# End-to-end RAG eval
python evals/rag_eval.py

# Chunking strategy eval
python evals/chunk_eval.py
```

---

## Configuration Reference

All tuneable knobs live in `backend/config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `RETRIEVER_CANDIDATE_K` | `20` | Recall pool size before reranking |
| `RERANK_ENABLED` | `true` | Enable/disable Bedrock reranking |
| `RERANK_TOP_K` | `5` | Number of docs kept after reranking |
| `LANGGRAPH_MAX_ITERATIONS` | `2` | Max RAG graph iterations |
| `HISTORY_BACKEND` | `postgres` | `postgres` or `memory` |
| `EMPLOYEE_SEED_ON_STARTUP` | `true` | Seed demo employees at startup |
| `MULTI_AGENT_ENABLED` | `true` | Assemble multi-agent graph at startup |
| `MULTI_AGENT_RECURSION_LIMIT` | `12` | Max ReAct steps across sub-agents |

---

## Tech Stack

**Backend**: Python 3.11, FastAPI, LangGraph, LangChain, LangChain-AWS, AWS Bedrock (Claude, Titan Embeddings, Rerank), PostgreSQL + pgvector, psycopg v3, rank-bm25, MCP (Model Context Protocol)

**Frontend**: React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, Zustand, Radix UI

---

## License

MIT
