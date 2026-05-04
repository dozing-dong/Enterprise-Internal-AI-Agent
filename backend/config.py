import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
DOTENV_LOCAL_PATH = PROJECT_ROOT / ".env.local"

# Load local config files in order, without overriding system environment variables.
# This keeps CI, container, and developer-injected credentials all compatible.
load_dotenv(DOTENV_PATH, override=False)
load_dotenv(DOTENV_LOCAL_PATH, override=False)

MODEL_PROVIDER = "bedrock"
VECTOR_BACKEND = "pgvector"

RETRIEVER_TOP_K = 3
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4

# Number of candidates kept during the recall stage (vector, BM25, fusion).
# When reranking is on, typically expand to 20~50 to give the reranker a
# larger candidate pool; when reranking is off, keeping it equal to
# RETRIEVER_TOP_K is sufficient.
RETRIEVER_CANDIDATE_K = int(os.getenv("RETRIEVER_CANDIDATE_K", "20"))

# Reranking-related configuration.
# Enabled by default; if the runtime does not support it, disable explicitly
# via the environment variable.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "bedrock")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
BEDROCK_RERANK_MODEL_ID = os.getenv(
    "BEDROCK_RERANK_MODEL_ID",
    "amazon.rerank-v1:0",
)
# The Bedrock Rerank API uses bedrock-agent-runtime; its regional availability
# may differ from bedrock-runtime, so it has its own configuration for easy switching.
BEDROCK_RERANK_REGION = os.getenv("BEDROCK_RERANK_REGION", "")

CHUNK_SEPARATORS = ["\u3002", "\uff0c", "\n", ""]
DEFAULT_CHUNK_PROFILE_NAME = "balanced_default"
CHUNK_PROFILES = {
    "small_dense": {
        "chunk_size": 200,
        "chunk_overlap": 40,
        "description": "Smaller chunks for tighter matching.",
    },
    "balanced_default": {
        "chunk_size": 400,
        "chunk_overlap": 80,
        "description": "Balanced chunking for the default demo flow.",
    },
    "large_context": {
        "chunk_size": 600,
        "chunk_overlap": 120,
        "description": "Larger chunks that preserve more context.",
    },
}

QUERY_REWRITE_ENABLED = False

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_CHAT_MODEL_ID = os.getenv(
    "BEDROCK_CHAT_MODEL_ID",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
)
BEDROCK_EMBEDDING_MODEL_ID = os.getenv(
    "BEDROCK_EMBEDDING_MODEL_ID",
    "amazon.titan-embed-text-v2:0",
)

PGVECTOR_CONNECTION = os.getenv(
    "PGVECTOR_CONNECTION",
    "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/rag_demo",
)
PGVECTOR_COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION_NAME", "test01_rag_demo")
PGVECTOR_DISTANCE_STRATEGY = os.getenv("PGVECTOR_DISTANCE_STRATEGY", "cosine")
PGVECTOR_PRE_DELETE_COLLECTION = True
PGVECTOR_COLLECTIONS_TABLE = os.getenv("PGVECTOR_COLLECTIONS_TABLE", "rag_collections")
PGVECTOR_EMBEDDINGS_TABLE = os.getenv("PGVECTOR_EMBEDDINGS_TABLE", "rag_embeddings")
RRF_K = int(os.getenv("RRF_K", "60"))
BM25_TOKENIZER_NGRAM = int(os.getenv("BM25_TOKENIZER_NGRAM", "2"))

LANGGRAPH_MAX_ITERATIONS = int(os.getenv("LANGGRAPH_MAX_ITERATIONS", "2"))
LANGGRAPH_MIN_SOURCES = int(os.getenv("LANGGRAPH_MIN_SOURCES", "1"))

# Chat history storage backend: postgres / memory.
# Defaults to postgres, sharing the same connection config as the vector store;
# `memory` is intended for test isolation only.
HISTORY_BACKEND = os.getenv("HISTORY_BACKEND", "postgres")
HISTORY_TABLE = os.getenv("HISTORY_TABLE", "rag_chat_history")

# Employee structured retrieval: shares the same PG connection as the vector store.
# - ``EMPLOYEE_TABLE``: employee directory table name (CREATE IF NOT EXISTS at startup).
# - ``EMPLOYEE_LOOKUP_TOP_K``: default upper bound for results in a single fuzzy lookup.
# - ``EMPLOYEE_RAG_MANDATORY``: whether the RAG chain always runs the employee lookup node.
# - ``EMPLOYEE_SEED_ON_STARTUP``: whether to insert demo employees at service startup.
EMPLOYEE_TABLE = os.getenv("EMPLOYEE_TABLE", "rag_employees")
EMPLOYEE_LOOKUP_TOP_K = int(os.getenv("EMPLOYEE_LOOKUP_TOP_K", "5"))
EMPLOYEE_RAG_MANDATORY = os.getenv("EMPLOYEE_RAG_MANDATORY", "true").lower() == "true"
EMPLOYEE_SEED_ON_STARTUP = os.getenv("EMPLOYEE_SEED_ON_STARTUP", "true").lower() == "true"


# ---------------------------------------------------------------------------
# Multi-Agent + MCP configuration
#
# - ``MULTI_AGENT_ENABLED``: whether to assemble the multi-agent graph at
#   runtime startup. If any critical dependency (e.g. langchain-mcp-adapters)
#   is missing or a server fails to start, this falls back to None and
#   ``mode=multi_agent`` requests return 503; rag/agent are unaffected.
# - ``MCP_*_COMMAND`` / ``MCP_*_ARGS``: stdio launch command for each MCP server.
#   Following the community-first strategy, weather and web_search default
#   to open-source npm MCP servers (require Node installed locally);
#   business_calendar uses our own Python MCP server.
# - ``MCP_*_ENABLED``: independently controls whether a server joins the
#   tool set; failure of one server never blocks the others.
# ---------------------------------------------------------------------------

MULTI_AGENT_ENABLED = os.getenv("MULTI_AGENT_ENABLED", "true").lower() == "true"

# Maximum number of ReAct steps shared by supervisor / policy / external / writer sub-agents.
MULTI_AGENT_RECURSION_LIMIT = int(os.getenv("MULTI_AGENT_RECURSION_LIMIT", "12"))

# Weather MCP server (project-local, based on the free Open-Meteo API; global
# coverage with no API key required).
# To switch back to the community npm server, override in .env:
#   MCP_WEATHER_COMMAND=npx
#   MCP_WEATHER_ARGS=-y @h1deya/mcp-server-weather
MCP_WEATHER_ENABLED = os.getenv("MCP_WEATHER_ENABLED", "true").lower() == "true"
MCP_WEATHER_COMMAND = os.getenv("MCP_WEATHER_COMMAND", "python")
MCP_WEATHER_ARGS = os.getenv(
    "MCP_WEATHER_ARGS",
    "-m backend.mcp_servers.weather_openmeteo",
)

# Brave Search MCP server. Requires BRAVE_API_KEY to call successfully.
MCP_BRAVE_SEARCH_ENABLED = (
    os.getenv("MCP_BRAVE_SEARCH_ENABLED", "true").lower() == "true"
)
MCP_BRAVE_SEARCH_COMMAND = os.getenv("MCP_BRAVE_SEARCH_COMMAND", "npx")
MCP_BRAVE_SEARCH_ARGS = os.getenv(
    "MCP_BRAVE_SEARCH_ARGS",
    "-y @modelcontextprotocol/server-brave-search",
)
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

# Business Calendar MCP server (project-local, based on the `holidays` package).
MCP_BUSINESS_CALENDAR_ENABLED = (
    os.getenv("MCP_BUSINESS_CALENDAR_ENABLED", "true").lower() == "true"
)
MCP_BUSINESS_CALENDAR_COMMAND = os.getenv("MCP_BUSINESS_CALENDAR_COMMAND", "python")
MCP_BUSINESS_CALENDAR_ARGS = os.getenv(
    "MCP_BUSINESS_CALENDAR_ARGS",
    "-m backend.mcp_servers.business_calendar",
)
# Default region for the business calendar (aligned with the example query
# "next week in Auckland").
BUSINESS_CALENDAR_DEFAULT_COUNTRY = os.getenv(
    "BUSINESS_CALENDAR_DEFAULT_COUNTRY", "NZ"
)
