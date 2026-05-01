import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORY_DIR = PROJECT_ROOT / "chat_history"

EXECUTION_MODE = "langgraph"
MODEL_PROVIDER = "bedrock"
VECTOR_BACKEND = "pgvector"

RETRIEVER_TOP_K = 3
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4

CHUNK_SEPARATORS = ["。", "，", "\n", ""]
DEFAULT_CHUNK_PROFILE_NAME = "balanced_default"
CHUNK_PROFILES = {
    "small_dense": {
        "chunk_size": 80,
        "chunk_overlap": 20,
        "description": "Smaller chunks for tighter matching.",
    },
    "balanced_default": {
        "chunk_size": 120,
        "chunk_overlap": 30,
        "description": "Balanced chunking for the default demo flow.",
    },
    "large_context": {
        "chunk_size": 200,
        "chunk_overlap": 50,
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

LANGGRAPH_MAX_ITERATIONS = int(os.getenv("LANGGRAPH_MAX_ITERATIONS", "2"))
LANGGRAPH_MIN_SOURCES = int(os.getenv("LANGGRAPH_MIN_SOURCES", "1"))
