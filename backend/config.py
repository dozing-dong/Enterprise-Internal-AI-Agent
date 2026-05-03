import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
DOTENV_LOCAL_PATH = PROJECT_ROOT / ".env.local"

# 按顺序加载本地配置文件，且不覆盖系统环境变量。
# 这样可以兼容 CI、容器和开发者本机手动注入的凭证。
load_dotenv(DOTENV_PATH, override=False)
load_dotenv(DOTENV_LOCAL_PATH, override=False)

MODEL_PROVIDER = "bedrock"
VECTOR_BACKEND = "pgvector"

RETRIEVER_TOP_K = 3
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4

# 召回阶段（向量、BM25、融合）保留的候选数量。
# 重排开启时通常放大到 20~50，让 reranker 有更大的候选池可选；
# 关闭重排时与 RETRIEVER_TOP_K 等价即可。
RETRIEVER_CANDIDATE_K = int(os.getenv("RETRIEVER_CANDIDATE_K", "20"))

# 重排相关配置。
# 默认启用：若运行环境不支持，可通过环境变量显式关闭。
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "bedrock")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
BEDROCK_RERANK_MODEL_ID = os.getenv(
    "BEDROCK_RERANK_MODEL_ID",
    "amazon.rerank-v1:0",
)
# Bedrock Rerank API 走的是 bedrock-agent-runtime；
# 该服务在 region 上的可用性与 bedrock-runtime 不一定一致，单独配置以便切换。
BEDROCK_RERANK_REGION = os.getenv("BEDROCK_RERANK_REGION", "")

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
PGVECTOR_COLLECTIONS_TABLE = os.getenv("PGVECTOR_COLLECTIONS_TABLE", "rag_collections")
PGVECTOR_EMBEDDINGS_TABLE = os.getenv("PGVECTOR_EMBEDDINGS_TABLE", "rag_embeddings")
RRF_K = int(os.getenv("RRF_K", "60"))
BM25_TOKENIZER_NGRAM = int(os.getenv("BM25_TOKENIZER_NGRAM", "2"))

LANGGRAPH_MAX_ITERATIONS = int(os.getenv("LANGGRAPH_MAX_ITERATIONS", "2"))
LANGGRAPH_MIN_SOURCES = int(os.getenv("LANGGRAPH_MIN_SOURCES", "1"))

# 会话历史存储后端：postgres / memory。
# 默认走 postgres，与向量库共用同一个连接配置；memory 仅用于测试隔离。
HISTORY_BACKEND = os.getenv("HISTORY_BACKEND", "postgres")
HISTORY_TABLE = os.getenv("HISTORY_TABLE", "rag_chat_history")
