"""Central configuration for all tunable parameters and API credentials.

Override any threshold at runtime by setting the matching environment variable
before launching, e.g.:
    RERANKER_THRESHOLD=0.2 python main.py
"""
import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Gemini embedding model (3072-d default, supports 768/1536/3072 via MRL)
GEMINI_EMBED_MODEL: str = "gemini-embedding-001"
GEMINI_EMBED_DIMENSION: int = 3072

# Gemini embedding input token limit
GEMINI_EMBED_TOKEN_LIMIT: int = 2048

# Cohere Rerank model
COHERE_RERANK_MODEL: str = "rerank-v4.0-pro"

# Gemini model for query expansion
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
GEMINI_TEMPERATURE: float = 0.4
GEMINI_MAX_TOKENS: int = 256

# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "events-index")
PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")

# ---------------------------------------------------------------------------
# Retrieval (hybrid search)
# ---------------------------------------------------------------------------
# Candidates fetched from each retrieval leg (BM25 and dense) before fusion
RETRIEVAL_TOP_N: int = 100

# RRF smoothing constant — 60 is the standard default
RRF_K: int = 60

# RRF leg weighting — alpha controls dense vs BM25 contribution
# rrf_score = alpha * 1/(k + dense_rank) + (1 - alpha) * 1/(k + bm25_rank)
# alpha=0.7 favors dense/semantic (70%) over BM25/keyword (30%)
RRF_ALPHA: float = 0.7

# Minimum fused RRF score to pass into reranking.
# RRF scores are small (max ~0.033 with 2 legs), so 0.0 passes everything.
# Raise this to pre-filter extremely weak matches before reranking.
RRF_THRESHOLD: float = float(os.getenv("RRF_THRESHOLD", "0.0"))

# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
# Cross-encoder score threshold (min-max normalised to [0, 1])
RERANKER_THRESHOLD: float = float(os.getenv("RERANKER_THRESHOLD", "0.0"))

# Maximum results returned to the user after reranking
RERANKER_TOP_K: int = 100

# ---------------------------------------------------------------------------
# API keys (loaded from .env)
# ---------------------------------------------------------------------------
COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
