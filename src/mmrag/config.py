"""Central configuration for the multimodal RAG pipeline.

Every setting (API keys, model names, index names, paths, retrieval knobs)
lives here, loaded from environment variables / the .env file. No other
module should read os.environ directly; they call get_settings() instead.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/mmrag/config.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API keys (SecretStr hides the value when printed or logged) ---
    openrouter_api_key: SecretStr
    pinecone_api_key: SecretStr

    # --- LLMs via OpenRouter (OpenAI-compatible API) ---
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    text_model: str  # used for text/table answers
    vision_model: str  # must accept image input; may be the same ID as text_model
    llm_temperature: float = 0.0  # deterministic answers for a factual RAG system
    llm_reasoning_effort: str = "low"  # "low" | "medium" | "high"; higher is slower
    llm_timeout: int = 120  # seconds per request before it is retried
    llm_hedge_after: float = 3.0  # seconds before a slow call gets a backup request

    # --- Embeddings (run locally, no API calls) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384  # must match the embedding model's output size

    # --- Pinecone vector store ---
    pinecone_index_name: str = "multimodal-rag"
    pinecone_namespace: str = "documents"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_metric: str = "cosine"

    # --- Retrieval ---
    top_k: int = 3  # chunks passed to the LLM
    retrieval_candidates: int = 20  # chunks fetched per search before fusion/reranking
    hybrid_search: bool = True  # combine dense (Pinecone) and keyword (BM25) results
    rerank: bool = True  # reorder candidates with a cross-encoder
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Query rewriting (one LLM call before retrieval) ---
    query_rewrite: bool = True  # resolve follow-ups and split multi-part questions
    max_search_queries: int = 3  # upper limit on queries per question
    history_turns: int = 3  # previous question/answer pairs shown to the rewriter
    rewrite_timeout: int = 10  # seconds; on timeout, search with the original question

    # --- Paths ---
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    extracted_assets_dir: Path = PROJECT_ROOT / "data" / "extracted_assets"
    ground_truth_dir: Path = PROJECT_ROOT / "data" / "ground_truth"
    index_dir: Path = PROJECT_ROOT / "data" / "indexes"  # local BM25 chunk store


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse the same object everywhere."""
    return Settings()
