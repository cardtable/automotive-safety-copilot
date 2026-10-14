"""Central settings. Every knob can be overridden with a COPILOT_* environment variable or .env entry."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
# Makes ANTHROPIC_API_KEY from .env visible to the Anthropic SDK (shell variables take precedence).
load_dotenv(ROOT / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COPILOT_", env_file=ROOT / ".env", extra="ignore")

    # Paths
    data_dir: Path = ROOT / "data"
    sources_file: Path = ROOT / "config" / "sources.yaml"

    # Models
    llm_model: str = "claude-opus-5-5"
    judge_model: str = "claude-opus-5-5"
    llm_effort: str = "medium"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"

    # Chunking
    chunk_max_chars: int = 2400
    chunk_min_chars: int = 200

    # Retrieval
    dense_k: int = 40
    sparse_k: int = 40
    rrf_k: int = 60
    rerank_candidates: int = 30
    top_k: int = 8
    # Abstention: if the best cross-encoder score (sigmoid, 0..1) is below this, nothing grounds the query.
    min_rerank_score: float = 0.20
    max_retrieval_attempts: int = 2

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def chunks_file(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def manifest_file(self) -> Path:
        return self.processed_dir / "manifest.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_sources() -> dict:
    return yaml.safe_load(get_settings().sources_file.read_text())
