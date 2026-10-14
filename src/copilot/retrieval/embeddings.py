from __future__ import annotations

import os
from functools import lru_cache

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import CrossEncoder, SentenceTransformer

from copilot.config import get_settings

# BGE models are trained with an instruction prefix on the query side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache
def embedder() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embed_model, device=_device())


@lru_cache
def reranker() -> CrossEncoder:
    return CrossEncoder(get_settings().rerank_model, device=_device(), max_length=512)


def embed_passages(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    vecs = embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    prefix = BGE_QUERY_PREFIX if "bge" in get_settings().embed_model.lower() else ""
    return embedder().encode(prefix + text, normalize_embeddings=True).tolist()
