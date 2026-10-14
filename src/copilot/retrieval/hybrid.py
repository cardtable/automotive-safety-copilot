"""Hybrid retrieval: dense (ChromaDB, BGE) + sparse (BM25) fused with Reciprocal Rank Fusion,
then re-scored by a cross-encoder.

Regulatory questions mix semantics ("how early must the warning come") with exact tokens that dense
models blur ("5.2.1.1", "S4.1.5", "R152", "HIC15"), so BM25 tokens keep clause numbers intact.
Optional metadata filters narrow the search to a family or document when the question names one.
"""

from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi

from copilot.config import get_settings
from copilot.ingest.build_index import COLLECTION
from copilot.retrieval.embeddings import embed_query, reranker

TOKEN = re.compile(r"[a-z]+\d*[a-z]*|\d+(?:\.\d+)*|s\d+(?:\.\d+)*", re.IGNORECASE)
STOP = set(["a", "an", "the", "of", "to", "in", "for", "on", "and", "or", "is", "are", "be", "shall", "by", "with", "as", "at", "from", "that", "this", "which", "any", "each", "its", "it", "than", "not", "may", "if", "into", "such", "other", "these", "those", "under", "within", "per", "what", "how", "when", "does", "do", "must"])


def tokenize(text: str) -> list[str]:
    return [t for t in (m.lower().rstrip(".") for m in TOKEN.findall(text)) if t not in STOP]


@dataclass
class Hit:
    id: str
    text: str
    meta: dict
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf: float = 0.0
    rerank: float | None = None       # sigmoid(cross-encoder logit), 0..1
    scores: dict = field(default_factory=dict)

    @property
    def clauses(self) -> list[str]:
        return self.meta["clauses"].split("|")

    @property
    def label(self) -> str:
        return self.meta["label"]


class Corpus:
    """Chunk store shared by the dense and sparse retrievers."""

    def __init__(self) -> None:
        s = get_settings()
        self.rows = [json.loads(line) for line in s.chunks_file.open()]
        self.by_id = {r["id"]: r for r in self.rows}
        self.bm25 = BM25Okapi([tokenize(r["embed_text"]) for r in self.rows])
        client = chromadb.PersistentClient(path=str(s.chroma_dir))
        self.col = client.get_collection(COLLECTION)

    def _meta(self, row: dict) -> dict:
        return {k: v for k, v in row.items() if k not in ("text", "embed_text", "id")}

    def _allowed(self, row: dict, where: dict | None) -> bool:
        if not where:
            return True
        for k, v in where.items():
            vals = v if isinstance(v, list) else [v]
            if row.get(k) not in vals:
                return False
        return True

    @staticmethod
    def _chroma_where(where: dict | None) -> dict | None:
        if not where:
            return None
        conds = [{k: {"$in": v}} if isinstance(v, list) else {k: v} for k, v in where.items()]
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def dense(self, query: str, k: int, where: dict | None = None) -> list[str]:
        res = self.col.query(query_embeddings=[embed_query(query)], n_results=k, where=self._chroma_where(where))
        return res["ids"][0]

    def sparse(self, query: str, k: int, where: dict | None = None) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order:
            if scores[i] <= 0 or len(out) >= k:
                break
            if self._allowed(self.rows[i], where):
                out.append(self.rows[i]["id"])
        return out


# Models on MPS/CUDA are not safe to drive from several threads at once (the eval runs questions in
# parallel), so retrieval is serialised; it is a small share of end-to-end latency.
_LOCK = threading.Lock()


@lru_cache
def corpus() -> Corpus:
    return Corpus()


def hybrid_search(queries: list[str], where: dict | None = None, top_k: int | None = None) -> list[Hit]:
    """Retrieve for one or more query phrasings, fuse with RRF, rerank (max over phrasings)."""
    with _LOCK:
        return _hybrid_search(queries, where, top_k)


def _hybrid_search(queries: list[str], where: dict | None, top_k: int | None) -> list[Hit]:
    s = get_settings()
    c = corpus()
    hits: dict[str, Hit] = {}
    for q in queries:
        for rank, cid in enumerate(c.dense(q, s.dense_k, where)):
            h = hits.setdefault(cid, Hit(cid, c.by_id[cid]["text"], c._meta(c.by_id[cid])))
            h.dense_rank = rank if h.dense_rank is None else min(h.dense_rank, rank)
            h.rrf += 1.0 / (s.rrf_k + rank + 1)
        for rank, cid in enumerate(c.sparse(q, s.sparse_k, where)):
            h = hits.setdefault(cid, Hit(cid, c.by_id[cid]["text"], c._meta(c.by_id[cid])))
            h.sparse_rank = rank if h.sparse_rank is None else min(h.sparse_rank, rank)
            h.rrf += 1.0 / (s.rrf_k + rank + 1)

    candidates = sorted(hits.values(), key=lambda h: -h.rrf)[: s.rerank_candidates]
    if not candidates:
        return []
    # Score each candidate against every query phrasing and keep the best: a plain-language question and
    # its regulatory rewrite often match the same clause with very different confidence.
    pairs = [(q, f"{h.meta['context']}\n{h.text}") for q in queries for h in candidates]
    scores = reranker().predict(pairs, batch_size=32)
    n = len(candidates)
    for qi in range(len(queries)):
        for h, z in zip(candidates, scores[qi * n:(qi + 1) * n]):
            z = float(z)
            # Single-label cross-encoders ship with a sigmoid activation; map raw logits if one doesn't.
            p = z if 0.0 <= z <= 1.0 else 1.0 / (1.0 + math.exp(-z))
            h.scores[queries[qi]] = p
            h.rerank = max(h.rerank or 0.0, p)
    ranked = sorted(candidates, key=lambda h: -(h.rerank or 0.0))
    return ranked[: top_k or s.top_k]
