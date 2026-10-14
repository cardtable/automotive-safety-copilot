"""Detect explicit regulation references in a question ("FMVSS 208", "UN R152", "ECE R94",
"Regulation No. 16", "Euro NCAP") and map them onto indexed documents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from copilot.config import get_settings

FMVSS = re.compile(r"\b(?:FMVSS|MVSS|Standard)\s*(?:No\.?\s*)?(\d{3}[a-z]?)\b|§\s*571\.(\d{3}[a-z]?)", re.IGNORECASE)
UNECE = re.compile(
    r"\b(?:UN(?:/ECE)?|ECE|UNECE)[\s-]*(?:Regulation\s*)?(?:No\.?\s*)?R?\s*(\d{1,3})\b"
    r"|\bRegulation\s+No\.?\s*(\d{1,3})\b|\bR(\d{2,3})\b",
    re.IGNORECASE,
)
NCAP = re.compile(r"\b(?:Euro\s*NCAP|NCAP)\b", re.IGNORECASE)


@dataclass
class Refs:
    doc_ids: list[str] = field(default_factory=list)       # referenced and indexed
    missing: list[str] = field(default_factory=list)       # referenced but not in the corpus
    families: list[str] = field(default_factory=list)

    def where(self) -> dict | None:
        if self.doc_ids:
            return {"doc_id": self.doc_ids}
        if self.families:
            return {"family": self.families}
        return None


@lru_cache
def indexed_docs() -> dict[str, dict]:
    m = json.loads(get_settings().manifest_file.read_text())
    return {d["doc_id"]: d for d in m["documents"]}


def catalog_text() -> str:
    lines = []
    for d in indexed_docs().values():
        lines.append(f"- {d['title']} [{d['family']}]")
    return "\n".join(lines)


def detect_refs(question: str) -> Refs:
    docs = indexed_docs()
    r = Refs()
    for m in FMVSS.finditer(question):
        num = (m.group(1) or m.group(2)).lower()
        did = f"fmvss-{num}"
        (r.doc_ids if did in docs else r.missing).append(did if did in docs else f"FMVSS {num}")
    for m in UNECE.finditer(question):
        num = int(next(g for g in m.groups() if g))
        did = f"unece-r{num}"
        (r.doc_ids if did in docs else r.missing).append(did if did in docs else f"UN R{num}")
    if NCAP.search(question):
        r.families.append("Euro NCAP")
        # A question naming Euro NCAP plus a regulation should still search the protocols.
        if r.doc_ids:
            r.doc_ids += [d for d, v in docs.items() if v["family"] == "Euro NCAP"]
    r.doc_ids = list(dict.fromkeys(r.doc_ids))
    r.missing = list(dict.fromkeys(r.missing))
    return r
