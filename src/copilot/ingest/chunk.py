"""Clause-aware chunking.

Chunks never cross a document or a top-level section, and every chunk carries the list of clause
numbers it contains, so a citation can be checked against the exact provisions that were shown to
the model. Short neighbouring clauses (definitions, heading-only paragraphs) are packed together;
over-long clauses are split on sub-headings, lettered items or paragraph breaks, and each piece
keeps the clause number so the citation stays clause-accurate.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from copilot.ingest.models import Clause, short_doc_name

SPLIT_POINTS = re.compile(r"\n(?=## |\([a-z]{1,3}\)\s|\(\d{1,2}\)\s|\([ivx]{1,5}\)\s)")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    family: str
    doc_title: str
    clauses: list[str]
    text: str                       # what the LLM sees, clause numbers inline
    context: str                    # breadcrumb prepended for embedding / BM25
    page_start: int | None
    page_end: int | None
    source_url: str
    version: str
    clause_titles: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        first, last = self.clauses[0], self.clauses[-1]
        rng = first if first == last else f"{first}–{last.split(' §')[-1]}"
        name = short_doc_name({"doc_id": self.doc_id, "doc_title": self.doc_title})
        return f"{name} {rng}"

    @property
    def embed_text(self) -> str:
        return f"{self.context}\n{self.text}"

    def metadata(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "family": self.family,
            "doc_title": self.doc_title,
            "clauses": "|".join(self.clauses),
            "label": self.label,
            "page_start": self.page_start or 0,
            "page_end": self.page_end or 0,
            "source_url": self.source_url,
            "version": self.version,
            "context": self.context,
        }


def _section_key(c: Clause) -> str:
    """Top-level grouping: the annex/appendix context, else the first number component."""
    if c.number.startswith(("Annex", "Appendix")):
        return c.number.split(" §")[0]
    m = re.match(r"S?(\d+)", c.number)
    return m.group(1) if m else c.number


def _split_long(text: str, max_chars: int) -> list[str]:
    pieces = [p for p in SPLIT_POINTS.split(text) if p.strip()]
    out: list[str] = []
    buf = ""
    for p in pieces:
        if len(p) > max_chars:
            # Fall back to line boundaries for very long paragraphs (tables, definitions lists).
            for line in p.split("\n"):
                if len(buf) + len(line) + 1 > max_chars and buf:
                    out.append(buf)
                    buf = ""
                buf += ("\n" if buf else "") + line
            continue
        if len(buf) + len(p) + 1 > max_chars and buf:
            out.append(buf)
            buf = ""
        buf += ("\n" if buf else "") + p
    if buf:
        out.append(buf)
    return out


def _context(c: Clause) -> str:
    trail = " > ".join(c.parents)
    head = f"{c.number} {c.title}".strip()
    return " | ".join(x for x in (c.doc_title, trail, head) if x)


def _make_id(doc_id: str, first: str, n: int) -> str:
    h = hashlib.sha1(f"{doc_id}|{first}|{n}".encode()).hexdigest()[:8]
    slug = re.sub(r"[^A-Za-z0-9.]+", "_", first).strip("_")
    return f"{doc_id}::{slug}::{h}"


def chunk_document(clauses: list[Clause], max_chars: int = 2400, target_chars: int = 1400) -> list[Chunk]:
    chunks: list[Chunk] = []
    group: list[Clause] = []

    def emit(group: list[Clause]):
        if not group:
            return
        first = group[0]
        text = "\n\n".join(c.text for c in group)
        chunks.append(Chunk(
            chunk_id=_make_id(first.doc_id, first.number, len(chunks)),
            doc_id=first.doc_id, family=first.family, doc_title=first.doc_title,
            clauses=[c.number for c in group], text=text, context=_context(first),
            page_start=first.page_start, page_end=group[-1].page_end or group[-1].page_start,
            source_url=first.source_url, version=first.version,
            clause_titles={c.number: c.title for c in group if c.title},
        ))

    for c in clauses:
        if len(c.text) > max_chars:
            emit(group)
            group = []
            parts = _split_long(c.text, max_chars)
            for i, part in enumerate(parts):
                body = part if i == 0 else f"[{c.number}, continued]\n{part}"
                piece = Clause(**{**c.__dict__, "text": body})
                emit([piece])
            continue
        size = sum(len(x.text) for x in group)
        if group and (_section_key(c) != _section_key(group[-1]) or size + len(c.text) > target_chars):
            # A heading-only clause should travel with what follows it, not be stranded.
            if size < 200 and _section_key(c) == _section_key(group[-1]) and size + len(c.text) <= max_chars:
                group.append(c)
                continue
            emit(group)
            group = []
        group.append(c)
    emit(group)
    return chunks
