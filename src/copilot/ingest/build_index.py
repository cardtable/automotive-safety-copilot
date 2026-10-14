"""Parse the downloaded corpus into clauses and chunks, embed them, and persist to ChromaDB.

Outputs (under data/processed and data/chroma):
  clauses.jsonl   every parsed provision, for clause lookup in the UI and eval
  chunks.jsonl    retrieval units with clause lists (also feeds the BM25 index)
  manifest.json   corpus statistics (documents, pages, clauses, chunks)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chromadb
import pymupdf

from copilot.config import get_settings
from copilot.ingest.chunk import chunk_document
from copilot.ingest.parse_fmvss import parse_fmvss
from copilot.ingest.parse_pdf import parse_pdf

COLLECTION = "regulations"
# One printed CFR page holds roughly 6,500 characters of the two-column regulatory text.
CFR_CHARS_PER_PAGE = 6500


def parse_all(docs: list[dict]):
    clauses_by_doc = {}
    for d in docs:
        path = Path(d["path"])
        clauses = parse_fmvss(path, d) if d["family"] == "FMVSS" else parse_pdf(path, d)
        clauses_by_doc[d["doc_id"]] = clauses
        print(f"  {d['doc_id']:22s} {len(clauses):5d} clauses")
    return clauses_by_doc


def page_count(d: dict, clauses) -> tuple[int, str]:
    if d["path"].endswith(".pdf"):
        return pymupdf.open(d["path"]).page_count, "pdf"
    chars = sum(len(c.text) for c in clauses)
    return max(1, round(chars / CFR_CHARS_PER_PAGE)), "cfr-estimate"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-embed", action="store_true", help="only parse and chunk")
    args = ap.parse_args()

    s = get_settings()
    docs = json.loads((s.raw_dir / "documents.json").read_text())
    s.processed_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing")
    clauses_by_doc = parse_all(docs)

    chunks = []
    stats = []
    with (s.processed_dir / "clauses.jsonl").open("w") as fc:
        for d in docs:
            cl = clauses_by_doc[d["doc_id"]]
            for c in cl:
                fc.write(json.dumps(c.to_dict()) + "\n")
            doc_chunks = chunk_document(cl, max_chars=s.chunk_max_chars)
            chunks += doc_chunks
            pages, basis = page_count(d, cl)
            stats.append({
                "doc_id": d["doc_id"], "family": d["family"], "title": cl[0].doc_title if cl else d.get("title"),
                "version": d["version"], "source_url": d["source_url"], "pages": pages, "page_basis": basis,
                "clauses": len(cl), "chunks": len(doc_chunks),
            })

    with s.chunks_file.open("w") as f:
        for ch in chunks:
            f.write(json.dumps({"id": ch.chunk_id, "text": ch.text, "embed_text": ch.embed_text,
                                "clause_titles": ch.clause_titles, **ch.metadata()}) + "\n")

    totals = {
        "documents": len(stats),
        "pdf_pages": sum(x["pages"] for x in stats if x["page_basis"] == "pdf"),
        "cfr_pages_estimated": sum(x["pages"] for x in stats if x["page_basis"] == "cfr-estimate"),
        "clauses": sum(x["clauses"] for x in stats),
        "chunks": len(chunks),
        "by_family": {},
    }
    for x in stats:
        fam = totals["by_family"].setdefault(x["family"], {"documents": 0, "pages": 0, "clauses": 0, "chunks": 0})
        fam["documents"] += 1
        fam["pages"] += x["pages"]
        fam["clauses"] += x["clauses"]
        fam["chunks"] += x["chunks"]
    totals["total_pages"] = totals["pdf_pages"] + totals["cfr_pages_estimated"]
    s.manifest_file.write_text(json.dumps({"totals": totals, "documents": stats,
                                           "built_at": time.strftime("%Y-%m-%d %H:%M")}, indent=2))
    print(f"\n{totals['documents']} documents, {totals['total_pages']} pages "
          f"({totals['pdf_pages']} PDF + ~{totals['cfr_pages_estimated']} CFR), "
          f"{totals['clauses']} clauses -> {totals['chunks']} chunks")

    if args.skip_embed:
        return

    from copilot.retrieval.embeddings import embed_passages

    print("Embedding")
    vectors = embed_passages([c.embed_text for c in chunks])
    client = chromadb.PersistentClient(path=str(s.chroma_dir))
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # noqa: BLE001 - collection may not exist yet
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine", "embed_model": s.embed_model})
    B = 1000
    for i in range(0, len(chunks), B):
        batch = chunks[i:i + B]
        col.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata() for c in batch],
            embeddings=vectors[i:i + B],
        )
    print(f"Indexed {col.count()} chunks into {s.chroma_dir} (collection '{COLLECTION}')")


if __name__ == "__main__":
    main()
