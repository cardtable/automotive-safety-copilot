# Build progress (resume checklist)

Goal: Automotive Safety Standards Copilot — Python, LangGraph, RAG, ChromaDB, Streamlit.
Claims to back with real numbers: 1,000+ pages of FMVSS / UN-ECE / Euro NCAP, clause-level citations,
abstention when ungrounded, eval harness (retrieval precision + citation grounding) with 90%+ faithfulness.

## Steps
- [x] Project skeleton, pyproject (uv venv, Python 3.12), config/sources.yaml
- [x] Downloader (eCFR API, UN ODS API documents.un.org, cdn.euroncap.com)
- [x] Run download, count pages
- [x] Parsers: eCFR XML -> clauses; PDF -> clauses (UN numbering / NCAP numbering)
- [x] Chunker + Chroma index + BM25
- [x] Hybrid retrieval (dense+BM25 RRF) + cross-encoder rerank
- [x] LangGraph pipeline: analyze -> retrieve -> grade -> (rewrite loop) -> generate -> verify citations -> abstain
- [x] Streamlit app
- [x] Eval dataset (answerable w/ gold clauses + unanswerable)
- [x] Eval harness: retrieval P@k/R@k/MRR, citation grounding, faithfulness (LLM judge), abstention
- [ ] Run eval with ANTHROPIC_API_KEY, record results in README
- [x] Tests, README

## Status notes
- Corpus: 46 docs, ~1,767 pages (1,545 PDF + ~222 CFR est.), 11,364 clauses -> 3,565 chunks (data/processed/manifest.json)
- Retrieval-only eval: hit@5 98.4%, recall@5 95.2%, MRR 0.838 (eval/results/latest_retrieval.json)
- 12 tests pass; Streamlit verified via AppTest with stub LLM
- NEXT: put ANTHROPIC_API_KEY in .env, run `python -m eval.run_eval --workers 4`, inspect failures, tune, fill README Results
- Graph verified offline with stub LLM (all paths). Real LLM run needs ANTHROPIC_API_KEY.
- Rebuild: `.venv/bin/python -m copilot.ingest.download && .venv/bin/python -m copilot.ingest.build_index`

## Notes
- unece.org blocks scripted downloads (403); use documents.un.org/api/symbol/access?s=<symbol>&l=en&t=pdf
- eCFR full endpoint requires Accept-Encoding gzip.
- No ANTHROPIC_API_KEY in env as of 2026-10-14; LLM steps need it.
