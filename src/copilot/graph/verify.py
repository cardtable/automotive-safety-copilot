"""Deterministic citation verification.

A citation is grounded only if (1) it points at an excerpt the model was shown, (2) the clause it names is
one of the clauses in that excerpt, and (3) its quote actually occurs in the excerpt text (after normalising
whitespace, hyphenation and typographic quotes; a near-exact fuzzy match is tolerated for PDF artefacts).
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

QUOTE_FUZZ = 0.92


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("­", "")
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)         # de-hyphenate line breaks
    s = re.sub(r"[‘’‛`´]", "'", s)
    s = re.sub(r"[“”„\"]", '"', s)
    s = re.sub(r"[‐-―−]", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def norm_clause(c: str) -> str:
    c = c.strip().lower().replace("paragraph", "").replace("para.", "").replace("§", " ")
    c = re.sub(r"\s+", " ", c)
    c = re.sub(r"\.(?=\s|$|\()", "", c)                 # trailing dots: "5.2.1." -> "5.2.1"
    return c.replace(" ", "")


def clause_matches(cited: str, listed: list[str]) -> bool:
    n = norm_clause(cited)
    for c in listed:
        lc = norm_clause(re.sub(r"\s*\(\d+\)$", "", c))   # drop dedupe suffixes like " (2)"
        if n == lc or n == lc.lstrip("s") or n.lstrip("s") == lc.lstrip("s"):
            return True
        if n.startswith(lc + "(") or n.lstrip("s").startswith(lc.lstrip("s") + "("):   # S5.2(a) within S5.2
            return True
    return False


def quote_score(quote: str, source_text: str) -> float:
    q, t = norm_text(quote).strip(' ".'), norm_text(source_text)
    if not q:
        return 0.0
    if q in t:
        return 1.0
    # Allow an ellipsis-joined quote only if every fragment is present.
    frags = [f.strip() for f in re.split(r"\.\.\.|…", q) if f.strip()]
    if len(frags) > 1 and all(f in t for f in frags):
        return 1.0
    m = SequenceMatcher(None, t, q, autojunk=False)
    blocks = m.get_matching_blocks()
    matched = sum(b.size for b in blocks)
    # Matched characters must cluster in one window of the source, not be scattered across it.
    if blocks and matched:
        span = (blocks[-2].a + blocks[-2].size - blocks[0].a) if len(blocks) > 1 else 0
        if span > len(q) * 1.3:
            return matched / len(q) * 0.5
    return matched / len(q)


def verify_answer(answer, sources: list[dict]) -> dict:
    """Return per-claim verification and summary stats. `sources` are the excerpts in prompt order."""
    results = []
    for ci, claim in enumerate(answer.claims):
        checks = []
        for cit in claim.citations:
            ok_src = 1 <= cit.source <= len(sources)
            src = sources[cit.source - 1] if ok_src else None
            ok_clause = bool(src) and clause_matches(cit.clause, src["clauses"])
            qs = quote_score(cit.quote, src["text"]) if src else 0.0
            checks.append({
                "source": cit.source, "clause": cit.clause, "quote": cit.quote,
                "source_ok": ok_src, "clause_ok": ok_clause, "quote_score": round(qs, 3),
                "grounded": ok_src and ok_clause and qs >= QUOTE_FUZZ,
            })
        results.append({
            "claim": claim.text, "citations": checks,
            "grounded": any(c["grounded"] for c in checks),
        })
    n_cit = sum(len(r["citations"]) for r in results)
    return {
        "claims": results,
        "n_claims": len(results),
        "grounded_claims": sum(r["grounded"] for r in results),
        "n_citations": n_cit,
        "grounded_citations": sum(c["grounded"] for r in results for c in r["citations"]),
    }


def problems_text(report: dict) -> str:
    lines = []
    for r in report["claims"]:
        for c in r["citations"]:
            if c["grounded"]:
                continue
            why = []
            if not c["source_ok"]:
                why.append(f"S{c['source']} does not exist")
            else:
                if not c["clause_ok"]:
                    why.append(f"clause '{c['clause']}' is not listed for S{c['source']}")
                if c["quote_score"] < QUOTE_FUZZ:
                    why.append(f"quote not found verbatim in S{c['source']}")
            lines.append(f"- Claim \"{r['claim'][:120]}\": {'; '.join(why)}")
        if not r["citations"]:
            lines.append(f"- Claim \"{r['claim'][:120]}\": no citation")
    return "\n".join(lines)
