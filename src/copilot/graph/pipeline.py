"""The LangGraph workflow.

    analyze ──► retrieve ──► grade ──► generate ──► verify ──► finalize
                   ▲           │                      │
                   └─ rewrite ◄┘ (weak evidence)      └─► generate (one revision if citations fail)
                               └─► abstain (still weak after rewrites, or out of scope)

Abstention happens at three gates: the query is out of scope, retrieval finds no evidence above the
cross-encoder threshold, or the generator/verifier cannot ground a single claim.
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from copilot.config import get_settings
from copilot.graph import prompts
from copilot.graph.refs import catalog_text, detect_refs
from copilot.graph.schemas import Answer, QueryPlan, Rewrites
from copilot.graph.verify import problems_text, verify_answer
from copilot.ingest.models import short_doc_name
from copilot.llm import LLMRefusal, structured
from copilot.retrieval.hybrid import hybrid_search


class State(TypedDict, total=False):
    question: str
    in_scope: bool
    queries: list[str]
    tried_queries: list[str]
    where: dict | None
    missing_refs: list[str]
    attempt: int
    hits: list[dict]
    evidence_ok: bool
    answer: dict | None
    verification: dict | None
    draft_verification: dict | None
    revisions: int
    abstained: bool
    abstain_reason: str
    final: dict
    trace: list[dict]


def _trace(state: State, node: str, **info: Any) -> list[dict]:
    return [*state.get("trace", []), {"node": node, "t": round(time.time(), 3), **info}]


def _hit_dict(h) -> dict:
    return {
        "id": h.id, "text": h.text, "label": h.label, "clauses": h.clauses,
        "doc_id": h.meta["doc_id"], "doc_title": h.meta["doc_title"], "family": h.meta["family"],
        "page_start": h.meta["page_start"], "page_end": h.meta["page_end"],
        "source_url": h.meta["source_url"], "version": h.meta["version"], "context": h.meta["context"],
        "clause_titles": h.meta.get("clause_titles", {}),
        "rerank": round(h.rerank or 0.0, 4), "dense_rank": h.dense_rank, "sparse_rank": h.sparse_rank,
    }


# ---------------------------------------------------------------- nodes

def analyze(state: State) -> State:
    q = state["question"]
    refs = detect_refs(q)
    plan = structured("analyze", prompts.ANALYZE_SYSTEM.format(catalog=catalog_text()),
                      f"Question: {q}", QueryPlan, effort="low", max_tokens=2000)
    queries = [q] + [x for x in plan.search_queries if x.strip() and x.strip() != q][:2]
    return {
        "in_scope": plan.in_scope, "queries": queries, "tried_queries": [], "where": refs.where(),
        "missing_refs": refs.missing, "attempt": 0, "revisions": 0,
        "trace": _trace(state, "analyze", in_scope=plan.in_scope, queries=queries,
                        where=refs.where(), missing=refs.missing),
    }


def retrieve_for(queries: list[str], where: dict | None):
    """Hybrid search; comparison questions naming several regulations are searched per regulation so
    one regime cannot crowd the other out of the context window, then merged by rerank score."""
    regs = [d for d in (where or {}).get("doc_id") or [] if not d.startswith("ncap-")]
    if len(regs) < 2:
        return hybrid_search(queries, where=where)
    per = max(3, get_settings().top_k // len(regs))
    merged = {}
    for d in regs:
        for h in hybrid_search(queries, where={"doc_id": d}, top_k=per):
            merged[h.id] = h
    return sorted(merged.values(), key=lambda h: -(h.rerank or 0.0))


def retrieve(state: State) -> State:
    hits = retrieve_for(state["queries"], state.get("where"))
    return {
        "hits": [_hit_dict(h) for h in hits],
        "tried_queries": state.get("tried_queries", []) + state["queries"],
        "attempt": state.get("attempt", 0) + 1,
        "trace": _trace(state, "retrieve", attempt=state.get("attempt", 0) + 1,
                        top=[(h.label, round(h.rerank or 0, 3)) for h in hits[:5]]),
    }


def grade(state: State) -> State:
    s = get_settings()
    best = max((h["rerank"] for h in state["hits"]), default=0.0)
    ok = best >= s.min_rerank_score
    # Keep only excerpts that clear the bar (but always keep the top few for context).
    kept = [h for i, h in enumerate(state["hits"]) if h["rerank"] >= s.min_rerank_score * 0.5 or i < 3]
    return {"evidence_ok": ok, "hits": kept if ok else state["hits"],
            "trace": _trace(state, "grade", best=round(best, 4), ok=ok, kept=len(kept))}


def rewrite(state: State) -> State:
    tried = "\n".join(f"- {q}" for q in state.get("tried_queries", []))
    out = structured("rewrite", prompts.REWRITE_SYSTEM,
                     f"Question: {state['question']}\n\nFailed queries:\n{tried}", Rewrites,
                     effort="low", max_tokens=2000)
    # Drop document filters on the retry: the user may have named the wrong regulation.
    return {"queries": out.search_queries[:3] or [state["question"]], "where": None,
            "trace": _trace(state, "rewrite", queries=out.search_queries)}


def _sources_block(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        pages = f", p. {h['page_start']}" if h["page_start"] else ""
        parts.append(
            f"[S{i}] {h['doc_title']} ({h['version']}{pages})\n"
            f"Section path: {h['context']}\n"
            f"Clauses in this excerpt: {', '.join(h['clauses'])}\n"
            f"<excerpt>\n{h['text']}\n</excerpt>"
        )
    return "\n\n".join(parts)


def generate(state: State) -> State:
    hits = state["hits"]
    user = f"{_sources_block(hits)}\n\nQuestion: {state['question']}"
    if state.get("missing_refs"):
        user += ("\n\nNote: the question names " + ", ".join(state["missing_refs"]) +
                 ", which is not in the indexed corpus. Do not answer about it from memory.")
    if state.get("verification") and state.get("revisions", 0) > 0:
        user += "\n\n" + prompts.REVISE_NOTE.format(problems=problems_text(state["verification"]))
    try:
        ans = structured("generate", prompts.ANSWER_SYSTEM, user, Answer, max_tokens=12000)
    except LLMRefusal as e:
        ans = Answer(abstain=True, abstain_reason=str(e))
    return {"answer": ans.model_dump(),
            "trace": _trace(state, "generate", abstain=ans.abstain, n_claims=len(ans.claims))}


def verify(state: State) -> State:
    ans = Answer.model_validate(state["answer"])
    report = verify_answer(ans, state["hits"])
    first = {} if state.get("draft_verification") else {"draft_verification": report}
    return {"verification": report, **first,
            "trace": _trace(state, "verify", grounded=f"{report['grounded_claims']}/{report['n_claims']}",
                            citations=f"{report['grounded_citations']}/{report['n_citations']}")}


def revise(state: State) -> State:
    return {"revisions": state.get("revisions", 0) + 1, "trace": _trace(state, "revise")}


def finalize(state: State) -> State:
    """Keep only verified claims and their verified citations; number references in order of use."""
    report = state["verification"]
    hits = state["hits"]
    refs: list[dict] = []
    ref_index: dict[tuple[int, str], int] = {}
    claims_out = []
    dropped = []
    for r in report["claims"]:
        if not r["grounded"]:
            dropped.append(r["claim"])
            continue
        marks = []
        for c in r["citations"]:
            if not c["grounded"]:
                continue
            key = (c["source"], c["clause"])
            if key not in ref_index:
                h = hits[c["source"] - 1]
                ref_index[key] = len(refs) + 1
                refs.append({
                    "n": ref_index[key],
                    "citation": f"{short_doc_name(h)} {c['clause']}",
                    "clause": c["clause"], "doc_title": h["doc_title"], "family": h["family"],
                    "quote": c["quote"], "page": h["page_start"] or None, "source_url": h["source_url"],
                    "version": h["version"], "chunk_id": h["id"], "excerpt": h["text"],
                })
            marks.append(ref_index[key])
        claims_out.append({"text": r["claim"], "refs": sorted(set(marks))})
    if not claims_out:
        return _abstain(state, "None of the drafted statements could be verified against the retrieved "
                               "regulation text.", dropped=dropped)
    return {
        "abstained": False,
        "final": {"claims": claims_out, "references": refs, "dropped_claims": dropped,
                  "verification": report, "missing_refs": state.get("missing_refs", [])},
        "trace": _trace(state, "finalize", claims=len(claims_out), dropped=len(dropped)),
    }


def _abstain(state: State, reason: str, **extra) -> State:
    return {
        "abstained": True, "abstain_reason": reason,
        "final": {"claims": [], "references": [], "abstain_reason": reason,
                  "missing_refs": state.get("missing_refs", []),
                  "closest": [{"label": h["label"], "rerank": h["rerank"]} for h in state.get("hits", [])[:3]],
                  **extra},
        "trace": _trace(state, "abstain", reason=reason),
    }


def abstain_scope(state: State) -> State:
    return _abstain(state, "The question is not about vehicle safety regulations, so there is no source to "
                           "ground an answer in.")


def abstain_evidence(state: State) -> State:
    miss = state.get("missing_refs") or []
    reason = ("No provision in the indexed FMVSS, UN/ECE and Euro NCAP documents addresses this question"
              + (f" ({', '.join(miss)} is not in the corpus)" if miss else "") + ".")
    return _abstain(state, reason)


def abstain_model(state: State) -> State:
    return _abstain(state, state["answer"].get("abstain_reason") or "The retrieved provisions do not answer "
                                                                    "the question.")


# ---------------------------------------------------------------- routing

def route_scope(state: State) -> str:
    return "retrieve" if state["in_scope"] else "abstain_scope"


def route_grade(state: State) -> str:
    if state["evidence_ok"]:
        return "generate"
    if state["attempt"] < get_settings().max_retrieval_attempts:
        return "rewrite"
    return "abstain_evidence"


def route_generate(state: State) -> str:
    return "abstain_model" if state["answer"]["abstain"] else "verify"


def route_verify(state: State) -> str:
    v = state["verification"]
    all_ok = v["grounded_citations"] == v["n_citations"] and v["grounded_claims"] == v["n_claims"]
    if not all_ok and state.get("revisions", 0) < 1:
        return "revise"
    return "finalize"


def build_graph():
    g = StateGraph(State)
    for name, fn in [("analyze", analyze), ("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
                     ("generate", generate), ("verify", verify), ("revise", revise), ("finalize", finalize),
                     ("abstain_scope", abstain_scope), ("abstain_evidence", abstain_evidence),
                     ("abstain_model", abstain_model)]:
        g.add_node(name, fn)
    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", route_scope, ["retrieve", "abstain_scope"])
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_grade, ["generate", "rewrite", "abstain_evidence"])
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("generate", route_generate, ["verify", "abstain_model"])
    g.add_conditional_edges("verify", route_verify, ["revise", "finalize"])
    g.add_edge("revise", "generate")
    for n in ("finalize", "abstain_scope", "abstain_evidence", "abstain_model"):
        g.add_edge(n, END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def ask(question: str) -> State:
    return get_graph().invoke({"question": question, "trace": []})
