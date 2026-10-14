"""Evaluation harness.

    python -m eval.run_eval                     # full pipeline + LLM judges (needs ANTHROPIC_API_KEY)
    python -m eval.run_eval --retrieval-only    # deterministic retrieval metrics, no LLM calls
    python -m eval.run_eval --limit 10 --workers 4

Metrics
  Retrieval (answerable questions, gold clauses, deterministic):
    hit@k, recall@k (share of gold clauses retrieved), MRR, gold-precision@k
  Context precision@5 (LLM judge): share of the top-5 excerpts that are relevant to the question
  Citation grounding (deterministic): share of citations whose clause is in the cited excerpt and whose quote
    occurs verbatim in it - reported for the model's first draft and for the final, verified answer
  Gold clause citation rate: answered questions that cite at least one gold clause
  Faithfulness (LLM judge): share of final claims fully supported by the excerpts they cite
  Answer correctness (LLM judge): share of reference facts stated in the answer
  Abstention: abstain rate on unanswerable questions, false-abstain rate on answerable ones
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from copilot.config import get_settings
from copilot.graph.verify import clause_matches

ROOT = Path(__file__).resolve().parent
KS = (1, 3, 5, 8)


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def is_relevant(hit: dict, gold: list[dict]) -> list[int]:
    """Indices of gold clauses contained in this hit."""
    return [i for i, g in enumerate(gold) if g["doc_id"] == hit["doc_id"] and clause_matches(g["clause"], hit["clauses"])]


def retrieval_metrics(hits: list[dict], gold: list[dict]) -> dict:
    out: dict = {}
    rel = [is_relevant(h, gold) for h in hits]
    first = next((i for i, r in enumerate(rel) if r), None)
    out["mrr"] = 0.0 if first is None else 1.0 / (first + 1)
    for k in KS:
        found = {g for r in rel[:k] for g in r}
        out[f"hit@{k}"] = float(bool(found))
        out[f"recall@{k}"] = len(found) / len(gold)
        out[f"gold_precision@{k}"] = sum(1 for r in rel[:k] if r) / max(1, min(k, len(hits)))
    return out


def answer_text(final: dict) -> str:
    return "\n".join(c["text"] for c in final.get("claims", []))


def run_one(item: dict, retrieval_only: bool) -> dict:
    res: dict = {"id": item["id"], "type": item["type"], "question": item["question"]}
    t0 = time.time()
    if retrieval_only:
        # Same retrieval path as the app, minus the LLM query planner: raw question + regex reference filters.
        from copilot.graph.pipeline import retrieve_for
        from copilot.graph.refs import detect_refs

        hits = retrieve_for([item["question"]], detect_refs(item["question"]).where())
        res["hits"] = [{"doc_id": h.meta["doc_id"], "clauses": h.clauses, "label": h.label, "rerank": h.rerank}
                       for h in hits]
        if item["gold"]:
            res["retrieval"] = retrieval_metrics(res["hits"], item["gold"])
        res["top_rerank"] = max((h.rerank or 0 for h in hits), default=0.0)
        res["seconds"] = round(time.time() - t0, 2)
        return res

    from copilot.graph.pipeline import ask
    from eval.judge import judge_context, judge_correctness, judge_faithfulness

    state = ask(item["question"])
    res["seconds"] = round(time.time() - t0, 2)
    final = state["final"]
    res["abstained"] = bool(state.get("abstained"))
    res["abstain_reason"] = final.get("abstain_reason")
    res["path"] = [t["node"] for t in state["trace"]]
    hits = state.get("hits") or []
    res["hits"] = [{"doc_id": h["doc_id"], "clauses": h["clauses"], "label": h["label"], "rerank": h["rerank"]}
                   for h in hits]
    if item["gold"] and hits:
        res["retrieval"] = retrieval_metrics(hits, item["gold"])
    if hits and item["type"] == "answerable":
        ctx = judge_context(item["question"], hits[:5])
        res["context_relevant"] = [v.relevant for v in sorted(ctx.verdicts, key=lambda v: v.excerpt)][:5]
    if res["abstained"]:
        return res

    res["answer"] = answer_text(final)
    res["references"] = [{k: r[k] for k in ("n", "citation", "clause", "quote", "page")} for r in final["references"]]
    draft = state.get("draft_verification") or {}
    res["draft_grounding"] = {"grounded": draft.get("grounded_citations", 0), "total": draft.get("n_citations", 0)}
    kept = sum(len(c["refs"]) for c in final["claims"])
    res["final_grounding"] = {"grounded": kept, "total": kept}
    res["dropped_claims"] = len(final.get("dropped_claims", []))
    res["revised"] = "revise" in res["path"]
    if item["gold"]:
        res["cites_gold"] = any(
            r["chunk_id"].split("::")[0] == g["doc_id"] and clause_matches(g["clause"], [r["clause"]])
            for r in final["references"] for g in item["gold"]
        )
    fr = judge_faithfulness(item["question"], final["claims"], final["references"])
    res["faith_verdicts"] = [v.model_dump() for v in fr.verdicts]
    if item.get("facts"):
        cr = judge_correctness(item["question"], res["answer"], item["facts"])
        res["facts_present"] = [f.present for f in sorted(cr.facts, key=lambda f: f.fact)]
    return res


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(results: list[dict], retrieval_only: bool) -> dict:
    ans = [r for r in results if r["type"] == "answerable"]
    una = [r for r in results if r["type"] == "unanswerable"]
    ret = [r["retrieval"] for r in ans if "retrieval" in r]
    s: dict = {"n_answerable": len(ans), "n_unanswerable": len(una), "retrieval": {}}
    if ret:
        for key in ret[0]:
            s["retrieval"][key] = round(mean([x[key] for x in ret]), 4)
    if retrieval_only:
        thr = get_settings().min_rerank_score
        s["evidence_gate"] = {
            "threshold": thr,
            "answerable_passing": round(mean([float(r["top_rerank"] >= thr) for r in ans]), 4),
            "unanswerable_blocked": round(mean([float(r["top_rerank"] < thr) for r in una]), 4),
        }
        return s

    ctx = [v for r in ans for v in r.get("context_relevant", [])]
    s["retrieval"]["precision@5"] = round(mean([float(v) for v in ctx]), 4)
    answered = [r for r in results if not r.get("abstained") and "faith_verdicts" in r]
    verdicts = [v["verdict"] for r in answered for v in r["faith_verdicts"]]
    s["faithfulness"] = round(mean([float(v == "supported") for v in verdicts]), 4)
    s["faithfulness_lenient"] = round(mean([{"supported": 1, "partial": .5}.get(v, 0) for v in verdicts]), 4)
    s["claims_judged"] = len(verdicts)
    s["claim_verdicts"] = {k: verdicts.count(k) for k in ("supported", "partial", "unsupported")}
    dg = [r["draft_grounding"] for r in answered]
    s["draft_citation_grounding"] = round(sum(d["grounded"] for d in dg) / max(1, sum(d["total"] for d in dg)), 4)
    s["citation_grounding"] = 1.0 if answered else float("nan")   # final answers keep only verified citations
    s["answers_revised"] = sum(r.get("revised", False) for r in answered)
    s["claims_dropped"] = sum(r.get("dropped_claims", 0) for r in answered)
    s["gold_clause_cited"] = round(mean([float(r["cites_gold"]) for r in answered if "cites_gold" in r]), 4)
    facts = [f for r in answered for f in r.get("facts_present", [])]
    s["answer_correctness"] = round(mean([float(f) for f in facts]), 4)
    tp = sum(r["abstained"] for r in una)
    fa = sum(r["abstained"] for r in ans)
    s["abstention"] = {
        "unanswerable_abstained": f"{tp}/{len(una)}",
        "answerable_false_abstain": f"{fa}/{len(ans)}",
        "accuracy": round((tp + len(ans) - fa) / max(1, len(results)), 4),
    }
    s["latency_s"] = {"mean": round(mean([r["seconds"] for r in results]), 1),
                      "p90": round(sorted(r["seconds"] for r in results)[int(0.9 * (len(results) - 1))], 1)}
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(ROOT / "dataset.jsonl"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retrieval-only", action="store_true")
    a = ap.parse_args()

    data = load_dataset(Path(a.dataset))
    if a.ids:
        data = [d for d in data if d["id"] in a.ids]
    if a.limit:
        data = data[: a.limit]

    from copilot.retrieval.hybrid import corpus
    corpus()  # load index once before threads start

    results: list[dict] = []
    workers = 1 if a.retrieval_only else a.workers
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, d, a.retrieval_only): d for d in data}
        for i, f in enumerate(as_completed(futs), 1):
            d = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001 - record and keep going
                r = {"id": d["id"], "type": d["type"], "question": d["question"], "error": repr(e),
                     "traceback": traceback.format_exc()[-1500:], "abstained": True, "seconds": 0}
            results.append(r)
            tag = "ERR" if "error" in r else ("ABSTAIN" if r.get("abstained") else "ok")
            print(f"[{i}/{len(data)}] {d['id']:24s} {tag:8s} {r.get('seconds', 0):6.1f}s")

    order = {d["id"]: i for i, d in enumerate(data)}
    results.sort(key=lambda r: order[r["id"]])
    s = get_settings()
    summary = summarize([r for r in results if "error" not in r], a.retrieval_only)
    summary["errors"] = sum("error" in r for r in results)
    from copilot.llm import USAGE
    report = {
        "run_at": time.strftime("%Y-%m-%d %H:%M"),
        "mode": "retrieval-only" if a.retrieval_only else "full",
        "n_questions": len(results),
        "config": {"llm_model": s.llm_model, "judge_model": s.judge_model, "effort": s.llm_effort,
                   "embed_model": s.embed_model, "rerank_model": s.rerank_model, "top_k": s.top_k,
                   "min_rerank_score": s.min_rerank_score},
        "usage": {"calls": USAGE.calls, "input_tokens": USAGE.input_tokens, "output_tokens": USAGE.output_tokens,
                  "cache_read_tokens": USAGE.cache_read_tokens, "by_step": USAGE.by_step},
        "summary": summary,
        "results": results,
    }
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{'retrieval' if a.retrieval_only else 'full'}-{stamp}.json"
    (out_dir / name).write_text(json.dumps(report, indent=2))
    latest = "latest_retrieval.json" if a.retrieval_only else "latest.json"
    (out_dir / latest).write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved eval/results/{name}")


if __name__ == "__main__":
    main()
