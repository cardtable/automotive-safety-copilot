"""Streamlit front end for the Automotive Safety Standards Copilot.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import streamlit as st

from copilot.config import get_settings

st.set_page_config(page_title="Safety Standards Copilot", page_icon="🚗", layout="wide")

S = get_settings()
ROOT = Path(__file__).resolve().parents[1]
NODE_LABELS = {
    "analyze": "Planning search", "retrieve": "Hybrid retrieval + rerank", "grade": "Grading evidence",
    "rewrite": "Rewriting query", "generate": "Drafting cited answer", "verify": "Verifying citations",
    "revise": "Revising failed citations", "finalize": "Finalizing", "abstain_scope": "Abstaining",
    "abstain_evidence": "Abstaining", "abstain_model": "Abstaining",
}
EXAMPLES = [
    "How early must an AEBS give a collision warning before emergency braking under UN R152?",
    "What HIC limit applies to the 50th percentile male dummy in FMVSS 208 frontal barrier tests?",
    "At what speed is the Euro NCAP side pole impact test run?",
    "Compare the strength requirements for safety-belt anchorages in FMVSS 210 and UN R14.",
    "What does FMVSS 302 require for interior flammability?",
]

st.markdown("""
<style>
.claim {margin: 0 0 .55rem 0; line-height: 1.55}
.ref {font-size: .72rem; vertical-align: super; padding: 0 .18rem; border-radius: 4px;
      background: rgba(49,130,206,.15); color: inherit; text-decoration: none; margin-left: 1px}
.abstain {border-left: 4px solid #d69e2e; padding: .6rem .9rem; background: rgba(214,158,46,.10);
          border-radius: 4px}
mark {background: rgba(236,201,75,.45); padding: 0 1px; border-radius: 2px}
.small {font-size: .82rem; opacity: .8}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def manifest() -> dict | None:
    return json.loads(S.manifest_file.read_text()) if S.manifest_file.exists() else None


@st.cache_data
def clauses_by_doc() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    p = S.processed_dir / "clauses.jsonl"
    if p.exists():
        for line in p.open():
            c = json.loads(line)
            out.setdefault(c["doc_id"], []).append(c)
    return out


@st.cache_resource(show_spinner="Loading retrieval models and index…")
def graph():
    from copilot.graph.pipeline import get_graph
    from copilot.retrieval.embeddings import embedder, reranker
    from copilot.retrieval.hybrid import corpus

    corpus(), embedder(), reranker()
    return get_graph()


def highlight(excerpt: str, quote: str) -> str:
    esc = html.escape(excerpt)
    words = [re.escape(html.escape(w)) for w in quote.split()]
    if words:
        pat = r"\s+".join(words)
        esc = re.sub(f"({pat})", r"<mark>\1</mark>", esc, count=1, flags=re.IGNORECASE)
    return esc.replace("\n", "<br>")


def render_answer(state: dict) -> None:
    f = state["final"]
    if state.get("abstained"):
        st.markdown(f"<div class='abstain'><b>No grounded answer.</b> {html.escape(f['abstain_reason'])}</div>",
                    unsafe_allow_html=True)
        if f.get("closest"):
            st.caption("Closest provisions retrieved: " +
                       ", ".join(f"{c['label']} (relevance {c['rerank']:.2f})" for c in f["closest"]))
        return
    if f.get("missing_refs"):
        st.info(f"Not in the indexed corpus: {', '.join(f['missing_refs'])}. The answer covers only indexed text.")
    body = []
    for c in f["claims"]:
        marks = "".join(f"<span class='ref'>{n}</span>" for n in c["refs"])
        body.append(f"<p class='claim'>{html.escape(c['text'])}{marks}</p>")
    st.markdown("".join(body), unsafe_allow_html=True)

    st.markdown("**Sources**")
    for r in f["references"]:
        page = f" · p. {r['page']}" if r.get("page") else ""
        with st.expander(f"[{r['n']}] {r['citation']}  —  {r['version']}{page}"):
            st.markdown(f"> {r['quote']}")
            st.markdown(f"<div class='small'>{highlight(r['excerpt'], r['quote'])}</div>", unsafe_allow_html=True)
            st.markdown(f"[Official source]({r['source_url']})")
    v = f["verification"]
    st.caption(f"Citation check: {v['grounded_citations']}/{v['n_citations']} citations verified verbatim "
               f"against the cited clause"
               + (f" · {len(f['dropped_claims'])} unverifiable statement(s) removed" if f["dropped_claims"] else ""))


def render_trace(state: dict) -> None:
    with st.expander("Pipeline trace"):
        for t in state.get("trace", []):
            info = {k: v for k, v in t.items() if k not in ("node", "t")}
            st.markdown(f"**{t['node']}** — `{json.dumps(info, default=str)[:600]}`")
        hits = state.get("hits") or []
        if hits:
            st.markdown("**Retrieved excerpts (after rerank)**")
            st.dataframe(
                [{"label": h["label"], "rerank": h["rerank"], "dense rank": h["dense_rank"],
                  "BM25 rank": h["sparse_rank"]} for h in hits],
                hide_index=True, width="stretch",
            )


def run_question(q: str) -> dict:
    g = graph()
    state: dict = {"question": q, "trace": []}
    with st.status("Thinking…", expanded=False) as status:
        for update in g.stream(state, stream_mode="updates"):
            for node, delta in update.items():
                status.update(label=NODE_LABELS.get(node, node) + "…")
                state.update(delta or {})
        status.update(label="Done", state="complete")
    return state


# ---------------------------------------------------------------- layout

m = manifest()
with st.sidebar:
    st.title("🚗 Safety Standards Copilot")
    st.caption("Clause-cited answers over FMVSS, UN/ECE Regulations and Euro NCAP protocols. "
               "Answers are built only from retrieved text and every citation is checked verbatim; "
               "if nothing grounds the question, the copilot abstains.")
    if m:
        t = m["totals"]
        c1, c2 = st.columns(2)
        c1.metric("Documents", t["documents"])
        c2.metric("Pages", f"{t['total_pages']:,}")
        c1.metric("Clauses", f"{t['clauses']:,}")
        c2.metric("Chunks", f"{t['chunks']:,}")
        for fam, v in t["by_family"].items():
            st.caption(f"**{fam}** — {v['documents']} docs · {v['pages']} pages · {v['clauses']} clauses")
    else:
        st.warning("No index found. Run `copilot-download` then `copilot-index`.")
    st.caption(f"LLM: `{S.llm_model}` · embeddings: `{S.embed_model}` · reranker: `{S.rerank_model}`")
    if st.button("Clear conversation"):
        st.session_state.history = []

tab_chat, tab_browse, tab_eval = st.tabs(["Ask", "Browse corpus", "Evaluation"])

with tab_chat:
    st.session_state.setdefault("history", [])
    if not st.session_state.history:
        st.markdown("##### Try a question")
        cols = st.columns(len(EXAMPLES))
        for col, ex in zip(cols, EXAMPLES):
            if col.button(ex, width="stretch"):
                st.session_state.pending = ex
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            render_answer(item["state"])
            render_trace(item["state"])

    q = st.chat_input("Ask about a vehicle safety requirement…") or st.session_state.pop("pending", None)
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                state = run_question(q)
            except Exception as e:  # noqa: BLE001 - surface API/config errors in the UI
                st.error(f"{type(e).__name__}: {e}")
                st.stop()
            render_answer(state)
            render_trace(state)
        st.session_state.history.append({"question": q, "state": state})

with tab_browse:
    docs = clauses_by_doc()
    if m and docs:
        titles = {d["doc_id"]: f"[{d['family']}] {d['title']}" for d in m["documents"]}
        doc_id = st.selectbox("Document", list(titles), format_func=titles.get)
        meta = next(d for d in m["documents"] if d["doc_id"] == doc_id)
        st.caption(f"{meta['version']} · {meta['pages']} pages · {meta['clauses']} clauses · "
                   f"[official source]({meta['source_url']})")
        flt = st.text_input("Filter clauses (number or text)")
        shown = [c for c in docs[doc_id] if not flt or flt.lower() in (c["number"] + " " + c["text"]).lower()]
        st.caption(f"{len(shown)} clauses")
        for c in shown[:200]:
            with st.expander(f"{c['number']}  {c['title'][:90]}"):
                if c["parents"]:
                    st.caption(" › ".join(c["parents"]))
                st.text(c["text"])
                if c.get("page_start"):
                    st.caption(f"p. {c['page_start']}" + (f"–{c['page_end']}" if c.get("page_end") and
                                                           c["page_end"] != c["page_start"] else ""))

with tab_eval:
    rep_path = ROOT / "eval" / "results" / "latest.json"
    if not rep_path.exists():
        st.info("No evaluation run yet. Run `python -m eval.run_eval`.")
    else:
        rep = json.loads(rep_path.read_text())
        st.caption(f"Run {rep['run_at']} · {rep['n_questions']} questions · model `{rep['config']['llm_model']}`")
        s = rep["summary"]
        cols = st.columns(4)
        cols[0].metric("Answer faithfulness", f"{s['faithfulness']:.1%}")
        cols[1].metric("Citation grounding", f"{s['citation_grounding']:.1%}")
        cols[2].metric("Retrieval precision@5", f"{s['retrieval']['precision@5']:.1%}")
        cols[3].metric("Abstention accuracy", f"{s['abstention']['accuracy']:.1%}")
        st.json(s, expanded=False)
        st.dataframe(
            [{"id": r["id"], "type": r["type"], "abstained": r["abstained"],
              "faithfulness": r.get("faithfulness"), "grounding": r.get("citation_grounding"),
              "recall@5": r.get("retrieval", {}).get("recall@5"), "question": r["question"]}
             for r in rep["results"]],
            hide_index=True, width="stretch",
        )
