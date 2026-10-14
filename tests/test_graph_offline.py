"""Exercise every LangGraph route against the real index with a stubbed LLM (no API calls)."""

import re

import pytest

from copilot.config import get_settings

pytestmark = pytest.mark.skipif(not get_settings().chunks_file.exists(), reason="index not built")


@pytest.fixture
def stub(monkeypatch):
    import copilot.graph.pipeline as P
    from copilot.graph.schemas import Answer, Citation, Claim, QueryPlan, Rewrites

    def fake(step, system, user, schema, **kw):
        if schema is QueryPlan:
            return QueryPlan(in_scope="cake" not in user, search_queries=[])
        if schema is Rewrites:
            return Rewrites(search_queries=["still nothing relevant here"])
        m = re.search(r"\[S1\].*?Clauses in this excerpt: ([^\n]+)\n<excerpt>\n(.*?)\n</excerpt>", user, re.S)
        good = Citation(source=1, clause=m.group(1).split(", ")[0], quote=" ".join(m.group(2).split()[1:10]))
        bad = Citation(source=2, clause="99.9", quote="text that is not in any excerpt")
        return Answer(abstain=False, claims=[Claim(text="grounded", citations=[good]),
                                             Claim(text="ungrounded", citations=[bad])])

    monkeypatch.setattr(P, "structured", fake)
    return P


def test_answer_path_drops_unverifiable_claims(stub):
    out = stub.ask("How early must the UN R152 collision warning be given before emergency braking?")
    path = [t["node"] for t in out["trace"]]
    assert path == ["analyze", "retrieve", "grade", "generate", "verify", "revise", "generate", "verify", "finalize"]
    assert [c["text"] for c in out["final"]["claims"]] == ["grounded"]
    assert out["final"]["dropped_claims"] == ["ungrounded"]
    assert out["final"]["references"][0]["citation"].startswith("UN R152 ")


def test_out_of_scope_abstains_without_retrieval(stub):
    out = stub.ask("What is a good recipe for chocolate cake?")
    assert out["abstained"] and [t["node"] for t in out["trace"]] == ["analyze", "abstain"]


def test_comparison_retrieves_both_regulations(stub):
    from copilot.graph.pipeline import retrieve_for
    from copilot.graph.refs import detect_refs

    q = "Compare the pole side impact test speeds in FMVSS 214 and UN R135."
    docs = {h.meta["doc_id"] for h in retrieve_for([q], detect_refs(q).where())}
    assert {"fmvss-214", "unece-r135"} <= docs
