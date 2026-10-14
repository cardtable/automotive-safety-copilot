"""Unit tests for the deterministic parts: clause segmentation, chunking, reference detection,
citation verification and retrieval metrics. No network or LLM calls."""

from copilot.graph.refs import FMVSS, UNECE
from copilot.graph.schemas import Answer, Citation, Claim
from copilot.graph.verify import clause_matches, norm_text, quote_score, verify_answer
from copilot.ingest.chunk import chunk_document
from copilot.ingest.models import Clause
from copilot.ingest.parse_pdf import plausible_successor
from eval.run_eval import retrieval_metrics

# ---------------------------------------------------------------- outline numbering

def test_successor_accepts_children_siblings_and_ancestor_siblings():
    assert plausible_successor([], [1])
    assert plausible_successor([5, 2], [5, 2, 1])          # first child
    assert plausible_successor([5, 2, 1], [5, 2, 2])       # sibling
    assert plausible_successor([5, 2, 1, 4], [5, 3])       # ancestor's sibling
    assert plausible_successor([5, 2], [5, 4])             # small gap (deleted paragraph)
    assert plausible_successor([1], [1, 1, 1])             # NCAP: level-2 number lost in extraction


def test_successor_rejects_wraps_and_cross_references():
    assert not plausible_successor([5, 2, 1], [2, 5])      # "2.5 m" at the start of a wrapped line
    assert not plausible_successor([5, 2], [5, 2])         # repeated number
    assert not plausible_successor([5, 2], [9])            # big jump
    assert not plausible_successor([5, 2], [5, 2, 5])      # child must start near 1


# ---------------------------------------------------------------- chunking

def _clause(num, text, doc="unece-r152"):
    return Clause(doc_id=doc, family="UN/ECE", doc_title="UN Regulation No. 152", number=num, title="",
                  text=text, page_start=1, page_end=1)


def test_chunks_pack_small_clauses_and_respect_sections():
    cl = [_clause("2.1", "2.1. " + "a" * 300), _clause("2.2", "2.2. " + "b" * 300), _clause("3", "3. Approval"),
          _clause("3.1", "3.1. " + "c" * 300)]
    chunks = chunk_document(cl, max_chars=2400, target_chars=1400)
    assert [c.clauses for c in chunks] == [["2.1", "2.2"], ["3", "3.1"]]
    assert chunks[0].label == "UN R152 2.1–2.2"


def test_long_clause_is_split_but_keeps_its_number():
    body = "5.1. Intro\n" + "\n".join(f"({chr(97 + i)}) " + "x" * 900 for i in range(5))
    chunks = chunk_document([_clause("5.1", body)], max_chars=2400)
    assert len(chunks) > 1
    assert all(c.clauses == ["5.1"] for c in chunks)
    assert chunks[1].text.startswith("[5.1, continued]")


# ---------------------------------------------------------------- reference detection

def test_reference_regexes():
    assert [m.group(1) for m in FMVSS.finditer("Compare FMVSS 208 and FMVSS No. 216a")] == ["208", "216a"]
    assert {next(g for g in m.groups() if g) for m in UNECE.finditer("UN R152, ECE R94 and Regulation No. 16")} \
        == {"152", "94", "16"}


# ---------------------------------------------------------------- citation verification

EXCERPT = ("5.2.1.1. Collision warning\nWhen a collision ... is imminent, a collision warning shall be "
           "provided as specified in paragraph 5.5.1., and shall be triggered at the latest 0.8 seconds\n"
           "before the start of emergency braking.")
SOURCES = [{"text": EXCERPT, "clauses": ["5.2.1.1", "5.2.1.2"]}]


def test_clause_matching_normalises_formats():
    assert clause_matches("5.2.1.1.", ["5.2.1.1"])
    assert clause_matches("paragraph 5.2.1.1", ["5.2.1.1"])
    assert clause_matches("S5.2(a)", ["S5.2"])
    assert clause_matches("Annex 3 §2.1", ["Annex 3 §2.1"])
    assert not clause_matches("5.2.1", ["5.2.1.1"])


def test_quote_matching_tolerates_line_breaks_but_not_paraphrase():
    assert quote_score("triggered at the latest 0.8 seconds before the start of emergency braking", EXCERPT) == 1.0
    assert quote_score("triggered at least 2 seconds before braking begins", EXCERPT) < 0.92
    assert norm_text("Tested-\nDevice") == "testeddevice"


def test_verify_answer_flags_bad_citations():
    ans = Answer(abstain=False, claims=[
        Claim(text="Warning at least 0.8 s before braking.", citations=[
            Citation(source=1, clause="5.2.1.1", quote="triggered at the latest 0.8 seconds")]),
        Claim(text="Made-up claim.", citations=[Citation(source=1, clause="9.9", quote="not there at all")]),
        Claim(text="Bad source.", citations=[Citation(source=4, clause="5.2.1.1", quote="collision warning")]),
    ])
    rep = verify_answer(ans, SOURCES)
    assert [c["grounded"] for c in rep["claims"]] == [True, False, False]
    assert rep["grounded_citations"] == 1 and rep["n_citations"] == 3


# ---------------------------------------------------------------- metrics

def test_retrieval_metrics():
    gold = [{"doc_id": "fmvss-214", "clause": "S9.1.2"}, {"doc_id": "unece-r135", "clause": "Annex 3 §9.6"}]
    hits = [
        {"doc_id": "fmvss-214", "clauses": ["S10.1"]},
        {"doc_id": "unece-r135", "clauses": ["Annex 3 §9.5", "Annex 3 §9.6"]},
        {"doc_id": "fmvss-214", "clauses": ["S9.1.1", "S9.1.2"]},
    ]
    m = retrieval_metrics(hits, gold)
    assert m["mrr"] == 0.5
    assert m["hit@1"] == 0.0 and m["recall@3"] == 1.0
    assert abs(m["gold_precision@3"] - 2 / 3) < 1e-9
