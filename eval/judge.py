"""LLM-as-judge graders. Each judge sees only what it needs, and returns per-item verdicts so the
scores can be audited question by question."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from copilot.config import get_settings
from copilot.llm import structured

# ---------------------------------------------------------------- faithfulness

FAITH_SYSTEM = """You audit answers produced by a regulation question-answering system. For each numbered \
claim you get the regulation excerpts it cites. Decide whether the cited excerpts fully support the claim.

- supported: every fact in the claim (numbers, units, conditions, applicability, modal verbs such as shall/may) \
is stated in, or follows directly from, the cited excerpts.
- partial: the main point is supported but some detail (a number, condition, scope, or qualifier) is not in \
the excerpts or is overstated.
- unsupported: the excerpts do not support the claim, or contradict it.

Judge only against the excerpts shown, not against your own knowledge of the regulations. A claim that says \
what the excerpts do not cover is supported if the excerpts indeed do not cover it."""


class ClaimVerdict(BaseModel):
    claim: int
    verdict: Literal["supported", "partial", "unsupported"]
    reason: str


class FaithReport(BaseModel):
    verdicts: list[ClaimVerdict]


def judge_faithfulness(question: str, claims: list[dict], references: list[dict]) -> FaithReport:
    ref_by_n = {r["n"]: r for r in references}
    blocks = []
    for i, c in enumerate(claims, 1):
        cited = "\n".join(
            f"  [{n}] {ref_by_n[n]['citation']}:\n  <excerpt>{ref_by_n[n]['excerpt']}</excerpt>" for n in c["refs"]
        )
        blocks.append(f"Claim {i}: {c['text']}\nCited excerpts:\n{cited}")
    user = f"Question: {question}\n\n" + "\n\n".join(blocks)
    return structured("judge_faithfulness", FAITH_SYSTEM, user, FaithReport,
                      model=get_settings().judge_model, effort="medium", max_tokens=8000)


# ---------------------------------------------------------------- answer correctness

CORRECT_SYSTEM = """You grade an answer against reference facts written by a regulations expert. For each \
reference fact, decide whether the answer states it (same value, unit and condition; wording may differ). \
Mark a fact as present only if the answer actually states it; do not give credit for facts the answer merely \
implies or for contradicting values."""


class FactVerdict(BaseModel):
    fact: int
    present: bool


class CorrectReport(BaseModel):
    facts: list[FactVerdict]


def judge_correctness(question: str, answer_text: str, facts: list[str]) -> CorrectReport:
    fl = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
    user = f"Question: {question}\n\nAnswer:\n{answer_text}\n\nReference facts:\n{fl}"
    return structured("judge_correctness", CORRECT_SYSTEM, user, CorrectReport,
                      model=get_settings().judge_model, effort="low", max_tokens=3000)


# ---------------------------------------------------------------- context precision

CTX_SYSTEM = """You assess a retrieval system for vehicle-safety regulations. For each retrieved excerpt, \
decide whether it is relevant: it contains a provision that is needed, or directly useful, to answer the \
question (the governing requirement, its definition, its test condition, or its applicability). Excerpts on a \
neighbouring topic that would not be cited in a good answer are not relevant."""


class CtxVerdict(BaseModel):
    excerpt: int
    relevant: bool


class CtxReport(BaseModel):
    verdicts: list[CtxVerdict]


def judge_context(question: str, hits: list[dict]) -> CtxReport:
    blocks = "\n\n".join(
        f"[{i}] {h['label']} ({h['context'][:160]})\n<excerpt>{h['text'][:2500]}</excerpt>"
        for i, h in enumerate(hits, 1)
    )
    user = f"Question: {question}\n\n{blocks}"
    return structured("judge_context", CTX_SYSTEM, user, CtxReport,
                      model=get_settings().judge_model, effort="low", max_tokens=3000)
