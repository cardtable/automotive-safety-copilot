"""Thin wrapper over the Anthropic SDK: typed structured outputs, prompt caching, refusal handling."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from copilot.config import get_settings

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMRefusal(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    seconds: float = 0.0
    by_step: dict = field(default_factory=dict)

    def add(self, step: str, u, seconds: float) -> None:
        self.calls += 1
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.cache_read_tokens += u.cache_read_input_tokens or 0
        self.cache_write_tokens += u.cache_creation_input_tokens or 0
        self.seconds += seconds
        self.by_step[step] = self.by_step.get(step, 0) + 1


USAGE = Usage()


@lru_cache
def client() -> anthropic.Anthropic:
    # Credentials resolve from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an `ant auth login` profile.
    return anthropic.Anthropic(max_retries=4, timeout=180.0)


def structured(
    step: str,
    system: str,
    user: str,
    schema: type[T],
    *,
    model: str | None = None,
    effort: str | None = None,
    max_tokens: int = 8000,
) -> T:
    """One call that returns a validated `schema` instance.

    The system prompt is stable across requests, so it is marked as a cache breakpoint; the
    per-question material (sources + question) goes in the user turn after it.
    """
    s = get_settings()
    t0 = time.time()
    resp = client().beta.messages.parse(
        model=model or s.llm_model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_format=schema,
        output_config={"effort": effort or s.llm_effort},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    USAGE.add(step, resp.usage, time.time() - t0)
    if resp.stop_reason == "refusal":
        raise LLMRefusal(f"{step}: model declined ({getattr(resp.stop_details, 'category', None)})")
    if resp.parsed_output is None:
        raise RuntimeError(f"{step}: no structured output (stop_reason={resp.stop_reason})")
    return resp.parsed_output
