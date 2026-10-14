from __future__ import annotations

from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    in_scope: bool = Field(description="True if the question is about vehicle safety regulation / testing / ratings")
    search_queries: list[str] = Field(description="1-3 search queries, most important first")


class Rewrites(BaseModel):
    search_queries: list[str]


class Citation(BaseModel):
    source: int = Field(description="Excerpt number n from [Sn]")
    clause: str = Field(description="Clause number exactly as listed for that excerpt")
    quote: str = Field(description="Verbatim span copied from the excerpt that supports the claim")


class Claim(BaseModel):
    text: str
    citations: list[Citation]


class Answer(BaseModel):
    abstain: bool
    abstain_reason: str = ""
    claims: list[Claim] = Field(default_factory=list)
