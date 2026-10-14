from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Clause:
    """One numbered provision of a regulation, the unit of citation."""

    doc_id: str
    family: str            # FMVSS | UN/ECE | Euro NCAP
    doc_title: str
    number: str            # "S4.1.5.1", "5.2.1", "Annex 3 §2.1"
    title: str             # heading text when the clause has one
    text: str
    page_start: int | None = None
    page_end: int | None = None
    parents: list[str] = field(default_factory=list)   # breadcrumb of ancestor "number title"
    source_url: str = ""
    version: str = ""

    @property
    def citation(self) -> str:
        return f"{short_doc_name(self)} {self.number}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citation"] = self.citation
        return d


def short_doc_name(c: Clause | dict) -> str:
    doc_id = c.doc_id if isinstance(c, Clause) else c["doc_id"]
    if doc_id.startswith("fmvss-"):
        return f"FMVSS {doc_id.split('-', 1)[1]}"
    if doc_id.startswith("unece-r"):
        return f"UN R{doc_id.removeprefix('unece-r')}"
    title = c.doc_title if isinstance(c, Clause) else c["doc_title"]
    return title.replace("Euro NCAP Protocol - ", "Euro NCAP ").rsplit(" v", 1)[0]
