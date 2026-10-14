"""Parse eCFR XML for one FMVSS section into S-numbered clauses.

eCFR emits each paragraph as <P>; a clause starts with a label such as "S4.1.5.1" (optionally
followed by an italic heading). Unlabelled paragraphs, "(a)" items, tables and notes belong to
the clause that precedes them.
"""

from __future__ import annotations

import re
from pathlib import Path

from lxml import etree

from copilot.ingest.models import Clause

LABEL = re.compile(r"^\s*(S\d+(?:\.\d+)*(?:\([a-z0-9]{1,4}\))*)\.?(?=\s|$|—|—)")
BLOCK_TAGS = {"P", "FP", "NOTE", "EXTRACT", "EXAMPLE", "TABLE", "GPOTABLE", "HD1", "HD2", "HD3", "MATH"}


def _text(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _table_text(el) -> str:
    rows = []
    cap = el.find(".//CAPTION")
    if cap is not None:
        rows.append(_text(cap))
    for tr in el.iter("TR"):
        cells = [_text(c) for c in tr if c.tag in ("TD", "TH")]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _heading(el) -> str:
    """The italic run right after the label, e.g. <P>S4.1 <I>Passenger cars.</I></P>."""
    i = el.find("I")
    if i is None:
        return ""
    lead = (el.text or "").strip()
    if not LABEL.match(lead):
        return ""
    return _text(i).rstrip(".")


def parse_fmvss(path: Path, doc: dict) -> list[Clause]:
    root = etree.parse(str(path)).getroot()
    head = _text(root.find("HEAD"))                       # "§ 571.208 Standard No. 208; Occupant crash protection."
    std = doc["doc_id"].split("-", 1)[1]
    m = re.search(r"Standard No\.\s*\S+?[;,]\s*([^;]+)", head)
    if m:
        name = m.group(1)
    else:  # e.g. "§ 571.213 Child restraint systems; Applicable unless ..."
        name = re.sub(r"^§\s*571\.\S+\s*", "", head).split(";")[0]
    name = name.strip().rstrip(".")
    doc_title = f"FMVSS No. {std} ({name[0].upper() + name[1:]})"

    clauses: list[Clause] = []
    stack: list[tuple[str, str]] = []                     # (label, heading) ancestors
    preamble: list[str] = []
    tables: list[Clause] = []
    cur: Clause | None = None

    def flush():
        if cur is not None and cur.text.strip():
            clauses.append(cur)

    for el in root.iter():
        tag = el.tag if isinstance(el.tag, str) else ""
        if tag not in BLOCK_TAGS:
            continue
        # Skip blocks nested inside another block we already consume whole.
        if any(a.tag in ("TABLE", "GPOTABLE", "EXTRACT", "NOTE") for a in el.iterancestors()):
            continue
        text = _table_text(el) if tag in ("TABLE", "GPOTABLE") else _text(el)
        if not text:
            continue
        # Captioned tables ("Table I-a—Required Lamps ...") are cited by caption, so they become their
        # own clauses without disturbing the paragraph the surrounding text belongs to.
        cap = re.match(r"^(Table [A-Z0-9IVX]+(?:-[a-z0-9]+)?)\s*[—\-:.]?\s*([^\n|]*)", text)
        if tag in ("TABLE", "GPOTABLE") and cap:
            tables.append(Clause(
                doc_id=doc["doc_id"], family="FMVSS", doc_title=doc_title, number=cap.group(1),
                title=cap.group(2).strip()[:120], text=text,
                parents=[f"{lab} {t}".strip() for lab, t in stack],
                source_url=doc["source_url"], version=doc["version"],
            ))
            continue
        m = LABEL.match(text) if tag in ("P", "FP") else None
        if m:
            flush()
            label = m.group(1)
            title = _heading(el)
            while stack and not label.startswith(stack[-1][0] + ".") and not label.startswith(stack[-1][0] + "("):
                stack.pop()
            parents = [f"{lab} {t}".strip() for lab, t in stack]
            cur = Clause(
                doc_id=doc["doc_id"], family="FMVSS", doc_title=doc_title, number=label, title=title,
                text=text, parents=parents, source_url=doc["source_url"], version=doc["version"],
            )
            stack.append((label, title))
        elif cur is None:
            preamble.append(text)
        else:
            cur.text += "\n" + text
    flush()
    clauses.extend(tables)
    if preamble:
        clauses.insert(0, Clause(
            doc_id=doc["doc_id"], family="FMVSS", doc_title=doc_title, number="Preamble", title="",
            text="\n".join(preamble), source_url=doc["source_url"], version=doc["version"],
        ))
    return clauses
