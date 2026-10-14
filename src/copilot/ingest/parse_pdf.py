"""Layout-aware clause segmentation for UN/ECE regulations and Euro NCAP protocols.

Both families number their provisions hierarchically ("5.2.1."), but PDF text extraction gives
no structure, so a line that *starts* with a number is only a clause start when:

  * it sits at the page's left text margin (wrapped body text in UN documents is indented), and
  * the number is a plausible successor of the current clause (child, sibling or ancestor's
    sibling, allowing small gaps for deleted paragraphs) - this rejects "2.5 m ..." wraps,
    table cells and cross-references, and
  * the token after it is not a unit of measure.

Annex / Appendix headings reset the numbering context, so clause ids become e.g.
"Annex 3 §2.1". Tables of contents, running headers/footers and page numbers are dropped.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from copilot.ingest.models import Clause

NUM_ONLY = re.compile(r"^(\d{1,2}(?:\.\d{1,3}){0,6})\.?$")
NUM_TEXT = re.compile(r"^(\d{1,2}(?:\.\d{1,3}){0,6})\.?\s+(\S.*)$")
ANNEX = re.compile(r"^(Annex|Appendix)\s+(\d{1,2}[A-Z]?)(?:\s*[-–—:.]?\s*(.*))?$", re.IGNORECASE)
UNITS = re.compile(
    r"^(mm|cm|m|km|km/h|m/s|m/s2|m/s²|s|ms|kg|g|N|kN|Nm|Hz|kHz|%|°|deg|degrees?|per|x|×|V|A|bar|kPa|MPa|lux|lx|"
    r"times|and|or|to|-|–|±|\+|<|>|=|seconds?|minutes?|years?|months?|points?|pts)\b",
)
DOT_LEADER = re.compile(r"\.{5,}|…{2,}|(\.\s){5,}")
CONTENTS = re.compile(r"^\s*(table of )?contents\s*$", re.IGNORECASE)


@dataclass
class Line:
    text: str
    x0: float
    page: int
    size: float = 0.0
    bold: bool = False


def plausible_successor(prev: list[int], cand: list[int]) -> bool:
    """True if `cand` can follow `prev` in a hierarchical outline (gaps of up to 3 allowed)."""
    if not prev:
        return cand[0] <= 3 and all(x <= 1 for x in cand[1:])
    for level in range(len(cand)):
        if cand[:level] != prev[:level]:
            return False
        base = prev[level] if level < len(prev) else 0
        if base < cand[level] <= base + 3 and all(x <= 1 for x in cand[level + 1:]):
            return True
        if level >= len(prev):
            return False
    return False


def _page_lines(page: pymupdf.Page, pno: int) -> list[Line]:
    h = page.rect.height
    out = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        for ln in block.get("lines", []):
            text = "".join(sp["text"] for sp in ln["spans"]).strip()
            if not text:
                continue
            x0, y0, _, y1 = ln["bbox"]
            if y1 < h * 0.075 or y0 > h * 0.93:      # running header / footer band
                continue
            size = max(sp["size"] for sp in ln["spans"])
            spans = [sp for sp in ln["spans"] if sp["text"].strip()]
            bold = bool(spans) and all(sp["flags"] & 16 or "Bold" in sp["font"] for sp in spans)
            out.append(Line(text=re.sub(r"\s+", " ", text), x0=x0, page=pno + 1, size=size, bold=bold))
    return out


def _repeated_lines(pages: list[list[Line]]) -> set[str]:
    """Lines (digits normalised) that recur on many pages: headers, footers, symbols."""
    norm = lambda t: re.sub(r"\d+", "#", t)
    counts = Counter()
    for pl in pages:
        for t in {norm(l.text) for l in pl
                  if not NUM_ONLY.match(l.text) and not NUM_TEXT.match(l.text) and not ANNEX.match(l.text)}:
            counts[t] += 1
    n = max(len(pages), 1)
    return {t for t, c in counts.items() if c >= max(4, 0.3 * n) and len(t) < 90}


def _is_toc_page(lines: list[Line]) -> bool:
    if any(CONTENTS.match(l.text) for l in lines[:8]):
        return True
    leaders = sum(1 for l in lines if DOT_LEADER.search(l.text))
    return leaders >= 5


def _left_margin(lines: list[Line]) -> float:
    xs = sorted(l.x0 for l in lines)
    return xs[max(0, len(xs) // 20)] if xs else 0.0   # 5th percentile is robust to stray marks


def _body_x(lines: list[Line]) -> float:
    """Most common line start: the body-text column (UN documents hang numbers left of it)."""
    c = Counter(round(l.x0) for l in lines)
    return float(c.most_common(1)[0][0]) if c else 0.0


def parse_pdf(path: Path, doc: dict) -> list[Clause]:
    pdf = pymupdf.open(path)
    pages = [_page_lines(pdf[i], i) for i in range(pdf.page_count)]
    repeated = _repeated_lines(pages)
    # Body font size = the size carrying the most characters across the whole document.
    size_chars: Counter = Counter()
    for pl in pages:
        for l in pl:
            size_chars[round(l.size, 1)] += len(l.text)
    body_size = size_chars.most_common(1)[0][0] if size_chars else 10.0
    implicit_numbering = doc["family"] == "Euro NCAP"
    doc_title = doc["title"]

    clauses: list[Clause] = []
    context = ""                       # "" for the main body, else "Annex 3" / "Annex 3 Appendix 1"
    annex_title = ""
    prev: list[int] = []
    titles: dict[tuple[str, tuple[int, ...]], str] = {}
    cur: Clause | None = None
    pending_num: tuple[list[int], int] | None = None   # UN style: number alone on its line
    preamble: list[str] = []
    body_started = False

    def start(nums: list[int], title: str, text: str, page: int):
        nonlocal cur, prev, body_started
        if cur is not None and cur.text.strip():
            clauses.append(cur)
        number = ".".join(map(str, nums))
        label = f"{context} §{number}" if context else number
        parents = []
        if context:
            parents.append(f"{context} {annex_title}".strip())
        for k in range(1, len(nums)):
            key = (context, tuple(nums[:k]))
            if key in titles:
                parents.append(f"{'.'.join(map(str, nums[:k]))} {titles[key]}".strip())
        titles[(context, tuple(nums))] = title[:120]
        cur = Clause(
            doc_id=doc["doc_id"], family=doc["family"], doc_title=doc_title, number=label,
            title=title[:160], text=f"{number}. {text}".strip(), page_start=page, page_end=page,
            parents=parents, source_url=doc["source_url"], version=doc["version"],
        )
        prev = nums
        body_started = True

    def append(text: str, page: int):
        if cur is None:
            preamble.append(text)
        else:
            cur.text += ("\n" if cur.text else "") + text
            cur.page_end = page

    for plines in pages:
        # Dot leaders also appear in fill-in forms inside annexes, so only skip ToC pages up front.
        if not plines or (not body_started and _is_toc_page(plines)):
            continue
        margin, body_x = _left_margin(plines), _body_x(plines)
        for ln in plines:
            t = ln.text
            if re.sub(r"\d+", "#", t) in repeated or re.fullmatch(r"\d{1,3}", t) and pending_num is None:
                continue
            at_margin = ln.x0 <= margin + 8 or ln.x0 < body_x - 12

            if pending_num is not None:
                nums, page = pending_num
                pending_num = None
                start(nums, t if len(t) < 120 else "", t, page)
                continue

            am = ANNEX.match(t)
            is_heading = (
                am is not None and at_margin and len(t) < 140 and body_started
                and ln.size >= body_size + 1.0      # real annex titles are set large; running heads/footnotes small
                and not re.match(r"(to|of|in|and)\b", (am.group(3) or ""), re.IGNORECASE)
                and _annex_order_ok(context, am.group(1).title(), am.group(2))
            )
            if am and not is_heading and ln.size < body_size - 0.5 and len(t) < 40:
                continue                          # running head such as "Annex 8" in the page margin
            if is_heading:
                kind, ident, rest = am.group(1).title(), am.group(2), (am.group(3) or "").strip()
                if kind == "Annex":
                    context, annex_title = f"Annex {ident}", rest
                else:
                    base = context.split(" Appendix")[0] if context.startswith("Annex") else ""
                    context, annex_title = f"{base} Appendix {ident}".strip(), rest
                prev = []
                if cur is not None and cur.text.strip():
                    clauses.append(cur)
                cur = Clause(
                    doc_id=doc["doc_id"], family=doc["family"], doc_title=doc_title, number=context,
                    title=rest[:160], text=t, page_start=ln.page, page_end=ln.page,
                    source_url=doc["source_url"], version=doc["version"],
                )
                continue

            # UN footnotes ("13 New vehicle types ...") are set smaller than body text and never start
            # clauses. Euro NCAP sets its numbered sub-headings smaller than body text, so exempt it.
            if at_margin and (implicit_numbering or ln.size >= body_size - 0.6 or ln.bold):
                m = NUM_ONLY.match(t)
                if m:
                    nums = [int(x) for x in m.group(1).split(".")]
                    if plausible_successor(prev, nums):
                        pending_num = (nums, ln.page)
                        continue
                m = NUM_TEXT.match(t)
                if m:
                    nums = [int(x) for x in m.group(1).split(".")]
                    rest = m.group(2)
                    single_ok = len(nums) > 1 or (rest[:1].isupper() and len(rest) < 100)
                    if single_ok and not UNITS.match(rest) and plausible_successor(prev, nums):
                        heading = rest if len(rest) < 100 and not rest.endswith((",", ";")) else ""
                        start(nums, heading, rest, ln.page)
                        continue
            # Euro NCAP renders level-2 headings ("1.1 Reference System") with auto-numbering that is
            # lost in extraction: a bold heading set larger than body text but without a number.
            if (implicit_numbering and ln.bold and ln.x0 <= margin + 45 and prev and len(t) < 80
                    and body_size + 1.0 <= ln.size < 13.5 and not NUM_TEXT.match(t)):
                nums = [prev[0], prev[1] + 1 if len(prev) > 1 else 1]
                start(nums, t, t, ln.page)
                continue
            # Unnumbered bold sub-headings (common in Euro NCAP) are kept as "## " markers so the
            # chunker can split long clauses on them and carry the heading into each piece.
            if ln.bold and at_margin and len(t) < 90 and not t.endswith((".", ",", ";", ":")) and cur is not None:
                append(f"## {t}", ln.page)
                continue
            append(t, ln.page)

    if cur is not None and cur.text.strip():
        clauses.append(cur)
    if preamble:
        clauses.insert(0, Clause(
            doc_id=doc["doc_id"], family=doc["family"], doc_title=doc_title, number="Introduction",
            title="", text="\n".join(preamble), page_start=1, page_end=None,
            source_url=doc["source_url"], version=doc["version"],
        ))
    _dedupe_numbers(clauses)
    return clauses


def _annex_key(ident: str) -> tuple[int, str]:
    m = re.match(r"(\d+)([A-Z]?)", ident)
    return (int(m.group(1)), m.group(2)) if m else (0, "")


def _annex_order_ok(context: str, kind: str, ident: str) -> bool:
    """Annexes appear in ascending order; appendices may restart within each annex."""
    if kind == "Appendix":
        return True
    m = re.match(r"Annex (\S+)", context)
    return m is None or _annex_key(ident) > _annex_key(m.group(1))


def _dedupe_numbers(clauses: list[Clause]) -> None:
    """Rare restarts (e.g. a repeated annex number) would create duplicate ids; suffix them."""
    seen: Counter = Counter()
    for c in clauses:
        seen[c.number] += 1
        if seen[c.number] > 1:
            c.number = f"{c.number} ({seen[c.number]})"
