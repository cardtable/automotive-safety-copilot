"""Fetch the regulation corpus from official publishers.

  FMVSS     -> eCFR versioner API, XML per section of 49 CFR Part 571
  UN/ECE    -> UN Official Document System, newest consolidated revision PDF
  Euro NCAP -> cdn.euroncap.com protocol PDFs

Downloads are idempotent: files already on disk are skipped. Run with --force to refresh.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

from copilot.config import get_settings, load_sources

UA = "Mozilla/5.0 (compatible; safety-copilot/0.1; research use)"
ECFR = "https://www.ecfr.gov/api/versioner/v1"
UNDOCS = "https://documents.un.org/api/symbol/access"


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"},
        follow_redirects=True,
        timeout=httpx.Timeout(120.0, connect=20.0),
    )


def _get(client: httpx.Client, url: str, params: dict | None = None, retries: int = 4) -> httpx.Response:
    for attempt in range(retries):
        try:
            r = client.get(url, params=params)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            return r
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def ecfr_latest_date(client: httpx.Client) -> str:
    titles = _get(client, f"{ECFR}/titles.json").json()["titles"]
    return next(t["up_to_date_as_of"] for t in titles if t["number"] == 49)


def download_fmvss(client: httpx.Client, cfg: dict, out: Path, force: bool) -> list[dict]:
    date = cfg.get("ecfr_date", "latest")
    if date == "latest":
        date = ecfr_latest_date(client)
    out.mkdir(parents=True, exist_ok=True)
    docs = []
    for sec in cfg["sections"]:
        path = out / f"571.{sec}.xml"
        if force or not path.exists():
            r = _get(client, f"{ECFR}/full/{date}/title-49.xml", params={"part": "571", "section": f"571.{sec}"})
            r.raise_for_status()
            path.write_bytes(r.content)
            print(f"  FMVSS 571.{sec}: {len(r.content) // 1024} KB")
        docs.append({
            "doc_id": f"fmvss-{sec}",
            "family": "FMVSS",
            "path": str(path),
            "source_url": f"https://www.ecfr.gov/on/{date}/title-49/section-571.{sec}",
            "version": f"eCFR as of {date}",
        })
    return docs


def _is_pdf(r: httpx.Response) -> bool:
    return r.status_code == 200 and r.content[:5] == b"%PDF-"


def resolve_unece_symbol(client: httpx.Client, reg: int, series: int) -> str | None:
    """Find the newest consolidated revision of a UN Regulation addendum.

    Regulation N is published as E/ECE/TRANS/505/Rev.S/Add.(N-1); consolidated
    texts append /Rev.K. Probe K upward until the document system stops answering.
    """
    base = f"E/ECE/TRANS/505/Rev.{series}/Add.{reg - 1}"
    best = None
    misses = 0
    k = 0
    # Early revisions predate the digital archive, so tolerate a longer gap before the first hit.
    while misses < (3 if best else 12) and k < 30:
        sym = base if k == 0 else f"{base}/Rev.{k}"
        r = client.get(UNDOCS, params={"s": sym, "l": "en", "t": "pdf"}, headers={"Range": "bytes=0-7"})
        if r.content[:5] == b"%PDF-":
            best, misses = sym, 0
        else:
            misses += 1
        k += 1
    return best


def download_unece(client: httpx.Client, regs: list[dict], out: Path, force: bool) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    cache_file = out / "_symbols.json"
    symbols = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    docs = []
    for spec in regs:
        reg = spec["reg"]
        path = out / f"R{reg:03d}.pdf"
        key = str(reg)
        if force or key not in symbols:
            symbols[key] = resolve_unece_symbol(client, reg, spec["series"])
            cache_file.write_text(json.dumps(symbols, indent=2))
        sym = symbols[key]
        if sym is None:
            print(f"  UN R{reg}: no document found, skipping")
            continue
        if force or not path.exists():
            r = _get(client, UNDOCS, params={"s": sym, "l": "en", "t": "pdf"})
            if not _is_pdf(r):
                print(f"  UN R{reg}: download failed ({r.status_code})")
                continue
            path.write_bytes(r.content)
            print(f"  UN R{reg}: {sym} {len(r.content) // 1024} KB")
        docs.append({
            "doc_id": f"unece-r{reg}",
            "family": "UN/ECE",
            "reg": reg,
            "title": f"UN Regulation No. {reg} - {spec['title']}",
            "path": str(path),
            "source_url": f"https://documents.un.org/api/symbol/access?s={sym}&l=en&t=pdf",
            "version": sym,
        })
    return docs


def download_euroncap(client: httpx.Client, items: list[dict], out: Path, force: bool) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    docs = []
    for item in items:
        path = out / f"{item['id']}.pdf"
        if force or not path.exists():
            r = _get(client, item["url"])
            if not _is_pdf(r):
                print(f"  {item['id']}: download failed ({r.status_code})")
                continue
            path.write_bytes(r.content)
            print(f"  {item['id']}: {len(r.content) // 1024} KB")
        docs.append({
            "doc_id": item["id"],
            "family": "Euro NCAP",
            "title": item["title"],
            "path": str(path),
            "source_url": item["url"],
            "version": item["title"].rsplit(" ", 1)[-1],
        })
    return docs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = ap.parse_args()

    s = get_settings()
    src = load_sources()
    docs: list[dict] = []
    with _client() as client:
        print("FMVSS (eCFR)")
        docs += download_fmvss(client, src["fmvss"], s.raw_dir / "fmvss", args.force)
        print("UN/ECE (UN ODS)")
        docs += download_unece(client, src["unece"], s.raw_dir / "unece", args.force)
        print("Euro NCAP")
        docs += download_euroncap(client, src["euroncap"], s.raw_dir / "euroncap", args.force)

    s.processed_dir.mkdir(parents=True, exist_ok=True)
    (s.raw_dir / "documents.json").write_text(json.dumps(docs, indent=2))
    print(f"\n{len(docs)} documents ready in {s.raw_dir}")


if __name__ == "__main__":
    main()
