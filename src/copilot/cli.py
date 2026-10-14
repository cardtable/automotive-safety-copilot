"""Ask the copilot from the terminal:  copilot-ask "How early must the R152 collision warning be given?" """

from __future__ import annotations

import argparse
import json
import textwrap

from copilot.graph.pipeline import ask


def render(state: dict) -> str:
    f = state["final"]
    if state.get("abstained"):
        out = f"ABSTAINED: {f['abstain_reason']}"
        if f.get("closest"):
            out += "\nClosest retrieved: " + ", ".join(f"{c['label']} ({c['rerank']:.2f})" for c in f["closest"])
        return out
    lines = []
    for c in f["claims"]:
        marks = "".join(f"[{n}]" for n in c["refs"])
        lines.append(textwrap.fill(f"{c['text']} {marks}", 100))
    lines.append("\nSources")
    for r in f["references"]:
        page = f", p. {r['page']}" if r["page"] else ""
        lines.append(f"  [{r['n']}] {r['citation']} — {r['version']}{page}")
        lines.append(textwrap.indent(textwrap.fill(f"“{r['quote']}”", 96), "      "))
    if f.get("dropped_claims"):
        lines.append(f"\n({len(f['dropped_claims'])} unverifiable statement(s) removed)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--json", action="store_true", help="print the full final state")
    ap.add_argument("--trace", action="store_true", help="print the graph trace")
    a = ap.parse_args()
    state = ask(a.question)
    if a.json:
        print(json.dumps({k: v for k, v in state.items() if k != "hits"}, indent=2, default=str))
        return
    print(render(state))
    if a.trace:
        print("\nTrace")
        for t in state["trace"]:
            print("  ", {k: v for k, v in t.items() if k != "t"})


if __name__ == "__main__":
    main()
