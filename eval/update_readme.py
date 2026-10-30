"""Regenerate the Results section of README.md from eval/results/*.json.

    python eval/update_readme.py
"""

import json
import re

from common import RESULTS, ROOT

README = ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
SYSTEMS = [("v1-ollama", "v1 pipeline<br>(local qwen3.5:4b)"),
           ("v2-ollama", "Echo-Sight offline<br>(local qwen3.5:4b)"),
           ("v2-groq", "Echo-Sight default<br>(Groq gpt-oss-120b)")]
PENDING = "pending"


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def pct(value):
    return f"{value:.1%}" if isinstance(value, float) and value == value else "—"


def secs(value):
    return f"{value:.1f} s" if isinstance(value, float) and value == value else "—"


def retrieval_section(rows) -> list[str]:
    v1, final = rows[0], next(r for r in rows if r["config"].endswith("(final)"))
    lines = [f"**Retrieval**: {final['n']} answerable test questions, no LLM involved.", "",
             "| Metric | v1 | Echo-Sight |", "|---|---|---|"]
    for key, label in (("hit@1", "Top-ranked passage contains the answer (Hit@1)"),
                       ("hit@5", "Answer in the top 5 passages (Hit@5)"),
                       ("recall@4", "Required evidence among the 4 passages the LLM sees")):
        lines.append(f"| {label} | {v1[key]:.1%} | **{final[key]:.1%}** |")
    lines.append(f"| Mean reciprocal rank (MRR@10) | {v1['mrr']:.3f} | **{final['mrr']:.3f}** |")
    return lines


def answers_section(rows) -> list[str]:
    by_run = {r["run"]: r for r in rows or []}
    present = [(run, label) for run, label in SYSTEMS if run in by_run]
    if not present:
        return ["**Answers**: run `bash eval/run_all.sh` to generate the end-to-end results."]
    sample = by_run[present[0][0]]
    judges = sorted({r["judge"] for r in by_run.values() if r.get("judge")})
    lines = [f"**Answers**: {sample['n']} test questions ({sample['n_answerable']} answerable, "
             f"{sample['n_out_of_scope']} out of scope).", "",
             "| Metric | " + " | ".join(label for _, label in present) + " |",
             "|---|" + "---|" * len(present)]

    def row(label, key, fmt, bold_best=True):
        values = [by_run[run].get(key) for run, _ in present]
        cells = [fmt(v) if v is not None else (PENDING if key in ("overall", "correct") else "—") for v in values]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row("Overall accuracy (correct answer or correct refusal)", "overall", pct)
    row("Answer accuracy (fully correct, LLM-judged)", "correct", pct)
    row("Partly or fully correct", "lenient", pct)
    row("Faithful to the retrieved passages (LLM-judged)", "faithful", pct)
    row("Citations point to the evidence", "citation_hit", pct)
    row("Out-of-scope questions declined", "correct_refusal", pct)
    row("Answerable questions wrongly declined", "false_refusal", pct)
    row("Required evidence in the LLM's context", "evidence_in_context", pct)
    row("Tokens per question (prompt + answer)", "tokens", lambda v: f"{v:,.0f}")
    row("First spoken sentence, median", "first_sentence_p50_s", secs)
    row("Full answer, median", "p50_s", secs)
    row("Full answer, 95th percentile", "p95_s", secs)
    note = "Judge: " + ", ".join(judges) + "." if judges else "Judging pending: needs a Groq key."
    lines += ["", f"{note} Citation accuracy is computed without a judge. The v1 pipeline does not cite."]
    return lines


def listening_section(listening, e2e) -> list[str]:
    if not listening:
        return []
    lin, head = listening["Linear listening_180"], listening["Heading navigation_360"]
    lines = ["**Time until a screen-reader user hears the answer** (median over test questions)", "",
             "| Approach | Median | 90th percentile |", "|---|---|---|",
             f"| Listen from the top of the PDF at 180 words/min | {lin['median_s'] / 60:.0f} min | "
             f"{lin['p90_s'] / 60:.0f} min |",
             f"| Expert heading navigation at 360 words/min | {head['median_s']:.0f} s | {head['p90_s']:.0f} s |"]
    for run, label in SYSTEMS[1:]:
        row = next((r for r in e2e or [] if r["run"] == run), None)
        if row:
            lines.append(f"| {label.replace('<br>', ' ')}: first spoken sentence | "
                         f"{row['first_sentence_p50_s']:.1f} s | — |")
    return lines


def ablation_table(rows) -> list[str]:
    lines = ["| Configuration | Hit@1 | Hit@5 | MRR@10 | Evidence recall@4 | Latency p50 |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        name = r["config"].replace("v1 baseline (regex sections, 500/80 chunks, MiniLM)",
                                   "**v1 baseline** (regex sections, 500/80 chunks, MiniLM, dense)")
        cells = [f"{r[k]:.1%}" for k in ("hit@1", "hit@5")] + [f"{r['mrr']:.3f}", f"{r['recall@4']:.1%}"]
        if r["config"].endswith("(final)"):
            name, cells = f"**{name}**", [f"**{c}**" for c in cells]
        latency = f"{r['p50_ms']:.0f} ms" if r["p50_ms"] >= 1 else "<1 ms"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {latency} |")
    return lines


def stair_table(stair) -> list[str]:
    names = {"dense (bge-base)": "Dense (bge-base)", "v2 final (hybrid + reranker)": "v2 (hybrid + reranker)"}
    lines = ["| Method | Recall@1 | Recall@3 |", "|---|---|---|"]
    best = max(r["R@1"] for r in stair["sections"])
    for r in stair["sections"]:
        name = "STAIR ToC router, zero-shot gpt-oss-20b" if r["method"].startswith("STAIR") else names.get(
            r["method"], r["method"].replace("hybrid (RRF)", "Hybrid (RRF)"))
        cells = [f"{r['R@1']:.1%}", f"{r['R@3']:.1%}"]
        if r["R@1"] == best:
            cells = [f"**{c}**" for c in cells]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def replace_block(text: str, tag: str, lines: list[str]) -> str:
    start, end = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    if start not in text or not lines:
        return text
    block = start + "\n" + "\n".join(lines) + "\n" + end
    return re.sub(re.escape(start) + ".*?" + re.escape(end), lambda _: block, text, flags=re.S)


def main():
    retrieval, e2e, listening = load("retrieval_test.json"), load("e2e_test.json"), load("listening_test.json")
    parts = [retrieval_section(retrieval), answers_section(e2e), listening_section(listening, e2e)]
    body = "\n\n".join("\n".join(p) for p in parts if p)
    text = README.read_text()
    block = f"{START}\n{body}\n{END}"
    if START in text:
        text = re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    else:
        text = text.replace("<!-- RESULTS -->", block)
    text = replace_block(text, "ABLATION", ablation_table(retrieval) if retrieval else [])
    stair = load("stair_test_gpt-oss-20b.json")
    text = replace_block(text, "STAIR", stair_table(stair) if stair else [])
    README.write_text(text)
    print(body)


if __name__ == "__main__":
    main()
