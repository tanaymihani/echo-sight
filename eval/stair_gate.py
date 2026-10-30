"""Derive the "v3" runs (v2 + STAIR scope gate) from cached v2 answers, verdicts and ToC routes.

The gate runs before generation and changes nothing else, so its end-to-end effect is exact without
new LLM calls: a gated question becomes a refusal (judged as declined), a low-confidence question that
passes the gate waits for the router, and every other answer is unchanged.

    python eval/stair_gate.py          # needs eval/results/stair_cache.json from stair_eval.py
"""

import json

from common import RESULTS, load_questions

from echosight.config import RetrievalConfig
from echosight.prompts import REFUSAL

E2E = RESULTS / "e2e"
ROUTER = "groq/openai/gpt-oss-20b"


def load(path):
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}


def main():
    threshold = RetrievalConfig().toc_gate_score
    cache = json.loads((RESULTS / "stair_cache.json").read_text())
    questions = {q["id"]: q for q in load_questions()}
    for source in ("v2-groq", "v2-ollama"):
        answers, verdicts = load(E2E / f"{source}.jsonl"), load(E2E / f"{source}.judge.jsonl")
        if not answers:
            continue
        target = source.replace("v2", "v3")
        rows, judged, gated = [], [], {"answerable": 0, "out_of_scope": 0}
        for qid, row in answers.items():
            route = cache.get(f"{ROUTER}|3|{questions[qid]['question']}")
            if route is None:
                raise SystemExit(f"No cached route for {qid}: run eval/stair_eval.py on its split first")
            row = json.loads(json.dumps(row))
            t = row["timings"]
            low = (row.get("top_score") or 0.0) < threshold
            wait = max(0.0, route["latency_s"] - t["retrieval_s"]) if low else 0.0
            verdict = dict(verdicts.get(qid, {}))
            if low and not route["sections"]:
                gated["out_of_scope" if questions[qid]["type"] == "out_of_scope" else "answerable"] += 1
                total = t["retrieval_s"] + wait
                row.update(text=REFUSAL, refused=True, gated=True, cited=[], tokens=0)
                row["timings"] = {"retrieval_s": t["retrieval_s"], "toc_wait_s": wait, "ttft_s": None,
                                  "llm_s": 0.0, "total_s": total, "first_sentence_s": total}
                verdict = {"id": qid, "judge": verdict.get("judge"), "declined": True, "correct": 0}
            elif wait:
                for key in ("ttft_s", "total_s", "first_sentence_s"):
                    if t.get(key) is not None:
                        t[key] += wait
                t["toc_wait_s"] = wait
            row["derived_from"] = source
            rows.append(row)
            if verdict:
                judged.append(verdict)
        (E2E / f"{target}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        if len(judged) == len(rows):
            (E2E / f"{target}.judge.jsonl").write_text("".join(json.dumps(v) + "\n" for v in judged))
        print(f"{target}: gated {gated['out_of_scope']} out-of-scope and {gated['answerable']} answerable "
              f"of {len(rows)} questions; verdicts {'copied' if len(judged) == len(rows) else 'pending'}")


if __name__ == "__main__":
    main()
