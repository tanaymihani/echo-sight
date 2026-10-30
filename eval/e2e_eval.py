"""End-to-end evaluation: answer correctness, refusals, citations and latency.

    python eval/e2e_eval.py generate --system v2 --provider groq      # answers are cached
    python eval/e2e_eval.py generate --system v1 --provider ollama
    python eval/e2e_eval.py judge --run v2-groq --faithfulness        # LLM-as-judge, cached
    python eval/e2e_eval.py report                                     # summary table

Runs are stored in eval/results/e2e/<run>.jsonl, judgements next to them.
"""

import argparse
import hashlib
import json
import re
import time
from collections import deque

from common import RESULTS, covers, is_relevant, load_questions, markdown_table, mean, percentile
from openai import RateLimitError

from echosight.config import LLMConfig
from echosight.llm import LLM
from echosight.rag import EchoSight, is_refusal
from echosight.retrieval import Retriever
from echosight.voice import SentenceStream
from eval import v1_baseline

E2E = RESULTS / "e2e"
# A different model family from the answering model (gpt-oss), to avoid self-preference.
JUDGE_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]

JUDGE_PROMPT = """You are grading an assistant that answers questions about Northeastern University's \
Code of Student Conduct.

Question: {question}
Reference answer, written from the Code: {reference}
{passages}Assistant's answer: {answer}

Reply with a JSON object with these keys:
"declined": true if the assistant said it could not find or could not answer the question, false if it gave an answer.
"correct": 2 if the answer agrees with the reference and covers its main point, 1 if it is partly right or misses an \
important detail, 0 if it is wrong, contradicts the reference, or declined. Extra correct detail is fine.
{faithful_key}"reason": one short sentence explaining the grade."""

FAITHFUL_KEY = ('"faithful": true if every claim in the answer is supported by the passages above, '
                'false if it states anything the passages do not support.\n')
NO_ANSWER = "None. The Code of Student Conduct does not answer this question, so the right response is to say so."


class Throttle:
    """Stays under a tokens-per-minute budget using the actual token counts of recent calls."""

    def __init__(self, tpm: int | None):
        self.tpm, self.calls = tpm, deque()

    def wait(self, expected: int) -> None:
        while self.tpm:
            now = time.time()
            while self.calls and now - self.calls[0][0] > 60:
                self.calls.popleft()
            if sum(t for _, t in self.calls) + expected <= self.tpm:
                return
            time.sleep(1)

    def record(self, tokens: int) -> None:
        self.calls.append((time.time(), tokens))


def run_name(system: str, provider: str, model: str | None) -> str:
    name = f"{system}-{provider}"
    return name if not model else f"{name}-{model.split('/')[-1]}"


def load_jsonl(path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {row["id"]: row for row in rows}


def append_jsonl(path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_sentence_timer(start: float):
    state = {"first": None}

    def emit(_sentence):
        if state["first"] is None:
            state["first"] = time.perf_counter() - start

    return SentenceStream(emit), state


def generate(args) -> None:
    llm = LLM(LLMConfig.from_env(args.provider, args.model))
    name = run_name(args.system, args.provider, args.model)
    path = E2E / f"{name}.jsonl"
    done = load_jsonl(path)
    items = [q for q in load_questions(args.split) if q["id"] not in done][: args.limit]
    throttle = Throttle(args.tpm if args.provider != "ollama" else None)
    print(f"{name}: {llm.name}, {len(items)} questions to answer ({len(done)} cached)")

    if args.system == "v2":
        assistant = EchoSight(Retriever(), llm)
    else:
        v1 = v1_baseline.V1Retriever()
    for i, item in enumerate(items, 1):
        throttle.wait(expected=1500)
        start = time.perf_counter()
        stream, first = first_sentence_timer(start)
        if args.system == "v2":
            answer = assistant.ask(item["question"], on_text=stream.feed)
            row = {"text": answer.text, "refused": answer.refused, "gated": answer.gated,
                   "passages": [h.chunk.text for h in answer.hits],
                   "sections": [h.chunk.breadcrumb for h in answer.hits],
                   "top_score": answer.hits[0].score if answer.hits else None,
                   "cited": [answer.hits.index(h) for h in answer.cited],
                   "timings": answer.timings, "tokens": answer.tokens}
        else:
            docs = v1.search(item["question"], 8)
            retrieval_s = time.perf_counter() - start
            completion = llm.chat(v1_baseline.messages(item["question"], docs), on_text=stream.feed)
            row = {"text": completion.text, "refused": None, "gated": False,
                   "passages": [d["text"] for d in docs], "sections": [d["section"] for d in docs],
                   "timings": {"retrieval_s": retrieval_s, "ttft_s": completion.ttft_s,
                               "llm_s": completion.total_s, "total_s": time.perf_counter() - start},
                   "tokens": completion.prompt_tokens + completion.completion_tokens}
        stream.flush()
        row["timings"]["first_sentence_s"] = first["first"]
        throttle.record(row["tokens"])
        append_jsonl(path, {"id": item["id"], "model": llm.name, **row})
        print(f"  [{i}/{len(items)}] {item['id']} {row['timings']['total_s']:.1f}s "
              f"{'REFUSED ' if row['refused'] else ''}{row['text'][:90]!r}", flush=True)


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0)) if match else {}


def pick_judge(provider: str) -> str:
    """First preferred judge model the provider currently serves (preview models come and go)."""
    try:
        llm = LLM(LLMConfig.from_env(provider, JUDGE_MODELS[0]))
        available = {m.id for m in llm.client.models.list()}
    except Exception:
        return JUDGE_MODELS[0]
    return next((m for m in JUDGE_MODELS if m in available), JUDGE_MODELS[0])


def judge(args) -> None:
    provider = args.judge_provider
    llm = LLM(LLMConfig.from_env(provider, args.judge_model or pick_judge(provider)))
    answers = load_jsonl(E2E / f"{args.run}.jsonl")
    path = E2E / f"{args.run}.judge.jsonl"
    done = load_jsonl(path)
    questions = {q["id"]: q for q in load_questions()}
    throttle = Throttle(args.tpm)
    todo = [a for a in answers.values() if a["id"] not in done]
    print(f"Judging {len(todo)} answers from {args.run} with {llm.name}")
    for i, row in enumerate(todo, 1):
        item = questions[row["id"]]
        passages = ""
        if args.faithfulness:
            passages = "Passages the assistant was given:\n" + "\n\n".join(
                f"[{n}] {p}" for n, p in enumerate(row["passages"], 1)) + "\n\n"
        prompt = JUDGE_PROMPT.format(
            question=item["question"], reference=item.get("answer") or NO_ANSWER, passages=passages,
            answer=row["text"], faithful_key=FAITHFUL_KEY if args.faithfulness else "")
        throttle.wait(expected=len(prompt) // 3)
        verdict, tokens = {}, 0
        for attempt in range(3):     # retry malformed verdicts, with some randomness after the first try
            completion = llm.chat([{"role": "user", "content": prompt}], max_tokens=600,
                                  temperature=0 if attempt == 0 else 0.5, json_mode=True)
            tokens += completion.prompt_tokens + completion.completion_tokens
            try:
                verdict = parse_json(completion.text)
            except json.JSONDecodeError:
                verdict = {}
            if "declined" in verdict and "correct" in verdict:
                break
        throttle.record(tokens)
        key = hashlib.sha1(prompt.encode()).hexdigest()[:12]
        append_jsonl(path, {"id": row["id"], "judge": llm.name, "prompt_hash": key, **verdict})
        print(f"  [{i}/{len(todo)}] {row['id']} declined={verdict.get('declined')} "
              f"correct={verdict.get('correct')} faithful={verdict.get('faithful')}", flush=True)


def summarize(run: str, split: str) -> dict | None:
    answers = load_jsonl(E2E / f"{run}.jsonl")
    verdicts = load_jsonl(E2E / f"{run}.judge.jsonl")
    items = [q for q in load_questions(split) if q["id"] in answers]
    if not items:
        return None
    answerable = [q for q in items if q["type"] != "out_of_scope"]
    unanswerable = [q for q in items if q["type"] == "out_of_scope"]
    judged = all(q["id"] in verdicts for q in items)

    def declined(q):
        verdict = verdicts.get(q["id"], {})
        return bool(verdict["declined"]) if "declined" in verdict else is_refusal(answers[q["id"]]["text"])

    row = {"run": run, "model": answers[items[0]["id"]]["model"], "n": len(items),
           "n_answerable": len(answerable), "n_out_of_scope": len(unanswerable)}
    if judged:
        row["judge"] = verdicts[items[0]["id"]].get("judge")
        scores = [int(verdicts[q["id"]].get("correct", 0)) for q in answerable]
        row["correct"] = mean(s == 2 for s in scores)
        row["lenient"] = mean(s >= 1 for s in scores)
        row["overall"] = (sum(s == 2 for s in scores) + sum(declined(q) for q in unanswerable)) / len(items)
        faithful = [verdicts[q["id"]]["faithful"] for q in answerable
                    if "faithful" in verdicts[q["id"]] and not declined(q)]
        if faithful:
            row["faithful"] = mean(bool(f) for f in faithful)
    # v1 has no fixed refusal sentence, so its refusals can only be counted by the judge.
    if judged or answers[items[0]["id"]]["refused"] is not None:
        row["false_refusal"] = mean(declined(q) for q in answerable)
        row["correct_refusal"] = mean(declined(q) for q in unanswerable)
    gated = [answers[q["id"]]["gated"] for q in unanswerable]
    row["gated_oos"] = mean(gated)
    cited_rows = [(q, answers[q["id"]]) for q in answerable if answers[q["id"]].get("cited")]
    if cited_rows:
        row["citation_hit"] = mean(any(is_relevant(a["passages"][c], q) for c in a["cited"])
                                   for q, a in cited_rows)
        row["cited_share"] = len(cited_rows) / len(answerable)
    llm_calls = [answers[q["id"]] for q in items if not answers[q["id"]]["gated"]]
    total = [answers[q["id"]]["timings"]["total_s"] for q in items]
    first = [answers[q["id"]]["timings"]["first_sentence_s"] for q in items
             if answers[q["id"]]["timings"].get("first_sentence_s")]
    row["p50_s"], row["p95_s"] = percentile(total, 50), percentile(total, 95)
    row["first_sentence_p50_s"] = percentile(first, 50)
    row["ttft_p50_s"] = percentile([a["timings"]["ttft_s"] for a in llm_calls if a["timings"].get("ttft_s")], 50)
    row["retrieval_p50_s"] = percentile([answers[q["id"]]["timings"]["retrieval_s"] for q in items], 50)
    row["tokens"] = mean(a["tokens"] for a in llm_calls)
    row["by_type"] = {}
    if judged:
        for kind in sorted({q["type"] for q in answerable}):
            group = [q for q in answerable if q["type"] == kind]
            row["by_type"][kind] = mean(int(verdicts[q["id"]].get("correct", 0)) == 2 for q in group)
    # Per-question share of required evidence in the context, averaged (same as retrieval recall@k).
    row["evidence_in_context"] = mean(
        mean(any(covers(p, g) for p in answers[q["id"]]["passages"]) for g in q["evidence"])
        for q in answerable)
    return row


def report(args) -> None:
    runs = sorted(p.stem for p in E2E.glob("*.jsonl") if not p.stem.endswith(".judge"))
    rows = [r for r in (summarize(run, args.split) for run in runs) if r]
    (RESULTS / f"e2e_{args.split}.json").write_text(json.dumps(rows, indent=2))
    table = markdown_table(rows, [
        ("run", "System"), ("model", "LLM"), ("overall", "Overall accuracy"), ("correct", "Answer accuracy"),
        ("faithful", "Faithfulness"), ("citation_hit", "Citation accuracy"),
        ("correct_refusal", "OOS refusal rate"), ("false_refusal", "False refusals"),
        ("first_sentence_p50_s", "First sentence p50 (s)"), ("p50_s", "Total p50 (s)"),
        ("p95_s", "Total p95 (s)")])
    (RESULTS / f"e2e_{args.split}.md").write_text(table + "\n")
    print(table)
    for r in rows:
        print(r["run"], {k: round(v, 3) for k, v in r["by_type"].items()},
              f"evidence in context {r['evidence_in_context']:.1%}", f"tokens/q {r['tokens']:.0f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--system", choices=["v1", "v2"], default="v2")
    g.add_argument("--provider", default="groq")
    g.add_argument("--model")
    g.add_argument("--split", default="test")
    g.add_argument("--tpm", type=int, default=7000, help="tokens-per-minute budget for hosted APIs")
    g.add_argument("--limit", type=int, help="answer at most this many new questions")
    j = sub.add_parser("judge")
    j.add_argument("--run", required=True)
    j.add_argument("--judge-provider", default="groq")
    j.add_argument("--judge-model", help=f"default: first available of {JUDGE_MODELS}")
    j.add_argument("--faithfulness", action="store_true", help="also check every claim against the passages")
    j.add_argument("--tpm", type=int, default=7000)
    r = sub.add_parser("report")
    r.add_argument("--split", default="test")
    args = parser.parse_args()
    E2E.mkdir(parents=True, exist_ok=True)
    try:
        {"generate": generate, "judge": judge, "report": report}[args.command](args)
    except RateLimitError as error:
        raise SystemExit(f"Rate limit reached ({error.message}). Progress is cached: rerun the same "
                         "command later to continue.") from None


if __name__ == "__main__":
    main()
