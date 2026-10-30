"""Evaluate STAIR-style ToC routing (echosight/toc.py) against the existing retrievers.

    python eval/stair_eval.py --split dev                                  # choose the fusion here
    python eval/stair_eval.py --split test
    python eval/stair_eval.py --split dev --router ollama:qwen3.5:4b       # local router

Reports (1) section-level Recall@1/@3, the metric STAIR is evaluated with, for the router and for
the chunk retrievers aggregated to sections; (2) chunk-level metrics for v2 with and without ToC
routing; (3) how often the router returns nothing, for answerable vs out-of-scope questions.
Routing decisions are cached in eval/results/stair_cache.json, so variants cost no extra API calls.
"""

import argparse
import json
import time
from dataclasses import replace

from common import RESULTS, covers, load_questions, markdown_table, mean, percentile, ranking_metrics

from echosight.config import LLMConfig, RetrievalConfig
from echosight.index import load_sections
from echosight.llm import LLM
from echosight.retrieval import Retriever
from echosight.toc import ToCRouter

FUSIONS = ("candidates", "final", "restrict")


def relevant_sections(sections, item) -> set[int]:
    """Sections whose own text contains a labelled evidence quote."""
    return {s for s, sec in enumerate(sections) if any(covers(sec.text, g) for g in item["evidence"])}


def section_order(hits, path_index) -> list[int]:
    order = []
    for hit in hits:
        s = path_index[hit.chunk.path]
        if s not in order:
            order.append(s)
    return order


def section_metrics(ranked: list[int], gold: set[int], sections) -> dict:
    def in_subtree(s):     # a routed parent counts if one of its subsections holds the evidence
        path = sections[s].path
        return any(sections[g].path[: len(path)] == path for g in gold)
    return {"R@1": float(bool(ranked[:1]) and ranked[0] in gold),
            "R@3": float(any(s in gold for s in ranked[:3])),
            "R@1_subtree": float(bool(ranked[:1]) and in_subtree(ranked[0]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--router", default="groq:openai/gpt-oss-20b")
    parser.add_argument("--tpm", type=int, default=7000)
    args = parser.parse_args()

    sections = list(load_sections())
    path_index = {s.path: i for i, s in enumerate(sections)}
    provider, model = args.router.split(":", 1)
    router = ToCRouter(sections, LLM(LLMConfig.from_env(provider, model)), k=3,
                       cache_path=RESULTS / "stair_cache.json")
    items = load_questions(args.split)
    answerable = [q for q in items if q["type"] != "out_of_scope"]
    print(f"{len(answerable)} answerable + {len(items) - len(answerable)} out-of-scope {args.split} questions; "
          f"router {router.llm.name}, ToC of {len(sections)} sections")

    # 1. Route every question once (cached), staying under the tokens-per-minute budget.
    routes, latencies = {}, []
    for i, q in enumerate(items, 1):
        cached = f"{router.llm.name}|3|{q['question']}" in router.cache
        routes[q["id"]] = router.route(q["question"])
        latencies.append(router.last_latency)
        if not cached and provider != "ollama":
            time.sleep(60 * 950 / args.tpm)
        if i % 20 == 0:
            print(f"  routed {i}/{len(items)}", flush=True)

    # 2. Section-level comparison (STAIR's metric).
    base = RetrievalConfig()
    variants = {"BM25": replace(base, mode="bm25", reranker=None),
                "dense (bge-base)": replace(base, mode="dense", reranker=None),
                "hybrid (RRF)": replace(base, reranker=None),
                "v2 final (hybrid + reranker)": base}
    section_rows = []
    for name, cfg in variants.items():
        retriever = Retriever(cfg)
        scores = [section_metrics(section_order(retriever.search(q["question"], 20), path_index),
                                  relevant_sections(sections, q), sections) for q in answerable]
        section_rows.append({"method": name, **{k: mean(s[k] for s in scores) for k in scores[0]}})
    scores = [section_metrics(routes[q["id"]], relevant_sections(sections, q), sections) for q in answerable]
    section_rows.append({"method": f"STAIR ToC router ({router.llm.name}, zero-shot)",
                         **{k: mean(s[k] for s in scores) for k in scores[0]}})

    # 3. Chunk-level: v2 with ToC routing fused in different ways (same cached routes).
    chunk_rows = []
    configs = [("v2 (no ToC routing)", base, None)] + [
        (f"v2 + STAIR, fusion={f}", replace(base, toc_router=args.router, toc_fusion=f), router) for f in FUSIONS]
    for name, cfg, r in configs:
        retriever = Retriever(cfg, router=r)
        per_q, local_ms = [], []
        for q in answerable:
            start = time.perf_counter()
            hits = retriever.search(q["question"], 10)
            local_ms.append((time.perf_counter() - start) * 1000)
            per_q.append(ranking_metrics([h.chunk.text for h in hits], q))
        row = {"config": name, **{k: mean(m[k] for m in per_q) for k in ("hit@1", "hit@3", "hit@5", "mrr", "recall@4")}}
        row["p50_ms"] = percentile(local_ms, 50)
        chunk_rows.append(row)
        print(f"  {name}: hit@1 {row['hit@1']:.1%} mrr {row['mrr']:.3f} recall@4 {row['recall@4']:.1%}", flush=True)

    oos = [q for q in items if q["type"] == "out_of_scope"]
    summary = {
        "split": args.split, "router": router.llm.name,
        "invalid_outputs": router.invalid,
        "empty_route_answerable": mean(not routes[q["id"]] for q in answerable),
        "empty_route_out_of_scope": mean(not routes[q["id"]] for q in oos) if oos else None,
        "routing_latency_p50_s": percentile(latencies, 50),
        "routing_latency_p95_s": percentile(latencies, 95),
        "sections": section_rows, "chunks": chunk_rows,
        "routes": {qid: [sections[s].title for s in r] for qid, r in routes.items()},
    }
    slug = model.split("/")[-1].replace(":", "-")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"stair_{args.split}_{slug}.json").write_text(json.dumps(summary, indent=2))
    text = "\n\n".join([
        f"Router: {router.llm.name}. {len(answerable)} answerable and {len(oos)} out-of-scope {args.split} questions.",
        "Section level (a section is relevant when its own text holds an evidence quote):",
        markdown_table(section_rows, [("method", "Method"), ("R@1", "Recall@1"), ("R@3", "Recall@3"),
                                      ("R@1_subtree", "Recall@1, subsections count")]),
        "Chunk level:",
        markdown_table(chunk_rows, [("config", "Configuration"), ("hit@1", "Hit@1"), ("hit@3", "Hit@3"),
                                    ("hit@5", "Hit@5"), ("mrr", "MRR@10"), ("recall@4", "Evidence recall@4")]),
        f"Routing latency p50 {summary['routing_latency_p50_s']:.2f} s, p95 {summary['routing_latency_p95_s']:.2f} s "
        f"(runs in parallel with local retrieval). Outputs rejected as invalid sections in uncached calls this run: {router.invalid}. "
        f"Empty routes: {summary['empty_route_answerable']:.1%} of answerable, "
        f"{(summary['empty_route_out_of_scope'] or 0):.1%} of out-of-scope questions.",
    ])
    (RESULTS / f"stair_{args.split}_{slug}.md").write_text(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
