"""Retrieval ablation: the v1 baseline, then each v2 change added one at a time.

    python eval/retrieval_eval.py --split dev     # choose settings here
    python eval/retrieval_eval.py --split test    # report these

Writes eval/results/retrieval_<split>.json and .md.
"""

import argparse
import gc
import json
import time
from dataclasses import asdict, replace

from common import RESULTS, load_questions, markdown_table, mean, percentile, ranking_metrics

from echosight import index, retrieval
from echosight.config import RetrievalConfig
from echosight.retrieval import Retriever
from eval.v1_baseline import V1Retriever

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
START = RetrievalConfig(embed_model=MINILM, chunk_size=800, section_headers=False, mode="dense",
                        reranker=None, min_score=None)
HEADERS = replace(START, section_headers=True)
BGE_BASE = replace(HEADERS, embed_model="BAAI/bge-base-en-v1.5")
HYBRID = replace(BGE_BASE, mode="hybrid")
FINAL = replace(HYBRID, reranker="cross-encoder/ms-marco-MiniLM-L-6-v2")
assert FINAL == RetrievalConfig(), "FINAL should match the app's default settings"

ABLATION = [
    ("v2 parsing + heading-aware chunks", START),
    ("+ heading path prepended to chunks", HEADERS),
    ("bge-small-en-v1.5 embeddings", replace(HEADERS, embed_model="BAAI/bge-small-en-v1.5")),
    ("bge-base-en-v1.5 embeddings", BGE_BASE),
    ("BM25 only", replace(BGE_BASE, mode="bm25")),
    ("hybrid: bge-base + BM25 (RRF)", HYBRID),
    ("+ ms-marco-MiniLM-L-6 reranker (final)", FINAL),
    ("alt: bge-reranker-base instead", replace(HYBRID, reranker="BAAI/bge-reranker-base")),
    ("alt: final with 400-char chunks", replace(FINAL, chunk_size=400)),
    ("alt: final with 1200-char chunks", replace(FINAL, chunk_size=1200)),
]


def auroc(positives: list[float], negatives: list[float]) -> float:
    """Probability that a random answerable question outscores a random out-of-scope one."""
    pairs = [(p > n) + 0.5 * (p == n) for p in positives for n in negatives]
    return mean(pairs)


def evaluate(name: str, search, items: list[dict], unanswerable: list[dict]) -> dict:
    for item in items[:3]:          # warm up caches and accelerator kernels
        search(item["question"])
    per_question, latencies = [], []
    for item in items:
        start = time.perf_counter()
        texts, top_score = search(item["question"])
        latencies.append(time.perf_counter() - start)
        metrics = ranking_metrics(texts, item)
        per_question.append({"id": item["id"], "type": item["type"], **metrics, "top_score": top_score})
    row = {"config": name, "n": len(items)}
    for key in ("hit@1", "hit@3", "hit@5", "hit@10", "mrr", "recall@4", "recall@8"):
        row[key] = mean(q[key] for q in per_question)
    row["p50_ms"] = percentile(latencies, 50) * 1000
    row["p95_ms"] = percentile(latencies, 95) * 1000
    if per_question[0]["top_score"] is not None and unanswerable:
        negatives = [search(q["question"])[1] for q in unanswerable]
        row["auroc"] = auroc([q["top_score"] for q in per_question], negatives)
        row["oos_scores"] = negatives
    row["per_question"] = per_question
    print(f"  {name}: hit@1 {row['hit@1']:.1%}  hit@5 {row['hit@5']:.1%}  mrr {row['mrr']:.3f}  "
          f"recall@4 {row['recall@4']:.1%}  p50 {row['p50_ms']:.0f} ms"
          + (f"  auroc {row['auroc']:.3f}" if "auroc" in row else ""), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    args = parser.parse_args()
    items = load_questions(args.split, answerable=True)
    unanswerable = load_questions(args.split, answerable=False)
    print(f"{len(items)} answerable questions ({args.split} split)")

    rows = []
    v1 = V1Retriever()
    rows.append(evaluate("v1 baseline (regex sections, 500/80 chunks, MiniLM)",
                         lambda q: ([c["text"] for c in v1.search(q, 10)], None), items, []))
    del v1
    for name, cfg in ABLATION:
        retriever = Retriever(cfg)

        def search(q, r=retriever):
            hits = r.search(q, 10)
            return [h.chunk.text for h in hits], (hits[0].score if r.reranker else None)

        row = evaluate(name, search, items, unanswerable)
        row["settings"] = asdict(cfg)
        rows.append(row)
        del retriever, search
        index.embedder.cache_clear()
        retrieval.cross_encoder.cache_clear()
        gc.collect()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"retrieval_{args.split}.json").write_text(json.dumps(rows, indent=2))
    table = markdown_table(rows, [("config", "Configuration"), ("hit@1", "Hit@1"), ("hit@3", "Hit@3"),
                                  ("hit@5", "Hit@5"), ("mrr", "MRR@10"), ("recall@4", "Evidence recall@4"),
                                  ("p50_ms", "p50 latency (ms)")])
    note = (f"{len(items)} answerable questions, {args.split} split. A passage is relevant when it "
            "contains a labelled evidence quote. Recall@4 = share of required evidence in the 4 passages "
            "sent to the LLM.")
    (RESULTS / f"retrieval_{args.split}.md").write_text(f"{table}\n\n{note}\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
