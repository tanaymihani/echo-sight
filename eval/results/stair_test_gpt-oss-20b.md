Router: groq/openai/gpt-oss-20b. 77 answerable and 15 out-of-scope test questions.

Section level (a section is relevant when its own text holds an evidence quote):

| Method | Recall@1 | Recall@3 | Recall@1, subsections count |
|---|---|---|---|
| BM25 | 70.1% | 84.4% | 70.1% |
| dense (bge-base) | 77.9% | 85.7% | 81.8% |
| hybrid (RRF) | 76.6% | 89.6% | 77.9% |
| v2 final (hybrid + reranker) | 83.1% | 90.9% | 84.4% |
| STAIR ToC router (groq/openai/gpt-oss-20b, zero-shot) | 59.7% | 71.4% | 63.6% |

Chunk level:

| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Evidence recall@4 |
|---|---|---|---|---|---|
| v2 (no ToC routing) | 76.6% | 88.3% | 89.6% | 0.827 | 89.0% |
| v2 + STAIR, fusion=candidates | 76.6% | 88.3% | 89.6% | 0.829 | 89.0% |
| v2 + STAIR, fusion=final | 61.0% | 85.7% | 89.6% | 0.733 | 87.7% |
| v2 + STAIR, fusion=restrict | 61.0% | 71.4% | 71.4% | 0.666 | 71.4% |

Routing latency p50 0.64 s, p95 1.25 s (runs in parallel with local retrieval). Outputs rejected as invalid sections: 0. Empty routes: 2.6% of answerable, 60.0% of out-of-scope questions.
