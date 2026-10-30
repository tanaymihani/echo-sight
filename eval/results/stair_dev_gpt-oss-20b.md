Router: groq/openai/gpt-oss-20b. 79 answerable and 15 out-of-scope dev questions.

Section level (a section is relevant when its own text holds an evidence quote):

| Method | Recall@1 | Recall@3 | Recall@1, subsections count |
|---|---|---|---|
| BM25 | 81.0% | 92.4% | 81.0% |
| dense (bge-base) | 74.7% | 88.6% | 75.9% |
| hybrid (RRF) | 82.3% | 94.9% | 84.8% |
| v2 final (hybrid + reranker) | 84.8% | 97.5% | 86.1% |
| STAIR ToC router (groq/openai/gpt-oss-20b, zero-shot) | 59.5% | 74.7% | 63.3% |

Chunk level:

| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Evidence recall@4 |
|---|---|---|---|---|---|
| v2 (no ToC routing) | 77.2% | 88.6% | 92.4% | 0.842 | 91.1% |
| v2 + STAIR, fusion=candidates | 77.2% | 88.6% | 92.4% | 0.842 | 91.1% |
| v2 + STAIR, fusion=final | 63.3% | 81.0% | 88.6% | 0.740 | 85.4% |
| v2 + STAIR, fusion=restrict | 62.0% | 69.6% | 73.4% | 0.667 | 71.5% |

Routing latency p50 0.58 s, p95 1.75 s (runs in parallel with local retrieval). Outputs rejected as invalid sections: 0. Empty routes: 1.3% of answerable, 60.0% of out-of-scope questions.
