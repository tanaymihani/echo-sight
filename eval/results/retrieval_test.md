| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Evidence recall@4 | p50 latency (ms) |
|---|---|---|---|---|---|---|
| v1 baseline (regex sections, 500/80 chunks, MiniLM) | 51.9% | 71.4% | 75.3% | 0.619 | 72.7% | 8.03 |
| v2 parsing + heading-aware chunks | 59.7% | 81.8% | 84.4% | 0.703 | 80.5% | 6.43 |
| + heading path prepended to chunks | 64.9% | 81.8% | 84.4% | 0.740 | 82.5% | 6.61 |
| bge-small-en-v1.5 embeddings | 68.8% | 83.1% | 87.0% | 0.763 | 83.1% | 11.83 |
| bge-base-en-v1.5 embeddings | 74.0% | 83.1% | 85.7% | 0.796 | 84.4% | 32.05 |
| BM25 only | 62.3% | 79.2% | 85.7% | 0.717 | 81.8% | 0.20 |
| hybrid: bge-base + BM25 (RRF) | 71.4% | 85.7% | 89.6% | 0.797 | 87.0% | 37.28 |
| + ms-marco-MiniLM-L-6 reranker (final) | 76.6% | 88.3% | 89.6% | 0.827 | 89.0% | 322.44 |
| alt: bge-reranker-base instead | 79.2% | 88.3% | 92.2% | 0.845 | 90.3% | 1181.18 |
| alt: final with 400-char chunks | 67.5% | 87.0% | 87.0% | 0.761 | 85.1% | 154.75 |
| alt: final with 1200-char chunks | 75.3% | 89.6% | 90.9% | 0.825 | 89.0% | 311.41 |

77 answerable questions, test split. A passage is relevant when it contains a labelled evidence quote. Recall@4 = share of required evidence in the 4 passages sent to the LLM.
