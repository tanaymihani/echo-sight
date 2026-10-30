| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Evidence recall@4 | p50 latency (ms) |
|---|---|---|---|---|---|---|
| v1 baseline (regex sections, 500/80 chunks, MiniLM) | 58.2% | 72.2% | 78.5% | 0.666 | 74.7% | 7.58 |
| v2 parsing + heading-aware chunks | 59.5% | 72.2% | 74.7% | 0.669 | 71.5% | 6.62 |
| + heading path prepended to chunks | 64.6% | 73.4% | 81.0% | 0.712 | 78.5% | 6.64 |
| bge-small-en-v1.5 embeddings | 59.5% | 75.9% | 78.5% | 0.682 | 77.8% | 11.46 |
| bge-base-en-v1.5 embeddings | 68.4% | 83.5% | 87.3% | 0.763 | 82.3% | 32.29 |
| BM25 only | 68.4% | 84.8% | 87.3% | 0.763 | 84.8% | 0.20 |
| hybrid: bge-base + BM25 (RRF) | 70.9% | 89.9% | 92.4% | 0.802 | 91.1% | 30.90 |
| + ms-marco-MiniLM-L-6 reranker (final) | 77.2% | 88.6% | 92.4% | 0.842 | 91.1% | 218.96 |
| alt: bge-reranker-base instead | 79.7% | 91.1% | 93.7% | 0.854 | 89.9% | 1114.45 |
| alt: final with 400-char chunks | 78.5% | 92.4% | 93.7% | 0.845 | 92.4% | 144.36 |
| alt: final with 1200-char chunks | 81.0% | 87.3% | 93.7% | 0.857 | 89.9% | 308.70 |

79 answerable questions, dev split. A passage is relevant when it contains a labelled evidence quote. Recall@4 = share of required evidence in the 4 passages sent to the LLM.
