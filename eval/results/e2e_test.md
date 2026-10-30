| System | LLM | Overall accuracy | Answer accuracy | Faithfulness | Citation accuracy | OOS refusal rate | False refusals | First sentence p50 (s) | Total p50 (s) | Total p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| v1-ollama | ollama/qwen3.5:4b | 64.1% | 58.4% |  |  | 93.3% | 10.4% | 6.88 | 10.21 | 12.61 |
| v2-groq | groq/openai/gpt-oss-120b | 80.4% | 76.6% | 94.4% | 94.4% | 100.0% | 6.5% | 0.89 | 0.94 | 2.81 |
| v2-ollama | ollama/qwen3.5:4b | 73.9% | 71.4% |  | 85.7% | 86.7% | 0.0% | 6.23 | 8.02 | 10.88 |
| v3-groq | groq/openai/gpt-oss-120b | 80.4% | 76.6% | 94.4% | 94.4% | 100.0% | 6.5% | 0.96 | 0.99 | 2.81 |
| v3-ollama | ollama/qwen3.5:4b | 75.0% | 71.4% |  | 85.7% | 93.3% | 0.0% | 6.13 | 7.84 | 10.88 |
