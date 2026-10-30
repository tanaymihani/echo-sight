#!/usr/bin/env bash
# Reproduce every number in the README. Cached answers and judgements are reused, so reruns are cheap.
#   bash eval/run_all.sh            (needs GROQ_API_KEY in .env and `ollama serve` running)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

$PY eval/check_dataset.py
$PY eval/retrieval_eval.py --split test
$PY eval/listening_baseline.py --split test

$PY eval/e2e_eval.py generate --system v2 --provider ollama     # offline mode
$PY eval/e2e_eval.py generate --system v1 --provider ollama     # original pipeline, same local LLM
$PY eval/e2e_eval.py generate --system v2 --provider groq       # default mode

$PY eval/e2e_eval.py judge --run v2-groq --faithfulness
$PY eval/e2e_eval.py judge --run v2-ollama
$PY eval/e2e_eval.py judge --run v1-ollama
$PY eval/stair_eval.py --split dev
$PY eval/stair_eval.py --split test
$PY eval/stair_gate.py
$PY eval/e2e_eval.py report
$PY eval/update_readme.py
