#!/usr/bin/env bash
# Run eval steps one after another, retrying each until it succeeds. Answers are cached, so a retry
# after a rate-limit stop resumes where it left off.
#   bash eval/run_resilient.sh "python eval/e2e_eval.py generate --system v2 --provider groq" "..."
cd "$(dirname "$0")/.."
WAIT=${RETRY_WAIT:-600}
for step in "$@"; do
  for attempt in $(seq 1 ${MAX_ATTEMPTS:-30}); do
    echo "[$(date +%T)] start ($attempt): $step"
    if eval "$step"; then
      echo "[$(date +%T)] done: $step"
      continue 2
    fi
    echo "[$(date +%T)] failed, retrying in ${WAIT}s: $step"
    sleep "$WAIT"
  done
  echo "[$(date +%T)] GAVE UP: $step"
  exit 1
done
echo "[$(date +%T)] ALL STEPS DONE"
