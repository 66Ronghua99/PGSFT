#!/usr/bin/env bash
set -euo pipefail

PREDICTIONS=${1:?predictions.jsonl required}
REPORT=${2:?report path required}
WORKERS=${3:-4}
RUN_ID="pgsft_$(sha256sum "$PREDICTIONS" | cut -c1-16)"
OUT_DIR=$(dirname "$REPORT")

python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --split test \
  --predictions_path "$PREDICTIONS" \
  --max_workers "$WORKERS" \
  --run_id "$RUN_ID" \
  --cache_level env \
  --report_dir "$OUT_DIR"

FOUND=$(find "$OUT_DIR" . -maxdepth 1 -name "*.${RUN_ID}.json" -print -quit)
test -n "$FOUND"
mv "$FOUND" "$REPORT"
