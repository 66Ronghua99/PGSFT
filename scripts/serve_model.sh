#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?model directory required}
SERVED_NAME=${2:?served model name required}
TEMPLATE=${3:?chat template required}
PORT=${4:-8000}
TP=${5:-1}
CONTEXT=${6:-65536}
MEMORY=${7:-0.90}
GPUS=${8:-0}

export CUDA_VISIBLE_DEVICES="$GPUS"
SERVING_BIN="$(pwd)/envs/serving/.venv/bin"
export PATH="$SERVING_BIN:$PATH"
exec uv run --directory envs/serving vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --chat-template "$TEMPLATE" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$CONTEXT" \
  --gpu-memory-utilization "$MEMORY" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
