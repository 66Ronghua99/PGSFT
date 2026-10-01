# PG-SFT reproduction code

This directory is the standalone supplementary-material package for the PG-SFT
paper. It contains the complete implementation of PG-SFT, four comparison
methods (SFT, DFT, SFT+Base-KL, and PSFT), deterministic trace preparation,
Qwen3/Qwen3.5 templates and recipes, model merging, and the GPQA, BFCL,
LiveCodeBench and SWE-bench evaluation entry points.

Fixed-alpha and NLL-SFT are intentionally outside the paper reproduction set.

## Install

Python 3.13 and CUDA 13 are used by the training lock. From this directory:

```bash
uv sync --group test
uv run pgsft-prepare-data --help
uv run pgsft-train --help
uv run pgsft-merge --help
uv run pgsft-eval --help
```

BFCL, LiveCodeBench, SWE-bench, and vLLM have isolated environments under
`envs/` because their dependency sets conflict with training. Create them with
`uv sync --directory envs/<name>`.

## Minimal reproduction

1. Download and canonicalize Nemotron SWE traces:

```bash
uv run pgsft-prepare-data download \
  --revision 3f73de64c1fe928a8f538fe45ccc10c228cc4c6a \
  --output-dir data/raw/nemotron
uv run pgsft-prepare-data canonicalize \
  --source data/raw/nemotron --backbone qwen35 \
  --output-dir data/canonical/qwen35
uv run pgsft-prepare-data split \
  --canonical data/canonical/qwen35/canonical.jsonl --backbone qwen35 \
  --output-dir data/splits/qwen35
uv run pgsft-prepare-data tokenize \
  --canonical data/splits/qwen35/train-1000.jsonl --backbone qwen35 \
  --output-dir data/tokenized/qwen35/train
uv run pgsft-prepare-data tokenize \
  --canonical data/splits/qwen35/holdout-120.jsonl --backbone qwen35 \
  --output-dir data/tokenized/qwen35/holdout
```

2. Point the selected recipe's `data` fields at the generated JSONL files and
train. Multi-GPU launch uses Accelerate in the normal way:

```bash
uv run accelerate launch --num_processes 2 -m pgsft.cli.train \
  --config configs/training/qwen35/pg_sft.yaml \
  --run-id qwen35-pgsft --output-root outputs/train
uv run pgsft-merge --run-dir outputs/train/qwen35-pgsft \
  --backbone qwen35
```

3. Serve the merged model and evaluate:

```bash
bash scripts/serve_model.sh outputs/train/qwen35-pgsft/merged \
  pgsft-qwen35 templates/qwen35_inference.jinja 8000 1 65536 0.90 0
uv run pgsft-eval --config configs/evaluation/qwen35/gpqa.yaml \
  --endpoint http://127.0.0.1:8000/v1 \
  --model-root outputs/train/qwen35-pgsft/merged \
  --output-dir outputs/eval/qwen35-pgsft-gpqa
```

See [DATA.md](docs/DATA.md), [TRAINING.md](docs/TRAINING.md), and
[EVALUATION.md](docs/EVALUATION.md) for exact contracts. All paths stored in
recipes are relative to this directory; model weights and raw datasets are not
redistributed.
