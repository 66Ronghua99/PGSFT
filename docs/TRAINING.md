# Training

## Implemented methods

All losses are summed over active assistant tokens and divided by the global
active-token count across the whole optimizer window and all distributed ranks.

- **SFT:** ordinary assistant-token negative log likelihood.
- **DFT:** SFT weighted by the detached probability assigned to the demonstrated
  token, `stopgrad(p_theta(y|x)) * -log p_theta(y|x)`.
- **SFT+Base-KL:** SFT plus `0.1 * KL(p_base || p_theta)` at every active token.
  The Base teacher is fixed and detached.
- **PSFT:** upper-clipped likelihood ratio objective against an old-policy LoRA
  snapshot, with epsilon `0.28`; the snapshot refreshes every 8 optimizer steps.
- **PG-SFT:** computes `KL(p_privileged || p_student)` and subtracts the component
  parallel to the fixed-Base direction. The projection allocation coefficient is
  computed per assistant turn. Both teacher lanes are detached.

Full-vocabulary KL/projection terms are evaluated exactly in FP32 chunks; they
do not use a top-k vocabulary approximation. Implementations are in
`src/pgsft/methods/`, while optimizer, accumulation, checkpoint, resume, old
policy refresh and selection logic are in `src/pgsft/training/`.

## Recipes

There are ten recipes under `configs/training/{qwen3,qwen35}`: `sft.yaml`,
`dft.yaml`, `sft_base_kl.yaml`, `psft.yaml`, and `pg_sft.yaml`. Each fixes the
model revision, exact train/inference template, LoRA rank 32 / alpha 64,
optimizer schedule, seed, checkpoint cadence and selection rule. A recipe marked
`paper_reported` corresponds to a reported paper run; `supported_unreported`
means the implementation is supported but that exact backbone/method cell was
not reported.

Training outputs are self-contained:

```text
outputs/train/<run-id>/
  resolved_config.json
  manifest.json
  train.log
  checkpoint-*/
  selected_ckpt.txt
  merged/
```

Resume with the same command plus `--resume`. The resolved config and immutable
input digests must match. Merge only after checkpoint selection:

```bash
uv run pgsft-merge --run-dir outputs/train/<run-id> --backbone qwen3
```

Qwen3.5 uses FlashAttention 4 in the reported recipe and requires a compatible
GPU/build. Qwen3 recipes should be run on at least two high-memory GPUs for the
paper batch/sequence lengths. Reduce only per-device microbatch size; preserve
global batch size, loss normalization and other recipe fields when reproducing.
