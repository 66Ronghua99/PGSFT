# Reproducibility notes

- Every model, dataset and external evaluator is referenced by an immutable
  revision or exact package version.
- Raw datasets and model weights are downloaded separately and are not part of
  this directory.
- Manifests contain SHA-256 records for transformed inputs, templates, selected
  checkpoints and merged outputs.
- Random seed 42 is used unless a benchmark recipe explicitly leaves sampling
  seed unset. Formal benchmark IDs and denominators are checked in.
- Training resume restores optimizer/scheduler/sampler state and, for PSFT, the
  old-policy adapter snapshot and refresh step.
- Evaluation environments are isolated because BFCL, LCB, SWE-bench and vLLM
  have mutually incompatible dependencies. Use the `uv.lock` beside each
  environment's `pyproject.toml`.
- GPU kernels, Docker runtime and driver versions can affect throughput and
  occasionally numerical ties. Record them with each run. The code does not
  claim bitwise equivalence across GPU architectures.
