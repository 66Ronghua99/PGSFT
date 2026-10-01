# Evaluation

GPQA, BFCL and LiveCodeBench recipes for both backbones use a uniform 65,536
token context. SWE-bench keeps its independent agent contract: 8,192 output
tokens per turn and at most 150 tool iterations, within the same served 64K
context.

## GPQA Diamond

The fixed denominator is 198. Scoring examines only text after a closed
`</think>` and accepts one unambiguous answer from `Answer: X`, `The answer is
(X)`, `\\boxed{X}`, a final standalone letter, or exact normalized option text.
Conflicting answers, unfinished reasoning and absent explicit answers fail.

## BFCL v4

`configs/evaluation/bfcl-200.json` freezes 200 IDs: 50 each from multi-turn base,
missing-parameter, long-context and missing-function categories. The isolated
`bfcl-eval==2026.3.23` environment owns dialogue state and official scoring.
The local handler preserves reasoning separately and converts only valid
`<tool_call>` JSON objects. A 64K overflow is submitted as an empty response and
remains in the denominator.

## LiveCodeBench release v6

`livecodebench-120.json` freezes 37 easy, 43 medium and 40 hard IDs. Run
`scripts/prepare_lcb.py` in `envs/livecodebench` against the pinned official
dataset script to create `data/evaluation/livecodebench/prepared.jsonl`. Code is
extracted using official `LMStyle.OpenAIChat` and graded inside the Docker image:

```bash
docker build -f envs/livecodebench/Dockerfile -t pgsft-lcb .
bash scripts/fetch_lcb_source.sh
PYTHONPATH=envs/livecodebench/source uv run --directory envs/livecodebench \
  python scripts/prepare_lcb.py \
  --dataset-script data/raw/livecodebench/code_generation_lite.py \
  --inventory configs/evaluation/livecodebench-120.json \
  --output data/evaluation/livecodebench/prepared.jsonl
```

## SWE-bench Lite tools-v2

The 89-task inventory is the ordered union of easy-50 and the frozen random-50
(11 overlap). The agent supports JSON-framed fresh-shell bash, structured editor
operations, exact thinking/action boundaries, maximum-iteration and repeated
format termination, and a temporary Git index that captures tracked plus
non-ignored untracked files without mutating the workspace index or HEAD.
Predictions are passed to official `swebench==4.1.0` through
`scripts/run_swebench_harness.sh`; empty patches and runner failures remain in
the 89-task denominator.

Evaluation output always contains one result per inventory ID and reports status
counts rather than silently dropping malformed or failed generations.
