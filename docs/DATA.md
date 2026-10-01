# Data preparation

## Source

Training traces come from `nvidia/Nemotron-SFT-SWE-v3` at revision
`3f73de64c1fe928a8f538fe45ccc10c228cc4c6a`. The downloader records file
digests and never modifies the downloaded snapshot. The canonicalizer applies
the paper's trace filters and writes accepted rows, rejected rows with reasons,
and a manifest.

## Canonical JSONL

Each line is one object:

```json
{
  "id": "unique-trace-id",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "reasoning_content": "..."},
    {"role": "tool", "content": "..."}
  ],
  "tools": [],
  "source": {"dataset": "..."}
}
```

Roles are `system`, `user`, `assistant`, and `tool`. Every assistant turn must
follow a user/tool carrier and have nonempty action content. `source` must be
JSON-serializable and must not contain secrets or absolute local paths.

## Tokenized SFT rows

SFT, DFT, SFT+Base-KL and PSFT consume:

```json
{"id":"...","input_ids":[...],"labels":[...],"assistant_turn_ids":[...]}
```

Non-assistant labels are `-100`. `assistant_turn_ids` associates every trained
token with its assistant turn and uses `-1` elsewhere.

## Tokenized PG-SFT rows

PG-SFT consumes aligned student/privileged pairs:

```json
{
  "id": "...",
  "student_input_ids": [...],
  "privileged_input_ids": [...],
  "student_positions": [...],
  "privileged_positions": [...],
  "target_ids": [...],
  "turn_indices": [...],
  "token_indices": [...]
}
```

Before each assistant turn, the privileged lane appends the demonstrated next
action to the immediately preceding user/tool message:

```text
<HINT>
A suggested action for the next step is:
{assistant action}
Continue thinking about the next step.
</HINT>
```

`student_positions[k]` and `privileged_positions[k]` identify the two hidden
states predicting the same `target_ids[k]`; `(turn_indices[k], token_indices[k])`
is the semantic alignment key. Rows exceeding the pinned backbone length are
rejected, not truncated.

The tokenizer command writes both `sft.jsonl` and `pgsft.jsonl` plus a manifest.
Recipes may rename/split these files, but their schema is unchanged.
