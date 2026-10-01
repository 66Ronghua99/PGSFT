"""Right-padding collator for immutable SFT token rows."""

from __future__ import annotations

from typing import Any, Sequence

import torch


class SFTCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not rows:
            raise ValueError("cannot collate an empty batch")
        width = max(len(row["input_ids"]) for row in rows)
        input_ids = []
        labels = []
        attention = []
        turn_ids = []
        for row in rows:
            length = len(row["input_ids"])
            padding = width - length
            input_ids.append([*row["input_ids"], *([self.pad_token_id] * padding)])
            labels.append([*row["labels"], *([-100] * padding)])
            attention.append([1] * length + [0] * padding)
            turn_ids.append([*row["assistant_turn_ids"], *([-100] * padding)])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "assistant_turn_ids": torch.tensor(turn_ids, dtype=torch.long),
        }


class PairedCollator:
    """Independently pad paired lanes while retaining per-row target positions."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ValueError("cannot collate an empty paired batch")
        student_width = max(len(row["student_input_ids"]) for row in rows)
        privileged_width = max(len(row["privileged_input_ids"]) for row in rows)
        result: dict[str, Any] = {"ids": [row["id"] for row in rows]}
        for lane, width in (("student", student_width), ("privileged", privileged_width)):
            ids = []
            masks = []
            for row in rows:
                values = row[f"{lane}_input_ids"]
                padding = width - len(values)
                ids.append([*values, *([self.pad_token_id] * padding)])
                masks.append([1] * len(values) + [0] * padding)
            result[f"{lane}_input_ids"] = torch.tensor(ids, dtype=torch.long)
            result[f"{lane}_attention_mask"] = torch.tensor(masks, dtype=torch.long)
        for key in ("student_positions", "privileged_positions", "target_ids", "turn_indices"):
            lengths = {len(row[key]) for row in rows}
            if len(lengths) != 1:
                raise ValueError("paired batching requires an equal target count per row")
            result[key] = torch.tensor([row[key] for row in rows], dtype=torch.long)
        return result
