"""Turn-allocated path-projected PG-SFT objective."""

from __future__ import annotations

import torch

from pgsft.methods.base import ObjectiveContext, ObjectiveOutput, StatelessObjective
from pgsft.methods.full_vocab import chunked_pg_sft_kl


class PGSFTObjective(StatelessObjective):
    name = "pg_sft"
    required_row_schema = "paired"

    def __init__(self, chunk_size: int = 128):
        self.chunk_size = chunk_size

    def compute(self, batch: dict[str, torch.Tensor], context: ObjectiveContext) -> ObjectiveOutput:
        required = {"student_hidden", "base_hidden", "privileged_hidden", "targets", "turn_ids"}
        if not required <= set(batch):
            raise ValueError(f"PG-SFT batch is missing {sorted(required - set(batch))}")
        divisor = (
            batch["targets"].numel()
            if context.global_token_divisor is None
            else context.global_token_divisor
        )
        result = chunked_pg_sft_kl(
            batch["student_hidden"],
            batch["base_hidden"],
            batch["privileged_hidden"],
            context.output_weight,
            batch["targets"],
            batch["turn_ids"],
            divisor=divisor,
            chunk_size=self.chunk_size,
        )
        return ObjectiveOutput(result.loss, result.loss_sum.detach(), result.token_count, result.diagnostics)
