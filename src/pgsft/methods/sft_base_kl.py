"""SFT plus fixed-Base forward-KL objective."""

from __future__ import annotations

import torch

from pgsft.methods.base import ObjectiveContext, ObjectiveOutput, StatelessObjective
from pgsft.methods.full_vocab import chunked_sft_base_kl


class SFTBaseKLObjective(StatelessObjective):
    name = "sft_base_kl"
    required_row_schema = "sft"

    def __init__(self, coefficient: float = 0.1, chunk_size: int = 128):
        if coefficient != 0.1:
            raise ValueError("formal SFT+Base-KL requires coefficient=0.1")
        self.coefficient = coefficient
        self.chunk_size = chunk_size

    def compute(self, batch: dict[str, torch.Tensor], context: ObjectiveContext) -> ObjectiveOutput:
        required = {"student_hidden", "base_hidden", "targets"}
        if not required <= set(batch):
            raise ValueError(f"SFT+Base-KL batch is missing {sorted(required - set(batch))}")
        divisor = (
            batch["targets"].numel()
            if context.global_token_divisor is None
            else context.global_token_divisor
        )
        result = chunked_sft_base_kl(
            batch["student_hidden"],
            batch["base_hidden"],
            context.output_weight,
            batch["targets"],
            divisor=divisor,
            coefficient=self.coefficient,
            chunk_size=self.chunk_size,
        )
        return ObjectiveOutput(result.loss, result.loss_sum.detach(), result.token_count, result.diagnostics)
