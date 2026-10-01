"""Detached-token-probability DFT objective with global token reduction."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pgsft.methods.base import (
    ObjectiveContext,
    ObjectiveOutput,
    StatelessObjective,
    active_rows,
    projection_chunks,
    resolve_divisor,
)


class DFTObjective(StatelessObjective):
    name = "dft"
    required_row_schema = "sft"

    def compute(
        self, batch: dict[str, torch.Tensor], context: ObjectiveContext
    ) -> ObjectiveOutput:
        rows, targets = active_rows(batch, context)
        loss_sum = torch.zeros((), dtype=torch.float32, device=rows.device)
        weight_sum = torch.zeros((), dtype=torch.float32, device=rows.device)
        for logits, chunk_targets in projection_chunks(rows, targets, context):
            log_probs = F.log_softmax(logits, dim=-1)
            log_q = log_probs.gather(1, chunk_targets[:, None]).squeeze(1)
            weights = log_q.exp().detach()
            loss_sum = loss_sum + (weights * -log_q).sum()
            weight_sum = weight_sum + weights.sum()
        count = torch.tensor(targets.numel(), dtype=torch.int64, device=loss_sum.device)
        loss = loss_sum / resolve_divisor(count, context)
        return ObjectiveOutput(
            loss=loss,
            loss_sum=loss_sum.detach(),
            token_count=count,
            metrics={
                "mean_demonstrated_probability": (weight_sum / count).detach(),
                "weight_sum": weight_sum.detach(),
            },
        )
