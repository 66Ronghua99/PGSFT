"""Complete-assistant supervised fine-tuning objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pgsft.methods.base import (
    ObjectiveContext,
    ObjectiveOutput,
    StatelessObjective,
    active_projection,
    active_rows,
    projection_chunks,
    resolve_divisor,
)


class SFTObjective(StatelessObjective):
    name = "sft"
    required_row_schema = "sft"

    def compute(
        self, batch: dict[str, torch.Tensor], context: ObjectiveContext
    ) -> ObjectiveOutput:
        rows, targets = active_rows(batch, context)
        backend = context.extras.get("projection_backend", "chunked")
        if backend == "liger":
            if rows.device.type != "cuda":
                raise ValueError("Liger fused projection requires CUDA")
            from liger_kernel.transformers.functional import (
                liger_fused_linear_cross_entropy,
            )

            loss_sum = liger_fused_linear_cross_entropy(
                rows,
                context.output_weight,
                targets,
                bias=context.output_bias,
                reduction="sum",
            )
            with torch.no_grad():
                correct = sum(
                    (logits.argmax(dim=-1) == chunk_targets).sum()
                    for logits, chunk_targets in projection_chunks(rows, targets, context)
                )
        elif backend in {"chunked", "dense_reference"}:
            chunks = (
                [(active_projection(batch, context))]
                if backend == "dense_reference"
                else projection_chunks(rows, targets, context)
            )
            loss_sum = torch.zeros((), dtype=torch.float32, device=rows.device)
            correct = torch.zeros((), dtype=torch.int64, device=rows.device)
            for logits, chunk_targets in chunks:
                loss_sum = loss_sum + F.cross_entropy(
                    logits, chunk_targets, reduction="sum"
                )
                correct = correct + (logits.argmax(dim=-1) == chunk_targets).sum()
        else:
            raise ValueError(f"unsupported projection backend: {backend}")
        count = torch.tensor(targets.numel(), dtype=torch.int64, device=loss_sum.device)
        loss = loss_sum / resolve_divisor(count, context)
        accuracy = correct.float() / count
        return ObjectiveOutput(
            loss=loss,
            loss_sum=loss_sum.detach(),
            token_count=count,
            metrics={"token_accuracy": accuracy.detach()},
        )
