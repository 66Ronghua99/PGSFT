"""Exact update-global token normalization for accumulation plus DDP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


@dataclass(frozen=True)
class NormalizationPlan:
    local_token_count: int
    global_token_count: int
    local_microbatches: int
    world_size: int
    divisor: float


class GlobalTokenNormalizer:
    """Compute the per-forward divisor before an actual optimizer update."""

    @staticmethod
    def count(batch: dict[str, torch.Tensor]) -> int:
        labels = batch.get("labels")
        if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
            raise ValueError("normalization batch requires rank-2 labels")
        return int((labels[:, 1:] != -100).sum().item())

    def prepare(
        self,
        local_microbatches: Sequence[dict[str, torch.Tensor]],
        *,
        world_size: int,
        all_reduce_sum: Callable[[int], int],
    ) -> NormalizationPlan:
        if not local_microbatches:
            raise ValueError("optimizer update has no local microbatches")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        local_count = sum(self.count(batch) for batch in local_microbatches)
        global_count = int(all_reduce_sum(local_count))
        if local_count <= 0 or global_count <= 0:
            raise ValueError("optimizer update has no active targets")
        microbatches = len(local_microbatches)
        divisor = global_count / (world_size * microbatches)
        return NormalizationPlan(
            local_token_count=local_count,
            global_token_count=global_count,
            local_microbatches=microbatches,
            world_size=world_size,
            divisor=divisor,
        )

    @staticmethod
    def inject(
        local_microbatches: Sequence[dict[str, torch.Tensor]],
        plan: NormalizationPlan,
        *,
        key: str = "global_token_divisor",
    ) -> None:
        if len(local_microbatches) != plan.local_microbatches:
            raise ValueError("microbatch count drift after normalization planning")
        for batch in local_microbatches:
            if key in batch:
                raise ValueError(f"normalization key already present: {key}")
            batch[key] = torch.tensor(plan.divisor, dtype=torch.float32)
