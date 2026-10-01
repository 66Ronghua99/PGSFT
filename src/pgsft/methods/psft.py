"""Upper-clipped Proximal SFT objective with refreshable old policy."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from pgsft.methods.base import ObjectiveContext, ObjectiveOutput, StatelessObjective
from pgsft.training.snapshot import AdapterSnapshot, trainable_adapter_state


class PSFTObjective(StatelessObjective):
    name = "psft"
    required_row_schema = "sft"

    def __init__(self, epsilon_high: float = 0.28, refresh_steps: int = 8):
        if epsilon_high != 0.28:
            raise ValueError("formal PSFT requires epsilon_high=0.28")
        if refresh_steps != 8:
            raise ValueError("formal PSFT requires refresh_steps=8")
        self.epsilon_high = epsilon_high
        self.refresh_steps = refresh_steps
        self.snapshot: AdapterSnapshot | None = None

    def compute(
        self, batch: dict[str, torch.Tensor], context: ObjectiveContext
    ) -> ObjectiveOutput:
        current = batch.get("student_logp")
        old = batch.get("old_logp")
        if current is None or old is None or current.shape != old.shape or current.ndim != 1:
            raise ValueError("PSFT requires matching rank-1 student_logp and old_logp")
        if current.numel() == 0 or not bool(torch.isfinite(current).all()) or not bool(torch.isfinite(old).all()):
            raise ValueError("PSFT log probabilities must be non-empty and finite")
        difference = (current - old.detach()).clamp(min=-20.0, max=20.0)
        ratio = difference.exp()
        clipped = torch.clamp(ratio, max=1.0 + self.epsilon_high)
        token_objective = torch.minimum(ratio, clipped)
        token_losses = -token_objective
        loss_sum = token_losses.sum()
        count = torch.tensor(current.numel(), dtype=torch.int64, device=current.device)
        divisor = context.global_token_divisor
        if divisor is None:
            divisor = count
        denominator = torch.as_tensor(divisor, dtype=torch.float32, device=current.device)
        if denominator.numel() != 1 or float(denominator) <= 0:
            raise ValueError("PSFT divisor must be positive")
        return ObjectiveOutput(
            loss=loss_sum / denominator,
            loss_sum=loss_sum.detach(),
            token_count=count,
            metrics={
                "ratio_mean": ratio.mean().detach(),
                "upper_clip_fraction": (ratio > 1.0 + self.epsilon_high)
                .float()
                .mean()
                .detach(),
            },
        )

    def initialize_snapshot(self, model: nn.Module) -> None:
        if self.snapshot is not None:
            raise RuntimeError("old-policy snapshot is already initialized")
        self.snapshot = AdapterSnapshot.initialize(
            trainable_adapter_state(model), refresh_steps=self.refresh_steps
        )

    def on_optimizer_step_end(self, step: int, model: nn.Module) -> None:
        if self.snapshot is None:
            raise RuntimeError("old-policy snapshot must be initialized before step 1")
        self.snapshot = self.snapshot.maybe_refresh(step, trainable_adapter_state(model))

    def state_dict(self) -> dict[str, Any]:
        if self.snapshot is None:
            raise RuntimeError("old-policy snapshot is not initialized")
        return {
            "epsilon_high": self.epsilon_high,
            "refresh_steps": self.refresh_steps,
            "snapshot": self.snapshot.state_dict(),
        }

    def load_state_dict(self, value: dict[str, Any], *, completed_step: int = 0) -> None:
        if set(value) != {"epsilon_high", "refresh_steps", "snapshot"}:
            raise ValueError("invalid PSFT objective state")
        if value["epsilon_high"] != self.epsilon_high or value["refresh_steps"] != self.refresh_steps:
            raise ValueError("PSFT objective configuration drift on resume")
        self.snapshot = AdapterSnapshot.load_state_dict(
            value["snapshot"], completed_step=completed_step
        )
