"""Stable objective protocol shared by all training methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch
from torch import nn


@dataclass(frozen=True)
class ObjectiveContext:
    output_weight: torch.Tensor
    output_bias: torch.Tensor | None = None
    global_token_divisor: torch.Tensor | float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveOutput:
    loss: torch.Tensor
    loss_sum: torch.Tensor
    token_count: torch.Tensor
    metrics: dict[str, torch.Tensor]


class MethodObjective(Protocol):
    name: str
    required_row_schema: Literal["sft", "paired"]

    def compute(
        self, batch: dict[str, torch.Tensor], context: ObjectiveContext
    ) -> ObjectiveOutput: ...

    def on_optimizer_step_end(self, step: int, model: nn.Module) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, value: dict[str, Any]) -> None: ...


class StatelessObjective:
    def on_optimizer_step_end(self, step: int, model: nn.Module) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, value: dict[str, Any]) -> None:
        if value:
            raise ValueError(f"{self.name} has no objective state")


def active_rows(
    batch: dict[str, torch.Tensor], context: ObjectiveContext
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather active causal hidden rows before any output projection."""
    hidden = batch.get("hidden_states")
    labels = batch.get("labels")
    if hidden is None or labels is None:
        raise ValueError("batch requires hidden_states and labels")
    if hidden.ndim != 3 or labels.ndim != 2 or hidden.shape[:2] != labels.shape:
        raise ValueError("expected hidden_states [B,L,H] and labels [B,L]")
    shifted_hidden = hidden[:, :-1, :]
    shifted_labels = labels[:, 1:].to(hidden.device)
    active = shifted_labels != -100
    if not bool(active.any()):
        raise ValueError("batch has no active causal targets")
    targets = shifted_labels[active]
    vocab = context.output_weight.shape[0]
    if int(targets.min()) < 0 or int(targets.max()) >= vocab:
        raise ValueError("target is outside the vocabulary")
    return shifted_hidden[active], targets


def projection_chunks(
    rows: torch.Tensor,
    targets: torch.Tensor,
    context: ObjectiveContext,
):
    """Yield bounded full-vocabulary projections without changing semantics."""
    chunk_size = context.extras.get("projection_chunk_size", 256)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("projection_chunk_size must be a positive integer")
    for start in range(0, rows.shape[0], chunk_size):
        end = min(start + chunk_size, rows.shape[0])
        logits = rows[start:end] @ context.output_weight.t()
        if context.output_bias is not None:
            logits = logits + context.output_bias
        yield logits.float(), targets[start:end]


def active_projection(
    batch: dict[str, torch.Tensor], context: ObjectiveContext
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense reference projection, retained for semantic parity tests."""
    rows, targets = active_rows(batch, context)
    logits = rows @ context.output_weight.t()
    if context.output_bias is not None:
        logits = logits + context.output_bias
    return logits.float(), targets


def resolve_divisor(
    token_count: torch.Tensor, context: ObjectiveContext
) -> torch.Tensor:
    divisor = context.global_token_divisor
    if divisor is None:
        return token_count.to(dtype=torch.float32)
    value = torch.as_tensor(divisor, dtype=torch.float32, device=token_count.device)
    if value.numel() != 1 or not bool(torch.isfinite(value)) or float(value) <= 0:
        raise ValueError("global token divisor must be finite and positive")
    return value
