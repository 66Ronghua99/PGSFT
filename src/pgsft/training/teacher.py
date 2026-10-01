"""Fixed-Base teacher isolation for Base-KL and PG-SFT."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

import torch
from torch import nn


class FixedBaseTeacher:
    """Run Base/privileged forwards with adapters disabled and no gradients."""

    @staticmethod
    def validate_frozen(model: nn.Module, output_weight: torch.Tensor) -> None:
        base_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if "lora_" not in name
        ]
        if not base_parameters or any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in base_parameters
        ):
            raise ValueError("fixed Base contains trainable parameters or gradients")
        if output_weight.requires_grad:
            raise ValueError("fixed output head must not require gradients")

    @contextmanager
    def isolated(self, model: nn.Module) -> Iterator[None]:
        previous_training = model.training
        trainability = {
            name: parameter.requires_grad for name, parameter in model.named_parameters()
        }
        disable = getattr(model, "disable_adapter", None)
        if not callable(disable):
            raise TypeError("fixed Base teacher requires a PEFT model with disable_adapter")
        try:
            model.eval()
            with torch.no_grad(), disable():
                yield
        finally:
            model.train(previous_training)
            current = dict(model.named_parameters())
            if set(current) != set(trainability):
                raise RuntimeError("adapter isolation changed parameter inventory")
            for name, requires_grad in trainability.items():
                if current[name].requires_grad != requires_grad:
                    raise RuntimeError(f"adapter isolation changed trainability: {name}")

    def forward_hidden(
        self,
        model: nn.Module,
        hidden_forward: Callable[..., Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        with self.isolated(model):
            output = hidden_forward(model=model, **kwargs)
        if isinstance(output, torch.Tensor):
            hidden = output
        elif hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, (tuple, list)) and output:
            hidden = output[0]
        else:
            raise TypeError("teacher forward did not return hidden states")
        if not isinstance(hidden, torch.Tensor) or hidden.requires_grad:
            raise RuntimeError("teacher hidden states must be detached tensors")
        return hidden

    @staticmethod
    def validate_paired_lanes(
        *,
        student_input_ids: torch.Tensor,
        privileged_input_ids: torch.Tensor,
        student_positions: torch.Tensor,
        privileged_positions: torch.Tensor,
        target_ids: torch.Tensor,
        privileged_hint_positions: torch.Tensor | None = None,
    ) -> None:
        for name, ids, positions in (
            ("student", student_input_ids, student_positions),
            ("privileged", privileged_input_ids, privileged_positions),
        ):
            if ids.ndim != 1 or positions.ndim != 1 or positions.shape != target_ids.shape:
                raise ValueError(f"{name} lane has invalid paired shapes")
            if positions.numel() == 0 or bool((positions[1:] <= positions[:-1]).any()):
                raise ValueError(f"{name} target positions must be non-empty and increasing")
            if int(positions.min()) < 0 or int(positions.max()) + 1 >= ids.numel():
                raise ValueError(f"{name} target position is out of range")
            if not torch.equal(ids[positions + 1], target_ids):
                raise ValueError(f"{name} lane target identity mismatch")
        if torch.equal(student_input_ids, privileged_input_ids):
            raise ValueError("privileged lane contains no HINT-conditioned input change")
        if privileged_hint_positions is not None:
            target_token_positions = privileged_positions + 1
            if any(
                int(position) in set(target_token_positions.tolist())
                for position in privileged_hint_positions
            ):
                raise ValueError("HINT tokens overlap assistant target positions")
