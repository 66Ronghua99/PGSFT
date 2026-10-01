"""Serializable, content-addressed old-policy adapter snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Any, Mapping

import torch


def tensor_state_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    if not tensors:
        raise ValueError("adapter snapshot cannot be empty")
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().contiguous().cpu()
        if tensor.requires_grad:
            raise ValueError("snapshot tensor must be detached")
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(struct.pack(">Q", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack(">Q", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack(">Q", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack(">Q", dimension))
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _clone_state(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone().cpu() for name, value in sorted(tensors.items())
    }


@dataclass(frozen=True)
class AdapterSnapshot:
    tensors: dict[str, torch.Tensor]
    source_step: int
    tensor_sha256: str
    refresh_history: tuple[int, ...]
    refresh_steps: int = 8

    @classmethod
    def initialize(
        cls, tensors: Mapping[str, torch.Tensor], *, refresh_steps: int = 8
    ) -> "AdapterSnapshot":
        if refresh_steps != 8:
            raise ValueError("formal PSFT requires refresh_steps=8")
        cloned = _clone_state(tensors)
        return cls(cloned, 0, tensor_state_sha256(cloned), (), refresh_steps)

    def maybe_refresh(
        self, completed_step: int, tensors: Mapping[str, torch.Tensor]
    ) -> "AdapterSnapshot":
        if completed_step <= 0:
            raise ValueError("completed optimizer step must be positive")
        if completed_step % self.refresh_steps:
            return self
        cloned = _clone_state(tensors)
        return AdapterSnapshot(
            tensors=cloned,
            source_step=completed_step,
            tensor_sha256=tensor_state_sha256(cloned),
            refresh_history=(*self.refresh_history, completed_step),
            refresh_steps=self.refresh_steps,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "tensors": _clone_state(self.tensors),
            "source_step": self.source_step,
            "tensor_sha256": self.tensor_sha256,
            "refresh_history": list(self.refresh_history),
            "refresh_steps": self.refresh_steps,
        }

    @classmethod
    def load_state_dict(
        cls, value: Mapping[str, Any], *, completed_step: int
    ) -> "AdapterSnapshot":
        required = {
            "schema_version",
            "tensors",
            "source_step",
            "tensor_sha256",
            "refresh_history",
            "refresh_steps",
        }
        if set(value) != required or value["schema_version"] != 1:
            raise ValueError("invalid adapter snapshot state schema")
        tensors = value["tensors"]
        if not isinstance(tensors, Mapping) or not all(
            isinstance(name, str) and isinstance(tensor, torch.Tensor)
            for name, tensor in tensors.items()
        ):
            raise ValueError("invalid adapter snapshot tensors")
        refresh_steps = value["refresh_steps"]
        if refresh_steps != 8:
            raise ValueError("formal PSFT requires refresh_steps=8")
        expected_source = completed_step - (completed_step % refresh_steps)
        if value["source_step"] != expected_source:
            raise ValueError(
                f"snapshot source step mismatch: {value['source_step']} != {expected_source}"
            )
        history = tuple(value["refresh_history"])
        expected_history = tuple(range(refresh_steps, expected_source + 1, refresh_steps))
        if history != expected_history:
            raise ValueError("snapshot refresh history does not match completed step")
        cloned = _clone_state(tensors)
        digest = tensor_state_sha256(cloned)
        if digest != value["tensor_sha256"]:
            raise ValueError("adapter snapshot tensor digest mismatch")
        return cls(cloned, expected_source, digest, history, refresh_steps)

    def assert_distributed_identity(self, gathered_digests: list[str]) -> None:
        if not gathered_digests or any(
            digest != self.tensor_sha256 for digest in gathered_digests
        ):
            raise RuntimeError("old-policy snapshot digest differs across ranks")


def trainable_adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    }
    if not state:
        raise ValueError("model has no trainable LoRA adapter state")
    return state
