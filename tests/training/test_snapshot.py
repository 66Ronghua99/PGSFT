from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from pgsft.methods.base import ObjectiveContext
from pgsft.methods.psft import PSFTObjective
from pgsft.training.snapshot import AdapterSnapshot, tensor_state_sha256


class TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.lora_A = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.lora_B = nn.Parameter(torch.tensor([3.0]))


def test_snapshot_initializes_before_step_one_and_refreshes_only_at_eight() -> None:
    model = TinyAdapter()
    objective = PSFTObjective()
    objective.initialize_snapshot(model)
    assert objective.snapshot is not None
    initial_digest = objective.snapshot.tensor_sha256
    with torch.no_grad():
        model.lora_A.add_(5)
    for step in range(1, 8):
        objective.on_optimizer_step_end(step, model)
        assert objective.snapshot.tensor_sha256 == initial_digest
        assert objective.snapshot.source_step == 0
    objective.on_optimizer_step_end(8, model)
    assert objective.snapshot.tensor_sha256 != initial_digest
    assert objective.snapshot.source_step == 8
    assert objective.snapshot.refresh_history == (8,)
    assert all(not tensor.requires_grad for tensor in objective.snapshot.tensors.values())


def test_snapshot_digest_is_rank_invariant_and_detects_drift() -> None:
    state = {"lora_A": torch.tensor([[1.0, 2.0]]), "lora_B": torch.tensor([3.0])}
    snapshot = AdapterSnapshot.initialize(state)
    same = tensor_state_sha256({key: value.clone() for key, value in reversed(state.items())})
    snapshot.assert_distributed_identity([same, same])
    with pytest.raises(RuntimeError, match="differs across ranks"):
        snapshot.assert_distributed_identity([same, "0" * 64])


def test_save_resume_preserves_next_loss_and_rejects_missing_or_wrong_step() -> None:
    model = TinyAdapter()
    original = PSFTObjective()
    original.initialize_snapshot(model)
    for step in range(1, 9):
        original.on_optimizer_step_end(step, model)
    saved = copy.deepcopy(original.state_dict())
    resumed = PSFTObjective()
    resumed.load_state_dict(saved, completed_step=8)
    assert resumed.snapshot is not None and original.snapshot is not None
    old_original = torch.cat(
        [tensor.reshape(-1) for tensor in original.snapshot.tensors.values()]
    )[:2]
    old_resumed = torch.cat(
        [tensor.reshape(-1) for tensor in resumed.snapshot.tensors.values()]
    )[:2]
    student = torch.tensor([-0.4, -0.2], requires_grad=True)
    first = original.compute(
        {"student_logp": student, "old_logp": old_original},
        ObjectiveContext(torch.empty(0)),
    ).loss
    second = resumed.compute(
        {"student_logp": student.detach().clone(), "old_logp": old_resumed},
        ObjectiveContext(torch.empty(0)),
    ).loss
    assert first.detach().cpu().numpy().tobytes() == second.detach().cpu().numpy().tobytes()

    with pytest.raises(ValueError, match="source step mismatch"):
        PSFTObjective().load_state_dict(saved, completed_step=16)
    missing = dict(saved)
    missing.pop("snapshot")
    with pytest.raises(ValueError, match="invalid PSFT objective state"):
        PSFTObjective().load_state_dict(missing, completed_step=8)


def test_resume_rejects_snapshot_digest_corruption() -> None:
    snapshot = AdapterSnapshot.initialize({"lora_A": torch.tensor([1.0])})
    value = snapshot.state_dict()
    value["tensors"]["lora_A"].add_(1)
    with pytest.raises(ValueError, match="digest mismatch"):
        AdapterSnapshot.load_state_dict(value, completed_step=0)
