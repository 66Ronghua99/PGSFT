from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from pgsft.methods.full_vocab import chunked_forward_kl, chunked_pg_sft_kl
from pgsft.training.teacher import FixedBaseTeacher


def _fixture():
    torch.manual_seed(29)
    student = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    base = torch.randn(6, 4, dtype=torch.float64)
    privileged = base + torch.randn(6, 4, dtype=torch.float64) * 0.5
    weight = torch.randn(8, 4, dtype=torch.float64)
    targets = torch.tensor([0, 3, 2, 7, 1, 4], dtype=torch.long)
    turns = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    return student, base, privileged, weight, targets, turns


def _dense_reference(student, base, privileged, weight, targets, turns):
    student_logp = F.log_softmax(student.float() @ weight.float().t(), dim=-1)
    base_logp = F.log_softmax(base.float() @ weight.float().t(), dim=-1)
    privileged_logp = F.log_softmax(privileged.float() @ weight.float().t(), dim=-1)
    base_p = base_logp.exp()
    base_y = base_logp.gather(1, targets[:, None]).squeeze(1).exp()
    privileged_y = privileged_logp.gather(1, targets[:, None]).squeeze(1).exp()
    positive = (privileged_y - base_y).clamp_min(0)
    headroom = 1 - base_y
    gain = torch.zeros(3).scatter_add_(0, turns, positive)
    capacity = torch.zeros(3).scatter_add_(0, turns, headroom)
    alpha_turn = torch.where(capacity > 0, gain / capacity, torch.zeros_like(gain)).detach()
    alpha = alpha_turn[turns, None]
    projected = base_p * (1 - alpha)
    projected = projected.scatter_add(1, targets[:, None], alpha)
    loss_sum = (torch.special.xlogy(projected, projected) - projected * student_logp).sum()
    return loss_sum / targets.numel(), alpha_turn


@pytest.mark.parametrize("chunk_size", [1, 2, 6])
def test_pg_sft_matches_dense_value_gradient_and_turn_allocation(chunk_size: int) -> None:
    student, base, privileged, weight, targets, turns = _fixture()
    output = chunked_pg_sft_kl(
        student,
        base,
        privileged,
        weight,
        targets,
        turns,
        divisor=targets.numel(),
        chunk_size=chunk_size,
    )
    output.loss.backward()
    observed_gradient = student.grad.clone()
    reference_student = student.detach().clone().requires_grad_()
    reference, alpha = _dense_reference(
        reference_student, base, privileged, weight, targets, turns
    )
    reference.backward()
    torch.testing.assert_close(output.loss, reference, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(observed_gradient, reference_student.grad, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(output.diagnostics["turn_alphas"], alpha)
    assert bool(((alpha >= 0) & (alpha <= 1)).all())
    assert float(output.diagnostics["conservation_error"]) <= 2e-6
    assert output.max_chunk_rows <= chunk_size


def test_zero_gain_reduces_to_base_forward_kl() -> None:
    student, base, _, weight, targets, turns = _fixture()
    pg = chunked_pg_sft_kl(
        student, base, base.clone(), weight, targets, turns, divisor=6, chunk_size=2
    )
    anchored = chunked_forward_kl(
        student, base, weight, targets, divisor=6, chunk_size=3
    )
    torch.testing.assert_close(pg.loss, anchored.loss, rtol=1e-6, atol=1e-6)
    assert torch.count_nonzero(pg.diagnostics["turn_alphas"]) == 0


def test_alpha_near_one_reduces_to_ce_up_to_target_entropy_constant() -> None:
    student = torch.tensor([[0.2, -0.1]], requires_grad=True)
    base = torch.zeros_like(student).detach()
    privileged = torch.tensor([[80.0, 0.0]])
    weight = torch.eye(2)
    targets = torch.tensor([0])
    turns = torch.tensor([0])
    output = chunked_pg_sft_kl(
        student, base, privileged, weight, targets, turns, divisor=1, chunk_size=1
    )
    ce = F.cross_entropy(student @ weight.t(), targets)
    torch.testing.assert_close(output.diagnostics["turn_alphas"], torch.ones(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(output.loss, ce, atol=1e-5, rtol=1e-5)


class _DummyPeft(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.ones(1), requires_grad=False)
        self.lora_adapter = nn.Parameter(torch.ones(1), requires_grad=True)
        self.disabled = False

    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False


def test_fixed_base_teacher_restores_mode_and_validates_hint_boundaries() -> None:
    model = _DummyPeft()
    model.train()
    teacher = FixedBaseTeacher()

    def forward(*, model, input_ids):
        assert not model.training and model.disabled
        return input_ids.float().unsqueeze(-1)

    hidden = teacher.forward_hidden(model, forward, input_ids=torch.tensor([1, 2]))
    assert not hidden.requires_grad and model.training and not model.disabled
    teacher.validate_paired_lanes(
        student_input_ids=torch.tensor([9, 1, 2]),
        privileged_input_ids=torch.tensor([8, 8, 1, 2]),
        student_positions=torch.tensor([0, 1]),
        privileged_positions=torch.tensor([1, 2]),
        target_ids=torch.tensor([1, 2]),
        privileged_hint_positions=torch.tensor([0]),
    )
    with pytest.raises(ValueError, match="overlap"):
        teacher.validate_paired_lanes(
            student_input_ids=torch.tensor([9, 1, 2]),
            privileged_input_ids=torch.tensor([8, 8, 1, 2]),
            student_positions=torch.tensor([0, 1]),
            privileged_positions=torch.tensor([1, 2]),
            target_ids=torch.tensor([1, 2]),
            privileged_hint_positions=torch.tensor([2]),
        )
