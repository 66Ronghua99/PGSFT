from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pgsft.methods.full_vocab import chunked_forward_kl, chunked_sft_base_kl
from pgsft.methods.sft_base_kl import SFTBaseKLObjective


def _fixture():
    torch.manual_seed(23)
    student = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    base = torch.randn(5, 4, dtype=torch.float64)
    weight = torch.randn(7, 4, dtype=torch.float64)
    targets = torch.tensor([0, 2, 6, 1, 4], dtype=torch.long)
    return student, base, weight, targets


@pytest.mark.parametrize("chunk_size", [1, 3, 5])
def test_sft_base_kl_matches_dense_value_and_gradient(chunk_size: int) -> None:
    student, base, weight, targets = _fixture()
    output = chunked_sft_base_kl(
        student,
        base,
        weight,
        targets,
        divisor=5,
        chunk_size=chunk_size,
    )
    output.loss.backward()
    observed = student.grad.clone()

    reference_student = student.detach().clone().requires_grad_()
    student_logp = F.log_softmax(reference_student.float() @ weight.float().t(), dim=-1)
    base_logp = F.log_softmax(base.float() @ weight.float().t(), dim=-1)
    ce = F.nll_loss(student_logp, targets, reduction="sum")
    kl = (base_logp.exp() * (base_logp - student_logp)).sum()
    reference = ((ce + 0.1 * kl) / 1.1) / 5
    reference.backward()
    torch.testing.assert_close(output.loss, reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(observed, reference_student.grad, rtol=1e-6, atol=1e-6)
    assert output.max_chunk_rows <= chunk_size


def test_equal_base_has_zero_kl_gradient_but_nonzero_ce_gradient() -> None:
    student, _, weight, targets = _fixture()
    base = student.detach().clone()
    kl = chunked_forward_kl(student, base, weight, targets, divisor=5, chunk_size=2)
    torch.testing.assert_close(kl.loss, torch.zeros_like(kl.loss), atol=1e-6, rtol=0)
    kl.loss.backward(retain_graph=True)
    torch.testing.assert_close(student.grad, torch.zeros_like(student.grad), atol=1e-6, rtol=0)
    student.grad.zero_()
    combined = chunked_sft_base_kl(student, base, weight, targets, divisor=5)
    combined.loss.backward()
    assert float(student.grad.abs().sum()) > 0


def test_formal_coefficient_and_frozen_head_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="coefficient=0.1"):
        SFTBaseKLObjective(coefficient=0.2)
    student, base, weight, targets = _fixture()
    weight.requires_grad_(True)
    with pytest.raises(ValueError, match="output head"):
        chunked_sft_base_kl(student, base, weight, targets, divisor=5)
