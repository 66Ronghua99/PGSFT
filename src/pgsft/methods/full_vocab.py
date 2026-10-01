"""Bounded exact full-vocabulary losses for Base anchoring and PG-SFT."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FullVocabOutput:
    loss: torch.Tensor
    loss_sum: torch.Tensor
    token_count: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    max_chunk_rows: int


def _validate(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    output_weight: torch.Tensor,
    targets: torch.Tensor,
    divisor: float | torch.Tensor,
    chunk_size: int,
) -> float:
    if student_hidden.ndim != 2 or teacher_hidden.ndim != 2 or output_weight.ndim != 2:
        raise ValueError("hidden states and output weight must be rank-2")
    if student_hidden.shape != teacher_hidden.shape or student_hidden.shape[1] != output_weight.shape[1]:
        raise ValueError("student, teacher, and output weight shapes do not align")
    if student_hidden.shape[0] <= 0:
        raise ValueError("full-vocabulary objective requires at least one row")
    if not student_hidden.dtype.is_floating_point or not teacher_hidden.dtype.is_floating_point or not output_weight.dtype.is_floating_point:
        raise TypeError("hidden states and output weight must be floating point")
    if teacher_hidden.requires_grad:
        raise ValueError("teacher hidden states must be detached")
    if output_weight.requires_grad:
        raise ValueError("output head must remain frozen")
    if targets.dtype != torch.long or targets.ndim != 1 or targets.shape[0] != student_hidden.shape[0]:
        raise ValueError("targets must be a long vector with one ID per row")
    if student_hidden.device != teacher_hidden.device or student_hidden.device != output_weight.device or student_hidden.device != targets.device:
        raise ValueError("all full-vocabulary tensors must share a device")
    if int(targets.min()) < 0 or int(targets.max()) >= output_weight.shape[0]:
        raise ValueError("target is outside the complete vocabulary")
    if not all(bool(torch.isfinite(value).all()) for value in (student_hidden, teacher_hidden, output_weight)):
        raise ValueError("full-vocabulary inputs contain non-finite values")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 1 <= chunk_size <= 1024:
        raise ValueError("chunk_size must satisfy 1 <= chunk_size <= 1024")
    divisor_value = float(torch.as_tensor(divisor).item())
    if not math.isfinite(divisor_value) or divisor_value <= 0:
        raise ValueError("divisor must be finite and positive")
    return divisor_value


def chunked_forward_kl(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    output_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    divisor: float | torch.Tensor,
    chunk_size: int = 128,
) -> FullVocabOutput:
    """Compute exact forward KL(q_teacher || q_student) over the full vocabulary."""
    denominator = _validate(
        student_hidden, teacher_hidden, output_weight, targets, divisor, chunk_size
    )
    total = torch.zeros((), dtype=torch.float32, device=student_hidden.device)
    teacher_nll = torch.zeros_like(total)
    student_nll = torch.zeros_like(total)
    max_rows = 0
    weight = output_weight.detach().float()
    for start in range(0, targets.numel(), chunk_size):
        end = min(start + chunk_size, targets.numel())
        max_rows = max(max_rows, end - start)
        teacher_logits = teacher_hidden[start:end].float() @ weight.t()
        student_logits = student_hidden[start:end].float() @ weight.t()
        teacher_logp = F.log_softmax(teacher_logits, dim=-1)
        student_logp = F.log_softmax(student_logits, dim=-1)
        teacher_p = teacher_logp.exp()
        total = total + (teacher_p * (teacher_logp - student_logp)).sum()
        local_targets = targets[start:end, None]
        teacher_nll = teacher_nll - teacher_logp.gather(1, local_targets).sum()
        student_nll = student_nll - student_logp.gather(1, local_targets).sum()
    count = torch.tensor(targets.numel(), dtype=torch.int64, device=targets.device)
    return FullVocabOutput(
        loss=total / denominator,
        loss_sum=total,
        token_count=count,
        diagnostics={"teacher_nll_sum": teacher_nll, "student_nll_sum": student_nll},
        max_chunk_rows=max_rows,
    )


def chunked_sft_base_kl(
    student_hidden: torch.Tensor,
    base_hidden: torch.Tensor,
    output_weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    divisor: float | torch.Tensor,
    coefficient: float = 0.1,
    chunk_size: int = 128,
) -> FullVocabOutput:
    if coefficient != 0.1:
        raise ValueError("formal SFT+Base-KL requires coefficient=0.1")
    denominator = _validate(
        student_hidden, base_hidden, output_weight, targets, divisor, chunk_size
    )
    ce_sum = torch.zeros((), dtype=torch.float32, device=student_hidden.device)
    kl_sum = torch.zeros_like(ce_sum)
    max_rows = 0
    weight = output_weight.detach().float()
    for start in range(0, targets.numel(), chunk_size):
        end = min(start + chunk_size, targets.numel())
        max_rows = max(max_rows, end - start)
        student_logits = student_hidden[start:end].float() @ weight.t()
        base_logits = base_hidden[start:end].float() @ weight.t()
        student_logp = F.log_softmax(student_logits, dim=-1)
        base_logp = F.log_softmax(base_logits, dim=-1)
        ce_sum = ce_sum + F.nll_loss(
            student_logp, targets[start:end], reduction="sum"
        )
        base_p = base_logp.exp()
        kl_sum = kl_sum + (base_p * (base_logp - student_logp)).sum()
    total = (ce_sum + coefficient * kl_sum) / (1.0 + coefficient)
    count = torch.tensor(targets.numel(), dtype=torch.int64, device=targets.device)
    return FullVocabOutput(
        loss=total / denominator,
        loss_sum=total,
        token_count=count,
        diagnostics={"ce_sum": ce_sum, "base_kl_sum": kl_sum},
        max_chunk_rows=max_rows,
    )


def chunked_pg_sft_kl(
    student_hidden: torch.Tensor,
    base_hidden: torch.Tensor,
    privileged_hidden: torch.Tensor,
    output_weight: torch.Tensor,
    targets: torch.Tensor,
    turn_ids: torch.Tensor,
    *,
    divisor: float | torch.Tensor,
    chunk_size: int = 128,
) -> FullVocabOutput:
    denominator = _validate(
        student_hidden, base_hidden, output_weight, targets, divisor, chunk_size
    )
    if privileged_hidden.shape != student_hidden.shape or privileged_hidden.requires_grad:
        raise ValueError("privileged hidden states must be detached and match student rows")
    if privileged_hidden.device != student_hidden.device or not bool(torch.isfinite(privileged_hidden).all()):
        raise ValueError("privileged hidden states must be finite and share the student device")
    if turn_ids.dtype != torch.long or turn_ids.shape != targets.shape or turn_ids.device != targets.device:
        raise ValueError("turn IDs must be a long vector matching targets")
    unique = torch.unique(turn_ids, sorted=True)
    if unique.numel() == 0 or not torch.equal(unique, torch.arange(unique.numel(), device=turn_ids.device)):
        raise ValueError("turn IDs must be dense non-negative IDs")
    turn_count = unique.numel()
    weight = output_weight.detach().float()
    base_y = torch.empty(targets.shape, dtype=torch.float32, device=targets.device)
    privileged_y = torch.empty_like(base_y)
    base_nll = torch.zeros((), dtype=torch.float32, device=targets.device)
    privileged_nll = torch.zeros_like(base_nll)
    with torch.no_grad():
        for start in range(0, targets.numel(), chunk_size):
            end = min(start + chunk_size, targets.numel())
            local = targets[start:end, None]
            base_logp = F.log_softmax(base_hidden[start:end].float() @ weight.t(), dim=-1)
            privileged_logp = F.log_softmax(
                privileged_hidden[start:end].float() @ weight.t(), dim=-1
            )
            base_local = base_logp.gather(1, local).squeeze(1)
            privileged_local = privileged_logp.gather(1, local).squeeze(1)
            base_y[start:end] = base_local.exp()
            privileged_y[start:end] = privileged_local.exp()
            base_nll = base_nll - base_local.sum()
            privileged_nll = privileged_nll - privileged_local.sum()
    signed_gain = privileged_y - base_y
    positive = signed_gain.clamp_min(0)
    negative = (-signed_gain).clamp_min(0)
    headroom = 1.0 - base_y
    gain_by_turn = torch.zeros(turn_count, dtype=torch.float32, device=targets.device).scatter_add_(0, turn_ids, positive)
    headroom_by_turn = torch.zeros_like(gain_by_turn).scatter_add_(0, turn_ids, headroom)
    alpha_turn = torch.where(
        headroom_by_turn > 0,
        gain_by_turn / headroom_by_turn.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(gain_by_turn),
    ).clamp(0, 1).detach()
    alpha_token = alpha_turn[turn_ids]
    total = torch.zeros((), dtype=torch.float32, device=targets.device)
    base_kl = torch.zeros_like(total)
    student_nll = torch.zeros_like(total)
    max_rows = 0
    for start in range(0, targets.numel(), chunk_size):
        end = min(start + chunk_size, targets.numel())
        max_rows = max(max_rows, end - start)
        base_logp = F.log_softmax(base_hidden[start:end].float() @ weight.t(), dim=-1)
        student_logp = F.log_softmax(student_hidden[start:end].float() @ weight.t(), dim=-1)
        base_p = base_logp.exp()
        local_targets = targets[start:end, None]
        local_alpha = alpha_token[start:end, None]
        projected = base_p * (1.0 - local_alpha)
        projected = projected.scatter_add(1, local_targets, local_alpha)
        total = total + (torch.special.xlogy(projected, projected) - projected * student_logp).sum()
        base_kl = base_kl + (base_p * (base_logp - student_logp)).sum()
        student_nll = student_nll - student_logp.gather(1, local_targets).sum()
    conservation = (alpha_turn * headroom_by_turn - gain_by_turn).abs().max()
    count = torch.tensor(targets.numel(), dtype=torch.int64, device=targets.device)
    diagnostics = {
        "positive_gain_sum": positive.sum(),
        "negative_gain_sum": negative.sum(),
        "headroom_sum": headroom.sum(),
        "turn_alpha_sum": alpha_turn.sum(),
        "token_alpha_sum": alpha_token.sum(),
        "base_kl_sum": base_kl,
        "base_nll_sum": base_nll,
        "privileged_nll_sum": privileged_nll,
        "student_nll_sum": student_nll,
        "zero_turn_count": (alpha_turn == 0).sum(),
        "conservation_error": conservation,
        "turn_alphas": alpha_turn,
        "token_alphas": alpha_token,
    }
    return FullVocabOutput(total / denominator, total, count, diagnostics, max_rows)
