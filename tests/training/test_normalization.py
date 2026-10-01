from __future__ import annotations

import torch

from pgsft.methods.base import ObjectiveContext
from pgsft.methods.sft import SFTObjective
from pgsft.training.normalization import GlobalTokenNormalizer


def _batch(active: list[int], hidden: torch.Tensor) -> dict[str, torch.Tensor]:
    labels = torch.full((1, hidden.shape[1]), -100, dtype=torch.long)
    for position, target in enumerate(active, start=1):
        labels[0, position] = target
    return {"hidden_states": hidden, "labels": labels}


def test_two_rank_uneven_microbatches_match_dense_update_value_and_gradient() -> None:
    torch.manual_seed(19)
    rank_hidden = [
        [torch.randn(1, 4, 3, dtype=torch.float64), torch.randn(1, 4, 3, dtype=torch.float64)],
        [torch.randn(1, 4, 3, dtype=torch.float64), torch.randn(1, 4, 3, dtype=torch.float64)],
    ]
    batches = [
        [_batch([1], rank_hidden[0][0]), _batch([2, 3], rank_hidden[0][1])],
        [_batch([4, 1, 2], rank_hidden[1][0]), _batch([3], rank_hidden[1][1])],
    ]
    normalizer = GlobalTokenNormalizer()
    global_count = sum(normalizer.count(batch) for rank in batches for batch in rank)
    plans = [
        normalizer.prepare(
            rank,
            world_size=2,
            all_reduce_sum=lambda _local, total=global_count: total,
        )
        for rank in batches
    ]
    assert [plan.local_token_count for plan in plans] == [3, 4]
    assert all(plan.global_token_count == 7 for plan in plans)
    assert all(plan.divisor == 7 / 4 for plan in plans)

    weight = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    objective = SFTObjective()
    # Emulate Trainer's accumulation mean and DDP's rank mean.
    distributed_loss = sum(
        objective.compute(
            batch,
            ObjectiveContext(weight, global_token_divisor=plans[rank_index].divisor),
        ).loss
        for rank_index, rank in enumerate(batches)
        for batch in rank
    ) / (2 * 2)
    distributed_loss.backward()
    observed_gradient = weight.grad.clone()

    weight_ref = weight.detach().clone().requires_grad_()
    reference_sum = torch.zeros((), dtype=torch.float32)
    for rank in batches:
        for batch in rank:
            reference_sum = reference_sum + objective.compute(
                batch, ObjectiveContext(weight_ref, global_token_divisor=1)
            ).loss
    dense_reference = reference_sum / global_count
    dense_reference.backward()
    torch.testing.assert_close(distributed_loss, dense_reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(observed_gradient, weight_ref.grad, rtol=1e-6, atol=1e-6)


def test_tail_accumulation_uses_actual_microbatch_count_and_injects_once() -> None:
    batches = [
        {"labels": torch.tensor([[-100, 1, 2]])},
        {"labels": torch.tensor([[-100, 3, -100]])},
    ]
    normalizer = GlobalTokenNormalizer()
    plan = normalizer.prepare(batches, world_size=1, all_reduce_sum=lambda value: value)
    assert plan.local_microbatches == 2
    assert plan.global_token_count == 3
    assert plan.divisor == 1.5
    normalizer.inject(batches, plan)
    assert all(float(batch["global_token_divisor"]) == 1.5 for batch in batches)
