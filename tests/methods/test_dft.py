from __future__ import annotations

import torch

from pgsft.methods.base import ObjectiveContext
from pgsft.methods.dft import DFTObjective


def test_dft_uses_detached_q_weight_and_supervised_token_divisor() -> None:
    torch.manual_seed(11)
    hidden = torch.randn(1, 5, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[-100, 1, -100, 4, 2]])
    output = DFTObjective().compute(
        {"hidden_states": hidden, "labels": labels},
        ObjectiveContext(weight, global_token_divisor=3),
    )
    output.loss.backward()
    observed_hidden = hidden.grad.clone()
    observed_weight = weight.grad.clone()

    hidden_ref = hidden.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    logits = hidden_ref[:, :-1] @ weight_ref.t()
    active = labels[:, 1:] != -100
    targets = labels[:, 1:][active]
    log_probs = torch.log_softmax(logits[active].float(), dim=-1)
    log_q = log_probs.gather(1, targets[:, None]).squeeze(1)
    probabilities = log_q.exp().detach()
    reference = (probabilities * -log_q).sum() / 3
    reference.backward()
    torch.testing.assert_close(output.loss, reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(observed_hidden, hidden_ref.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(observed_weight, weight_ref.grad, rtol=1e-6, atol=1e-6)
    assert output.token_count.item() == 3
    assert not torch.isclose(output.metrics["weight_sum"], torch.tensor(3.0))


def test_dft_does_not_normalize_by_weight_sum() -> None:
    hidden = torch.zeros(1, 3, 2)
    weight = torch.zeros(4, 2)
    labels = torch.tensor([[-100, 1, 2]])
    output = DFTObjective().compute(
        {"hidden_states": hidden, "labels": labels},
        ObjectiveContext(weight),
    )
    # q=1/4 and NLL=log(4), then divide the weighted sum by two tokens.
    expected = 0.25 * torch.log(torch.tensor(4.0))
    torch.testing.assert_close(output.loss, expected)
