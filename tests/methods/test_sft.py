from __future__ import annotations

import torch
import torch.nn.functional as F

from pgsft.methods.base import ObjectiveContext
from pgsft.methods.registry import get_method
from pgsft.methods.sft import SFTObjective


def _inputs():
    torch.manual_seed(7)
    hidden = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(7, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor(
        [[-100, 1, 2, -100, 3], [-100, -100, 4, 5, -100]], dtype=torch.long
    )
    return hidden, weight, bias, labels


def test_sft_matches_dense_full_logits_value_and_all_gradients() -> None:
    hidden, weight, bias, labels = _inputs()
    output = SFTObjective().compute(
        {"hidden_states": hidden, "labels": labels},
        ObjectiveContext(weight, bias, global_token_divisor=5),
    )
    output.loss.backward()
    observed = (output.loss.detach(), hidden.grad.clone(), weight.grad.clone(), bias.grad.clone())

    hidden_ref = hidden.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_()
    dense_logits = (hidden_ref[:, :-1] @ weight_ref.t() + bias_ref).float()
    reference = F.cross_entropy(
        dense_logits.reshape(-1, 7), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum"
    ) / 5
    reference.backward()
    expected = (reference.detach(), hidden_ref.grad, weight_ref.grad, bias_ref.grad)
    for actual, wanted in zip(observed, expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=1e-6, atol=1e-6)
    assert output.token_count.item() == 5


def test_registry_is_fail_closed_before_model_loading() -> None:
    assert get_method("sft").name == "sft"
    assert get_method("dft").name == "dft"
    try:
        get_method("nll_sft")
    except ValueError as error:
        assert "unknown method" in str(error)
    else:
        raise AssertionError("unknown method was accepted")
