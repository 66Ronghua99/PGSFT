from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from pgsft.methods.base import ObjectiveContext
from pgsft.methods.psft import PSFTObjective


ROOT = Path(__file__).resolve().parents[2]


def test_upstream_contract_fixture_freezes_revision_equation_and_project_cadence() -> None:
    contract = json.loads(
        (ROOT / "tests" / "fixtures" / "psft_upstream_contract.json").read_text()
    )
    assert contract["revision"] == "930e23980a723ecef5af138e6e32aa3798b1fd64"
    assert contract["loss_source"].endswith("compute_policy_loss_vanilla")
    assert contract["epsilon_high"] == 0.28
    assert contract["project_refresh_steps"] == 8
    assert contract["project_training_learning_rate"] == 1e-6
    assert contract["project_training_epochs"] == 3
    assert "old_log_prob" in contract["detach_boundary"]


def test_psft_matches_official_positive_advantage_upper_clip_value_and_gradient() -> None:
    epsilon = 0.28
    ratios = torch.tensor([0.7, 1.28, 1.8], dtype=torch.float64)
    old_logp = torch.tensor([-2.0, -1.5, -1.0], dtype=torch.float64, requires_grad=True)
    current = (old_logp.detach() + ratios.log()).requires_grad_()
    objective = PSFTObjective(epsilon_high=epsilon, refresh_steps=8)
    output = objective.compute(
        {"student_logp": current, "old_logp": old_logp},
        ObjectiveContext(torch.empty(0), global_token_divisor=3),
    )
    expected = -torch.minimum(ratios, torch.clamp(ratios, max=1 + epsilon)).sum() / 3
    expected_gradient = torch.autograd.grad(expected, ratios)[0] if ratios.requires_grad else None
    # Rebuild the official expression on current log probabilities for gradient parity.
    reference_current = current.detach().clone().requires_grad_()
    official_ratio = torch.exp((reference_current - old_logp.detach()).clamp(-20, 20))
    reference = -torch.minimum(
        official_ratio, torch.clamp(official_ratio, max=1 + epsilon)
    ).sum() / 3
    output.loss.backward()
    reference.backward()
    torch.testing.assert_close(output.loss, reference)
    torch.testing.assert_close(current.grad, reference_current.grad)
    assert old_logp.grad is None
    assert current.grad[0] < 0
    assert current.grad[2] == 0
    assert float(output.metrics["upper_clip_fraction"]) == pytest.approx(1 / 3)


def test_formal_psft_hyperparameters_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="epsilon_high"):
        PSFTObjective(epsilon_high=0.2)
    with pytest.raises(ValueError, match="refresh_steps"):
        PSFTObjective(refresh_steps=4)
