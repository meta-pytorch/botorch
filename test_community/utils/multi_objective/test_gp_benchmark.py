# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""GP query accounting and separate predictive/true front-audit endpoints."""

import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from botorch.models import SingleTaskGP
from torch.quasirandom import SobolEngine

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "notebooks_community/pareto_decision_making/experiments/benchmark_gp_fpbi.py"
)
_SPEC = importlib.util.spec_from_file_location("gp_benchmark_under_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_EXPERIMENT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _EXPERIMENT
_SPEC.loader.exec_module(_EXPERIMENT)


def tiny_model(dimensions=1):
    """A real GP with default hyperparameters; these tests never fit it."""
    X = SobolEngine(dimensions, scramble=True, seed=13).draw(6).double()
    x = X.mean(-1)
    Y = torch.stack((x.square(), (1 - x).square()), -1)
    return SingleTaskGP(X, Y, train_Yvar=torch.full_like(Y, 0.01)).eval()


@pytest.mark.parametrize("dimensions", [1, 3])
def test_counted_gp_tracks_batched_decision_vectors_not_objective_components(
    dimensions,
):
    fitted = tiny_model(dimensions)
    model = _EXPERIMENT.CountedGP(fitted).eval()
    X = torch.full((2, 3, dimensions), 0.4, dtype=torch.double)
    posterior = model.posterior(X, observation_noise=True)
    with torch.no_grad():
        expected = fitted.posterior(X, observation_noise=True)
    torch.testing.assert_close(posterior.mean, expected.mean)
    assert posterior.mean.shape == torch.Size([2, 3, 2])
    assert model.num_outputs == 2 and model.batch_shape == fitted.batch_shape
    assert model.counter.posterior_calls == model.counter.objective_calls == 1
    assert model.counter.point_evaluations == 6
    model.posterior(torch.full((4, dimensions), 0.6, dtype=torch.double))
    assert model.counter.posterior_calls == model.counter.objective_calls == 2
    assert model.counter.point_evaluations == 10


@pytest.mark.parametrize("case", _EXPERIMENT.CASES, ids=lambda case: case.name)
def test_heldout_sobol_designs_exclude_fixed_training_anchors(case):
    X = _EXPERIMENT.feasible_designs(case, 16, 17, anchors=False)
    expected = SobolEngine(case.dimensions, scramble=True, seed=17).draw(16).double()
    if case.constrained:
        expected = 0.2 + 0.6 * expected
    torch.testing.assert_close(X, expected, rtol=0, atol=0)
    training = _EXPERIMENT.feasible_designs(case, 16, 17)
    anchor_count = 2 if case.dimensions == 1 else 3
    assert torch.cdist(X, training[:anchor_count]).min() > 0
    assert torch.equal(X[anchor_count:], training[anchor_count:])
    if case.constrained:
        assert case.constraints(X).min() >= 0


def test_fit_diagnostics_request_independent_unanchored_designs(monkeypatch):
    calls = []
    original = _EXPERIMENT.feasible_designs

    def record_designs(case, count, seed, *, anchors=True):
        calls.append((count, seed, anchors))
        return original(case, count, seed, anchors=anchors)

    monkeypatch.setattr(_EXPERIMENT, "feasible_designs", record_designs)
    # Exercise data construction and posterior audits without an optimizer fit.
    monkeypatch.setattr(_EXPERIMENT, "fit_gpytorch_mll", lambda *args, **kwargs: None)
    model, diagnostics = _EXPERIMENT.fit_case(
        _EXPERIMENT.CASES[0],
        7,
        {"train_2d": 6, "train_3d": 8, "noise_sd": 0.02, "fit_maxiter": 1},
    )
    assert calls == [(6, 7, True), (512, 4007, False)]
    assert model.train_inputs[0].shape[-2:] == torch.Size([6, 1])
    assert diagnostics["training_points"] == 6
    assert math.isfinite(diagnostics["heldout_normalized_rmse"])
    assert 0 <= diagnostics["heldout_latent_95_coverage"] <= 1


def constant_uncertainty_model():
    return SimpleNamespace(
        posterior=lambda X: SimpleNamespace(
            variance=torch.full_like(X.expand(-1, 2), 0.04)
        )
    )


def test_front_audit_keeps_optimistic_predictions_separate_from_true_quality():
    case = _EXPERIMENT.CASES[0]
    X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    truth = case.costs(X)
    measured = {"X": X, "Y": 0.5 * truth}
    audited = _EXPERIMENT.audit_front(case, constant_uncertainty_model(), measured, 3)
    reference_hv = 1.21 - 1 / 6
    assert audited["true_hv_ratio"] == pytest.approx(0.7725 / reference_hv)
    assert audited["predicted_hv_ratio"] == pytest.approx(1.100625 / reference_hv)
    assert audited["predicted_hv_ratio"] > 1 > audited["true_hv_ratio"]
    assert audited["returned_points"] == audited["audit_true_evaluations"] == 3
    assert audited["returned_mean_normalized_sd"] == pytest.approx(0.2)
    assert audited["returned_normalized_rmse"] == pytest.approx(math.sqrt(0.53125 / 6))
    torch.testing.assert_close(torch.tensor(audited["true_Y"]), truth.float())
    torch.testing.assert_close(
        torch.tensor(audited["predicted_Y"]), (truth * 0.5).float()
    )


def test_front_filter_uses_predictions_before_true_outcome_audit():
    case = _EXPERIMENT.CASES[0]
    X = torch.tensor([[0.25], [0.5], [0.75]], dtype=torch.double)
    predicted = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.2, 0.05]], dtype=torch.double)
    audited = _EXPERIMENT.audit_front(
        case, constant_uncertainty_model(), {"X": X, "Y": predicted}, 3
    )
    assert audited["raw_rows"] == 3 and audited["returned_points"] == 2
    assert audited["X"] == [[0.25], [0.75]]
    # All original designs are truly nondominated; the middle is excluded
    # because its prediction is dominated, never restored using oracle truth.
    assert audited["true_nondominated_points"] == 2


def test_generation_failure_preserves_completed_gp_query_effort(monkeypatch):
    def fail_after_query(model, bounds, **kwargs):
        model.posterior(torch.full((2, 3, 1), 0.5, dtype=torch.double))
        warnings.warn("partial solver warning", RuntimeWarning)
        raise ValueError("controlled generation failure")

    monkeypatch.setattr(_EXPERIMENT, "sample_model_fpbi", fail_after_query)
    config = {"steps_2d": 1, "steps_3d": 1, "restarts": 1, "maxiter": 2}
    with pytest.raises(_EXPERIMENT.GenerationFailure) as failure:
        _EXPERIMENT.generate(_EXPERIMENT.CASES[0], tiny_model(), 3, "fpbi", config)
    accounting = failure.value.accounting
    assert isinstance(failure.value.__cause__, ValueError)
    assert accounting["posterior_calls"] == accounting["objective_calls"] == 1
    assert accounting["point_evaluations"] == 6
    assert accounting["generation_seconds"] >= 0
    assert accounting["warnings"] == ["partial solver warning"]


def test_fixed_budget_arm_survives_missing_fpbi_budget(monkeypatch, tmp_path):
    calls = []

    def fake_fit(case, seed, config):
        return tiny_model(), {
            "training_points": 6,
            "training_sha256": "shared-test-training-data",
            "fit_seconds": 0.0,
            "heldout_normalized_rmse": 0.1,
            "heldout_latent_95_coverage": 0.9,
        }

    def fake_generation(case, model, seed, method, config, generations):
        calls.append(method)
        if method == "fpbi":
            raise _EXPERIMENT.GenerationFailure(
                ValueError("controlled FPBI failure"),
                {
                    "generation_seconds": 0.01,
                    "point_evaluations": 6,
                    "posterior_calls": 2,
                    "objective_calls": 2,
                    "warnings": [],
                },
            )
        assert method == "nsga_fixed"
        X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
        return {
            "X": X,
            "Y": case.costs(X),
            "generation_seconds": 0.01,
            "point_evaluations": 3,
            "posterior_calls": 1,
            "objective_calls": 1,
            "diagnostics": [],
            "failed_rays": None,
            "warnings": [],
        }

    monkeypatch.setattr(_EXPERIMENT, "fit_case", fake_fit)
    monkeypatch.setattr(_EXPERIMENT, "generate", fake_generation)
    output = tmp_path / "partial-failure"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_gp_fpbi.py",
            "--output-dir",
            str(output),
            "--cases",
            "convex_2d",
            "--seeds",
            "0",
        ],
    )
    _EXPERIMENT.main()
    assert calls == ["fpbi", "nsga_fixed"]
    rows = {
        row["method"]: row for row in json.loads((output / "runs.json").read_text())
    }
    assert rows["fpbi"]["error"] is not None
    assert rows["fpbi"]["point_evaluations"] == 6
    assert rows["nsga_fixed"]["error"] is None
    assert rows["nsga_fixed"]["budget_ratio"] is None
    assert "no measured budget" in rows["nsga_matched"]["error"]
    completion = json.loads((output / "completion.json").read_text())
    assert completion["runs"] == 3 and completion["failures"] == 2
