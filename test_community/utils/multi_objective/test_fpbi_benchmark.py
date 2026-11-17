# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Accounting, metric references and actual upstream comparator checks."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "notebooks_community/pareto_decision_making/experiments/benchmark_fpbi.py"
)
# Notebook experiments are repository files, not installed Python packages.
_SPEC = importlib.util.spec_from_file_location("fpbi_benchmark_under_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_EXPERIMENT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _EXPERIMENT
_SPEC.loader.exec_module(_EXPERIMENT)
CASES = _EXPERIMENT.CASES
checked_front = _EXPERIMENT.checked_front
common_cap = _EXPERIMENT.common_cap
CountedModel = _EXPERIMENT.CountedModel
matched_generations = _EXPERIMENT.matched_generations
quality_metrics = _EXPERIMENT.quality_metrics
run_fpbi = _EXPERIMENT.run_fpbi
run_nsga = _EXPERIMENT.run_nsga


def config():
    return {
        "steps_2d": 2,
        "steps_3d": 2,
        "restarts": 1,
        "maxiter": 60,
        "population": 8,
        "fpbi_api": "model",
    }


@pytest.mark.parametrize("case", CASES[:3], ids=lambda c: c.name)
def test_analytic_hypervolume_reference_and_igd_definition(case):
    reference = case.reference_front(4097)
    metrics = quality_metrics(case, reference, reference)
    assert abs(metrics["hv_ratio"] - 1) < 1e-3
    assert metrics["igd"] < 1e-7
    if case.name == "convex_2d":
        assert case.reference_hv() == pytest.approx(1.21 - 1 / 6)
    if case.name == "concave_2d":
        assert case.reference_hv() == pytest.approx(1.21 - math.pi / 4)


def test_sphere_reference_lies_on_known_front_and_known_volume():
    case = CASES[3]
    reference = case.reference_front(100)
    assert torch.allclose(
        reference.norm(dim=-1), torch.ones(len(reference), dtype=torch.double)
    )
    assert case.reference_hv() == pytest.approx(1.331 - math.pi / 6)


def test_point_counting_distinguishes_batch_calls_and_posterior_calls():
    model = CountedModel(CASES[0])
    model.posterior(torch.rand(5, 1, 1, dtype=torch.double))
    model.forward(torch.rand(2, 1, dtype=torch.double))
    assert model.counter.posterior_calls == 1
    assert model.counter.objective_calls == 2
    assert model.counter.point_evaluations == 7


def test_common_feasibility_dominance_duplicates_and_stable_output_cap():
    case = CASES[2]
    X = torch.tensor([[-0.1], [0.2], [0.5], [0.5], [0.8], [1.1]], dtype=torch.double)
    FX, FY, stats = checked_front(case, X, case.costs(X))
    assert (
        stats["raw_rows"] == 6
        and stats["feasible_rows"] == 4
        and stats["front_rows"] == 3
    )
    assert stats["max_violation"] == pytest.approx(0.3)
    assert FX[:, 0].tolist() == [0.2, 0.5, 0.8]
    a, b = common_cap(case, FX, FY, 2)
    aa, bb = common_cap(case, FX, FY, 2)
    assert len(a) == len(b) == 2
    assert torch.equal(a, aa) and torch.equal(b, bb)


def test_budget_matching_is_generation_granular_and_honors_guard():
    assert matched_generations(101, 32, 300) == 3
    assert matched_generations(112, 32, 300) == 4
    assert matched_generations(1, 32, 300) == 1
    assert matched_generations(1_000_000, 32, 300) == 300


def test_actual_model_and_direct_fpbi_adapters_agree():
    model_result = run_fpbi(CASES[0], 11, config())
    direct_result = run_fpbi(CASES[0], 11, {**config(), "fpbi_api": "direct"})
    assert torch.allclose(model_result["Y"], direct_result["Y"], atol=1e-8)
    assert model_result["point_evaluations"] == direct_result["point_evaluations"]
    assert model_result["posterior_calls"] > 0
    assert direct_result["posterior_calls"] == 0
    assert model_result["rejected_attempts"] <= model_result["solver_attempts"]


def test_actual_upstream_nsga_uses_counted_model_and_correct_cost_sign():
    pytest.importorskip("pymoo")
    measured = run_nsga(CASES[2], 11, 2, config())
    assert 8 <= measured["point_evaluations"] <= 16
    assert measured["posterior_calls"] == 2
    assert torch.allclose(measured["Y"], CASES[2].costs(measured["X"]))
    assert CASES[2].constraints(measured["X"]).min() >= -1e-6
