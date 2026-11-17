# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Check benchmark accounting and synchronization-independent CPU behavior."""

import importlib.util
import sys
from pathlib import Path

import torch

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "notebooks_community/pareto_decision_making/experiments/benchmark_selector.py"
)
# Load the uninstalled experiment without adding the repository to sys.path.
_SPEC = importlib.util.spec_from_file_location("selector_benchmark_under_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_EXPERIMENT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _EXPERIMENT
_SPEC.loader.exec_module(_EXPERIMENT)
compare_outputs = _EXPERIMENT.compare_outputs
make_inputs = _EXPERIMENT.make_inputs
measure = _EXPERIMENT.measure


def test_identical_seed_cast_and_transfer_inputs_are_reproducible():
    Y, p, fingerprint = make_inputs((2, 4, 3), torch.double, "cpu", 17)
    YY, pp, repeated = make_inputs((2, 4, 3), torch.double, "cpu", 17)
    assert torch.equal(Y, YY) and torch.equal(p, pp) and fingerprint == repeated
    assert (p > 0).all()
    assert torch.equal(Y.float(), make_inputs((2, 4, 3), torch.float32, "cpu", 17)[0])


def test_full_callable_is_executed_for_warmups_and_every_timed_repetition():
    calls = []
    Y, p, _ = make_inputs((2, 4, 3), torch.double, "cpu", 17)

    def function(Y, preferences, *, alpha, maximize):
        calls.append((alpha, maximize))
        return (Y * preferences.unsqueeze(-2)).sum(-1).argmax(-1)

    output, result = measure(function, Y, p, "select_preferred", 2, 5)
    assert len(calls) == 7 and all(call == (0.5, True) for call in calls)
    assert len(result["seconds"]) == 5 and result["median_seconds"] > 0
    assert result["min_seconds"] <= result["median_seconds"] <= result["max_seconds"]
    assert result["cuda_peak_allocated_bytes"] == []
    assert output.shape == (2,)


def test_parity_records_discrete_mismatches_without_relaxing_them():
    before = torch.tensor([0, 1, 2])
    same = compare_outputs(before, before.clone())
    changed = compare_outputs(before, torch.tensor([0, 2, 2]))
    assert (
        same["exact_match"] and same["matched_outputs"] == same["compared_outputs"] == 3
    )
    assert not changed["exact_match"] and changed["matched_outputs"] == 2
