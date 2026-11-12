# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Synthetic fixtures test the feedback workflow; no human data are included."""

import copy
import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "notebooks_community/pareto_decision_making/experiments/human_feedback_pilot.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "human_feedback_pilot_under_test", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
pilot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pilot)


def synthetic_responses(product_ids=("product_2", "product_2")):
    plan = pilot.make_plan(17)
    for answer, question, product_id in zip(
        plan["answers"], plan["questions"], product_ids
    ):
        answer["choice"] = (
            "abstain"
            if product_id is None
            else next(
                option["label"]
                for option in question["options"]
                if option["product_id"] == product_id
            )
        )
    return plan


def test_method_blind_catalog_is_reproducible_and_reordered():
    for seed in range(20):
        plan = pilot.make_plan(seed)
        assert plan == pilot.make_plan(seed)
        first, repeat = plan["questions"]
        assert first["options"] != repeat["options"]
        for question in plan["questions"]:
            assert {o["product_id"] for o in question["options"]} == {
                "product_1",
                "product_2",
                "product_3",
            }
            assert [o["label"] for o in question["options"]] == ["A", "B", "C"]
        assert not any(
            word in json.dumps(plan).lower() for word in ["knee", "weighted", "fpbi"]
        )


@pytest.mark.parametrize(
    "products,expected",
    [
        (("product_2", "product_2"), True),
        (("product_1", "product_3"), False),
        ((None, "product_1"), None),
        ((None, None), None),
    ],
)
def test_synthetic_fixture_consistency_uses_product_identity(products, expected):
    result = pilot.evaluate_responses(
        synthetic_responses(products), source="synthetic_fixture"
    )
    assert result["repeat_consistent"] is expected
    assert result["human_origin_independently_verified"] is False
    assert result["response_source"] == "synthetic_fixture"


@pytest.mark.parametrize("choice", ["D", "product_1", "", None, 1, True])
def test_unknown_choices_are_rejected(choice):
    plan = synthetic_responses()
    plan["answers"][0]["choice"] = choice
    with pytest.raises(ValueError, match="choice must be"):
        pilot.evaluate_responses(plan, source="synthetic_fixture")


def test_rejects_unknown_repeated_questions_and_changed_catalog():
    for mutation in ("unknown", "duplicate", "catalog"):
        plan = synthetic_responses()
        if mutation == "catalog":
            plan["questions"][0]["options"][0]["battery_hours"] = 1000
        else:
            plan["answers"][1]["question_id"] = (
                "other" if mutation == "unknown" else "initial"
            )
        with pytest.raises(ValueError):
            pilot.evaluate_responses(plan, source="synthetic_fixture")


def test_interactive_synthetic_input_retries_and_can_abstain():
    responses = iter(["unknown", "a", "abstain"])
    messages = []
    payload = pilot.collect_interactively(
        pilot.make_plan(2), input_fn=lambda _: next(responses), print_fn=messages.append
    )
    result = pilot.evaluate_responses(payload, source="synthetic_fixture")
    assert result["responses"][0]["choice"] == "A"
    assert result["responses"][1]["product_id"] is None
    assert "choice must be A, B, C, or abstain" in messages


def test_private_result_does_not_overwrite_and_has_owner_mode(tmp_path):
    result = pilot.evaluate_responses(synthetic_responses(), source="synthetic_fixture")
    output = tmp_path / "private.json"
    assert pilot.write_private_result(result, output) == output
    assert json.loads(output.read_text(encoding="utf-8")) == result
    if os.name == "posix":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        pilot.write_private_result(result, output)


def test_json_cli_outputs_only_private_path_and_preserves_template(tmp_path, capsys):
    payload = synthetic_responses()
    original = copy.deepcopy(payload)
    source = tmp_path / "answers.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "private.json"
    pilot.main(["--responses-json", str(source), "--output", str(output)])
    assert payload == original
    stdout = capsys.readouterr().out
    assert str(output) in stdout and "product_2" not in stdout
    assert json.loads(output.read_text())["repeat_consistent"] is True


def test_prepare_has_no_answers_and_default_private_directory(
    tmp_path, monkeypatch, capsys
):
    pilot.main(["--prepare", "--seed", "23"])
    template = json.loads(capsys.readouterr().out)
    assert all(a["choice"] is None for a in template["answers"])
    monkeypatch.setattr(pilot.tempfile, "mkdtemp", lambda **_: str(tmp_path))
    result = pilot.evaluate_responses(synthetic_responses(), source="synthetic_fixture")
    output = pilot.write_private_result(result, None)
    assert output == tmp_path / "responses.json"


def test_bad_json_cli_cannot_write_results(tmp_path):
    payload = synthetic_responses()
    payload["answers"][0]["choice"] = "not-a-choice"
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "private.json"
    with pytest.raises(SystemExit):
        pilot.main(["--responses-json", str(source), "--output", str(output)])
    assert not output.exists()
