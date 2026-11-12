# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Collect two private, method-blind choices from a hypothetical catalog.

Contributor: @M-Colley. This is a one-person workflow pilot, not a human
benefit study, preference estimator, or Bayesian optimization acquisition.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

CATALOG = (
    {"product_id": "product_1", "cost_usd": 299, "battery_hours": 8},
    {"product_id": "product_2", "cost_usd": 499, "battery_hours": 16},
    {"product_id": "product_3", "cost_usd": 799, "battery_hours": 24},
)
QUESTION_IDS = ("initial", "repeat")
LETTERS = ("A", "B", "C")
PROMPT = (
    "Choose one hypothetical product for yourself: lower cost and longer "
    "battery life are preferable. Treat all other features as identical. "
    "You may abstain. No purchase or experiment will follow."
)


def make_plan(seed: int) -> dict[str, Any]:
    """Create a reproducible two-question catalog with different option orders."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = random.Random(seed)
    first = rng.sample(range(len(CATALOG)), len(CATALOG))
    repeat = rng.sample(range(len(CATALOG)), len(CATALOG))
    while repeat == first:
        repeat = rng.sample(range(len(CATALOG)), len(CATALOG))
    questions = []
    for question_id, order in zip(QUESTION_IDS, (first, repeat)):
        questions.append(
            {
                "question_id": question_id,
                "prompt": PROMPT,
                "options": [
                    {"label": letter, **CATALOG[index]}
                    for letter, index in zip(LETTERS, order)
                ],
            }
        )
    return {
        "schema_version": 1,
        "seed": seed,
        "questions": questions,
        "answers": [
            {"question_id": question_id, "choice": None} for question_id in QUESTION_IDS
        ],
    }


def normalize_choice(choice: Any) -> str:
    """Accept displayed letters or abstention; reject every unknown choice."""
    if not isinstance(choice, str):
        raise ValueError("choice must be A, B, C, or abstain")
    value = choice.strip().upper()
    if value == "ABSTAIN":
        return "abstain"
    if value not in LETTERS:
        raise ValueError("choice must be A, B, C, or abstain")
    return value


def evaluate_responses(payload: dict[str, Any], source: str) -> dict[str, Any]:
    """Resolve responses against the exact randomized questions they answered."""
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("a schema_version=1 response object is required")
    plan = make_plan(payload.get("seed"))
    if payload.get("questions") != plan["questions"]:
        raise ValueError("questions differ from the recorded seed/catalog")
    answers = payload.get("answers")
    if not isinstance(answers, list) or len(answers) != len(QUESTION_IDS):
        raise ValueError("exactly one answer to each question is required")
    by_id = {}
    for answer in answers:
        if not isinstance(answer, dict) or set(answer) != {"question_id", "choice"}:
            raise ValueError("each answer requires question_id and choice only")
        question_id = answer["question_id"]
        if question_id not in QUESTION_IDS or question_id in by_id:
            raise ValueError("unknown or repeated question_id")
        by_id[question_id] = normalize_choice(answer["choice"])
    resolved = []
    for question in plan["questions"]:
        choice = by_id[question["question_id"]]
        product_id = next(
            (
                option["product_id"]
                for option in question["options"]
                if option["label"] == choice
            ),
            None,
        )
        resolved.append(
            {
                "question_id": question["question_id"],
                "choice": choice,
                "product_id": product_id,
            }
        )
    assessable = all(answer["product_id"] is not None for answer in resolved)
    return {
        "schema_version": 1,
        "scope": "one-person hypothetical decision workflow pilot",
        "response_source": source,
        "respondent_identity_collected": False,
        "human_origin_independently_verified": False,
        "seed": plan["seed"],
        "questions": plan["questions"],
        "responses": resolved,
        "repeat_consistent": (
            resolved[0]["product_id"] == resolved[1]["product_id"]
            if assessable
            else None
        ),
        "limitations": (
            "An immediate repeated choice is only a workflow consistency check. "
            "It does not establish true utility, feedback-error rates, human "
            "benefit, or a Bayesian optimization acquisition result."
        ),
    }


def collect_interactively(
    plan: dict[str, Any],
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Ask a person, allowing abstention and retrying unknown choices."""
    print_fn(
        "Two hypothetical choices follow. The second repeats the same trade-off "
        "with reordered options. You may abstain from either question."
    )
    answers = []
    for question in plan["questions"]:
        print_fn(question["prompt"])
        for option in question["options"]:
            print_fn(
                f"{option['label']}: ${option['cost_usd']}; "
                f"{option['battery_hours']} hours battery"
            )
        while True:
            try:
                choice = normalize_choice(input_fn("A, B, C, or abstain: "))
            except ValueError as error:
                print_fn(str(error))
            else:
                break
        answers.append({"question_id": question["question_id"], "choice": choice})
    return {**plan, "answers": answers}


def write_private_result(result: dict[str, Any], output: Path | None) -> Path:
    """Write once to a new private file; default to a unique OS temp directory."""
    if output is None:
        directory = Path(tempfile.mkdtemp(prefix="botorch-private-feedback-"))
        output = directory / "responses.json"
    else:
        output = output.expanduser().resolve()
        if not output.parent.is_dir():
            raise ValueError("output parent must already exist")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare", action="store_true", help="Print an unanswered JSON template."
    )
    mode.add_argument(
        "--responses-json", type=Path, help="Read a completed --prepare template."
    )
    parser.add_argument(
        "--seed", type=int, help="Randomized question-order seed (not with JSON input)."
    )
    parser.add_argument(
        "--output", type=Path, help="New private result file; never overwritten."
    )
    args = parser.parse_args(argv)
    if args.responses_json is not None and args.seed is not None:
        parser.error("JSON input already records its seed; do not supply --seed")
    if args.prepare and args.output is not None:
        parser.error(
            "--prepare prints an unanswered template; --output stores answers only"
        )
    try:
        if args.responses_json is not None:
            payload = json.loads(args.responses_json.read_text(encoding="utf-8"))
            result = evaluate_responses(payload, source="json")
        else:
            plan = make_plan(
                args.seed if args.seed is not None else secrets.randbits(32)
            )
            if args.prepare:
                print(json.dumps(plan, indent=2))
                return
            result = evaluate_responses(
                collect_interactively(plan), source="interactive"
            )
        output = write_private_result(result, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Private responses saved to {output}. No answers were published.")


if __name__ == "__main__":
    main()
