# Private hypothetical-choice workflow pilot

This optional pilot collects one person's two actual responses to the same
hypothetical product trade-off. It checks that the input, randomization,
abstention, private storage, and repeat-consistency workflow functions. It does
not estimate human benefit, true utility, preference drift, or a feedback-error
rate. It does not run a BO acquisition or change an optimization trajectory.
No human responses are included in this contribution. Automated tests use
explicitly synthetic fixtures.

The three products cost $299/$499/$799 and provide 8/16/24 hours of battery
life. Assume all other features are identical. Lower cost and longer battery
life are preferable, so every product is a legitimate Pareto alternative. No
product is labeled by a recommendation method, and no method is evaluated from
the answers. A seeded uniform shuffle assigns A/B/C labels; a second shuffle,
conditioned on differing from the first, repeats the same catalog. The repeat
is disclosed to the respondent. This immediate repeat is a weak consistency
check, not an independent retest or a measure of human error.

From the repository root, run the interactive CLI:

```text
python notebooks_community/pareto_decision_making/experiments/human_feedback_pilot.py
```

Choose A, B, C, or `abstain` at each prompt. Unknown choices are rejected and
asked again. There is no fabricated default answer. The output reports whether
the same **product**, rather than the same displayed letter, was selected
twice. If either answer is an abstention, consistency is `null`.

For asynchronous input, first prepare an unanswered template:

```text
python notebooks_community/pareto_decision_making/experiments/human_feedback_pilot.py --prepare --seed 23 > private-template.json
```

Read its displayed catalog and fill only the two `answers[*].choice` fields
with A/B/C or `abstain`. Keep that completed file private. Then run:

```text
python notebooks_community/pareto_decision_making/experiments/human_feedback_pilot.py --responses-json private-template.json
```

The input records its seed and complete displayed questions; altered questions,
duplicate or unknown question IDs, missing answers, and unknown choices are
rejected. Supplying a separate seed with JSON input is also rejected. JSON
input is a response channel, not proof that a human supplied the answers.

Results default to a unique `botorch-private-feedback-*` directory in the
operating system's temporary directory, outside the checkout. The CLI prints
only the private output path, never the answers. Temporary storage may be
deleted by the operating system; move it to an appropriate private location if
retention is needed. An explicit `--output PATH` must name a new file in an
existing directory and is never overwritten. On POSIX the created file has
owner-only mode `0600`; Windows uses the current user's temporary-directory
permissions. The script neither uploads responses nor publishes them.

Do not commit completed templates, response files, or participant-level results
to this repository. No name, contact information, participant ID, or free-text
response is requested. Record a run as a one-person workflow validation only
after the actual responses have been supplied. Do not substitute the synthetic
test fixtures for a human session or infer a population-level conclusion from
a consistent pair of choices.
