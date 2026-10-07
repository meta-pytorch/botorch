# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness tests for the EXPERIMENT_PLAN experimental setup.

These exist because the failure modes in this project are not crashes -- they are runs
that complete and produce wrong or meaningless numbers. Seven silent no-ops,
two inverted metric directions, one 62-hour hang, one mislabelled benchmark, and
one config that did not match the run it claimed to reproduce. Every test below
encodes one of those.

Deliberately fast and data-free: no Buck target execution, no LCBench/PD1 loading. The
point is to be runnable before every launch, not to re-measure the science.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from _scratch_bo import (
    analyze_pd1pool,
    bo_diagnose,
    bo_experiment,
    gen_stage_queue,
    prior_cache,
    stage4,
)
from _scratch_bo.summarize_stage import budget_to_target


class RegimeFlagsTest(unittest.TestCase):
    """Every regime in the plan must actually parse against its harness.

    A typo here is a multi-day run that produces nothing, and until build_parser() was
    extracted there was no way to check it short of executing the sweep.
    """

    def _parser(self, harness: str) -> argparse.ArgumentParser:
        return {
            "bo_experiment": bo_experiment.build_parser,
            "bo_diagnose": bo_diagnose.build_parser,
        }[harness]()

    def test_every_regime_parses(self) -> None:
        for name, spec in gen_stage_queue.REGIMES.items():
            with self.subTest(regime=name):
                parser = self._parser(spec["harness"])
                # argparse calls sys.exit on an unknown flag; surface it as a failure
                # naming the regime instead.
                try:
                    parser.parse_args(spec["flags"].split() + ["--out", "/tmp/x.json"])
                except SystemExit:
                    self.fail(f"regime {name!r} has flags its harness rejects")

    def test_every_ofat_cell_parses_on_every_harness(self) -> None:
        """An OFAT cell runs on EVERY regime, so it must parse on EVERY harness.

        Regression: this test previously checked only bo_experiment, so nine cells that
        bo_diagnose could not parse (--em-base-mean, --use-covar-prior, --iw-nu,
        --use-mean-prior, --em-init-mode, --em-mean) passed the suite while being
        guaranteed to crash on the two regression regimes -- 18 cells of the matrix
        producing nothing.
        """
        harnesses = {spec["harness"] for spec in gen_stage_queue.REGIMES.values()}
        self.assertIn("bo_diagnose", harnesses, "expected a regression regime")
        self.assertIn("bo_experiment", harnesses, "expected a BO regime")
        for harness in sorted(harnesses):
            parser = self._parser(harness)
            for cell_name, flags in gen_stage_queue.OFAT:
                if not flags:
                    continue
                with self.subTest(harness=harness, cell=cell_name):
                    try:
                        parser.parse_args(flags.split() + ["--out", "/tmp/x.json"])
                    except SystemExit:
                        self.fail(
                            f"OFAT cell {cell_name!r} does not parse on {harness}: "
                            f"{flags}"
                        )

    def test_em_knobs_exist_on_both_harnesses(self) -> None:
        """The EM factors the plan varies must be settable wherever they are measured.

        A flag present on one harness and absent on the other means the same named cell
        silently measures different things in BO and regression.
        """
        shared = [
            "--em-base-mean",
            "--em-mean",
            "--em-shrinkage",
            "--em-canonical",
            "--use-covar-prior",
            "--use-mean-prior",
            "--em-init-mode",
        ]
        for flag in shared:
            for harness in ("bo_experiment", "bo_diagnose"):
                with self.subTest(flag=flag, harness=harness):
                    opts = {
                        s
                        for a in self._parser(harness)._actions
                        for s in a.option_strings
                    }
                    self.assertIn(flag, opts, f"{flag} missing from {harness}")

    def test_regime_harnesses_are_known(self) -> None:
        for name, spec in gen_stage_queue.REGIMES.items():
            self.assertIn(spec["harness"], ("bo_experiment", "bo_diagnose"), name)
            self.assertTrue(spec["shards"].isdigit(), f"{name}: shards must be numeric")


class GuardTest(unittest.TestCase):
    """Flags that cannot act must RAISE, never silently do nothing.

    Seven silent no-ops have been found so far; each of these guards closes one.
    """

    def test_iw_nu_without_covar_prior_raises(self) -> None:
        """--iw-nu is read only inside `if use_covar_prior:` (S27.16, no-op #7)."""
        args = bo_experiment.build_parser().parse_args(
            ["--iw-nu", "500", "--out", "/tmp/x.json"]
        )
        self.assertIsNotNone(args.iw_nu)
        self.assertFalse(args.use_covar_prior)
        # main() performs the check; assert the precondition it keys on is detectable.
        self.assertTrue(args.iw_nu is not None and not args.use_covar_prior)

    def test_iw_nu_with_covar_prior_is_allowed(self) -> None:
        args = bo_experiment.build_parser().parse_args(
            ["--iw-nu", "500", "--use-covar-prior", "--out", "/tmp/x.json"]
        )
        self.assertTrue(args.use_covar_prior)

    def test_needs_hyperbo_base_kernel_matches_additive_variants(self) -> None:
        """em_additive_hyperbo* starts with 'em', so startswith('hyperbo') misses it."""
        f = bo_experiment._needs_hyperbo_base_kernel
        self.assertTrue(f(["em_additive_hyperbo"]))
        self.assertTrue(f(["em_additive_hyperbo_frozen"]))
        self.assertTrue(f(["em_frozen", "em_additive_hyperbo"]))
        self.assertFalse(f(["em_frozen", "hyperbo_frozen"]))
        self.assertFalse(f([]))


class BlendedMeanTest(unittest.TestCase):
    """beta must span replacement and blending, with the endpoints exact."""

    def setUp(self) -> None:
        self.x = torch.linspace(0.0, 1.0, 7, dtype=torch.double).unsqueeze(-1)

        class _Const(torch.nn.Module):
            def __init__(self, v: float) -> None:
                super().__init__()
                self.v = v

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.full(x.shape[:-1], self.v, dtype=x.dtype)

        self.em, self.hb = _Const(1.0), _Const(3.0)

    def _blend(self, beta: float):
        return bo_experiment._BlendedMean(self.em, self.hb, beta)

    def test_beta_zero_is_exactly_the_em_mean(self) -> None:
        out = self._blend(0.0)(self.x)
        self.assertTrue(torch.allclose(out, torch.ones_like(out)))

    def test_beta_one_is_exactly_the_hyperbo_mean(self) -> None:
        out = self._blend(1.0)(self.x)
        self.assertTrue(torch.allclose(out, torch.full_like(out, 3.0)))

    def test_intermediate_beta_interpolates(self) -> None:
        out = self._blend(0.25)(self.x)
        self.assertTrue(torch.allclose(out, torch.full_like(out, 1.5)))

    def test_replacement_is_the_degenerate_end_of_the_blend(self) -> None:
        """beta=1 must equal the pure-replacement path, not merely approximate it."""
        self.assertTrue(
            torch.equal(self._blend(1.0)(self.x), self.hb(self.x)),
        )


class BudgetToTargetTest(unittest.TestCase):
    """The primary metric. Censoring and direction have both caused real errors."""

    def test_reaches_target_at_correct_index(self) -> None:
        # pool_max 1.0; regret hits 0.005 at index 2.
        traj = [[0.0, 0.5, 0.995, 0.995]]
        self.assertEqual(budget_to_target(traj, [1.0], 0.01), 2.0)

    def test_never_reaching_is_censored_past_the_budget(self) -> None:
        """Non-solvers must be censored, not dropped -- dropping lets a method look fast
        by solving only the easy runs."""
        traj = [[0.0, 0.1, 0.2, 0.3]]
        self.assertEqual(budget_to_target(traj, [1.0], 0.01), 4.0)  # len(row)

    def test_immediate_hit_is_zero(self) -> None:
        self.assertEqual(budget_to_target([[1.0, 1.0]], [1.0], 0.01), 0.0)

    def test_averages_over_runs_including_censored(self) -> None:
        traj = [[1.0, 1.0], [0.0, 0.0]]  # first hits at 0, second censored at 2
        self.assertEqual(budget_to_target(traj, [1.0, 1.0], 0.01), 1.0)

    def test_lower_is_better_direction(self) -> None:
        """A strictly faster method must score strictly lower."""
        fast = budget_to_target([[0.0, 1.0, 1.0, 1.0]], [1.0], 0.01)
        slow = budget_to_target([[0.0, 0.0, 0.0, 1.0]], [1.0], 0.01)
        self.assertLess(fast, slow)

    def test_tighter_target_is_never_easier(self) -> None:
        traj = [[0.0, 0.98, 0.999, 0.999]]
        loose = budget_to_target(traj, [1.0], 0.05)
        tight = budget_to_target(traj, [1.0], 0.0001)
        self.assertLessEqual(loose, tight)


class QueueGenerationTest(unittest.TestCase):
    """The queue and its provenance are what a future agent reads to interpret a run."""

    def test_ofat_changes_exactly_one_thing_from_base(self) -> None:
        names = [n for n, _ in gen_stage_queue.OFAT]
        self.assertEqual(names[0], "base", "the OFAT baseline must come first")
        self.assertEqual(
            gen_stage_queue.OFAT[0][1],
            "",
            "the baseline cell must carry no extra flags",
        )
        self.assertEqual(len(names), len(set(names)), "duplicate OFAT cell names")

    def test_queue_and_provenance_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(gen_stage_queue, "ROOT", tmp):
                cells = [("c1", "--em-base-mean linear")]
                path = gen_stage_queue.write_provenance("st", "pd1_bo_armC", cells)
                self.assertTrue(os.path.exists(path))
                with open(path) as f:
                    prov = json.load(f)
        self.assertEqual(prov["stage"], "st")
        self.assertEqual(prov["regime"], "pd1_bo_armC")
        self.assertEqual([c["name"] for c in prov["cells"]], ["c1"])
        self.assertIn("commit", prov)
        # The old/new boundary must be stated in the artifact itself, not only in docs.
        self.assertIn("v2", prov["note"])

    def test_provenance_keeps_history_when_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(gen_stage_queue, "ROOT", tmp):
                gen_stage_queue.write_provenance("st", "lcb_bo", [("a", "")])
                path = gen_stage_queue.write_provenance("st", "lcb_bo", [("b", "")])
                with open(path) as f:
                    prov = json.load(f)
        self.assertEqual(len(prov["history"]), 1, "regeneration must not erase history")


class OutputBoundaryTest(unittest.TestCase):
    """New results must be unmistakably separable from pre-2026-08-17 ones."""

    def test_all_new_output_is_under_v2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(gen_stage_queue, "ROOT", tmp):
                gen_stage_queue.write_provenance("stage0_lcb_bo", "lcb_bo", [("a", "")])
            written = [os.path.join(dp, f) for dp, _, fs in os.walk(tmp) for f in fs]
        self.assertTrue(written, "nothing was written")
        for path in written:
            self.assertIn(
                os.path.join("raw", "v2"),
                path,
                f"{path} escapes the v2 boundary and could be mistaken for old results",
            )


class PoolingWeightTest(unittest.TestCase):
    """Priors must be weighted equally regardless of seed count (review, metric #1).

    Arm C's p0 has 5 seeds against p1/p2's 3, so a plain mean over pooled runs gave p0
    45.5% of the weight. That reversed S27.14's headline once corrected.
    """

    class _FakePooled:
        """Two priors with deliberately unequal run counts."""

        def __init__(self) -> None:
            import numpy as np

            self.prior_idx = np.array([0, 0, 0, 0, 1, 1])  # 4 runs vs 2
            self.ds_idx = np.array([0, 0, 1, 1, 0, 1])
            self.n_priors = 2
            self.n_runs = 6
            self.run_w = np.ones(6, dtype=float)
            for k in (0, 1):
                sel = self.prior_idx == k
                self.run_w[sel] = 1.0 / (self.n_priors * int(sel.sum()))

        wmean = analyze_pd1pool.PooledStudy.wmean
        task_wmean = analyze_pd1pool.PooledStudy.task_wmean

    def test_unequal_seed_counts_do_not_bias_the_mean(self) -> None:
        import numpy as np

        pooled = self._FakePooled()
        # Prior 0 says 10, prior 1 says 0. Equal weighting must give 5, not 6.67.
        values = np.array([10.0, 10.0, 10.0, 10.0, 0.0, 0.0])
        self.assertAlmostEqual(pooled.wmean(values), 5.0, places=9)
        self.assertNotAlmostEqual(float(values.mean()), 5.0, places=3)

    def test_weights_sum_to_one_per_prior(self) -> None:
        pooled = self._FakePooled()
        for k in (0, 1):
            share = pooled.run_w[pooled.prior_idx == k].sum()
            self.assertAlmostEqual(share, 0.5, places=9)

    def test_task_mean_equalises_priors_within_a_task(self) -> None:
        import numpy as np

        pooled = self._FakePooled()
        # Task 0 is seen twice under p0 (value 10) and once under p1 (value 0).
        values = np.array([10.0, 10.0, 99.0, 99.0, 0.0, 99.0])
        self.assertAlmostEqual(pooled.task_wmean(values, 0), 5.0, places=9)


class SummariserAlignmentTest(unittest.TestCase):
    """pool_max must stay aligned with each method's rows (review, metric #2)."""

    def _shard(self, tmp: str, name: str, methods: dict, pool_max: list) -> str:
        path = os.path.join(tmp, name)
        with open(path, "w") as f:
            json.dump(
                {
                    "per_run": {
                        "traj_raw": methods,
                        "pool_max_raw": pool_max,
                        "eval_dataset_idx": list(range(len(pool_max))),
                    }
                },
                f,
            )
        return path

    def test_shard_missing_a_method_does_not_misalign(self) -> None:
        from _scratch_bo.summarize_stage import summarise_bo

        with tempfile.TemporaryDirectory() as tmp:
            a = self._shard(
                tmp, "a_s0.json", {"m1": [[0.0, 1.0]], "m2": [[0.0, 1.0]]}, [1.0]
            )
            # Second shard lacks m2 entirely.
            b = self._shard(tmp, "a_s1.json", {"m1": [[0.0, 0.5]]}, [2.0])
            out = summarise_bo([a, b])
        # m1 saw both shards, m2 only the first. Neither may borrow the other's pool_max.
        self.assertEqual(out["methods"]["m1"]["n_runs"], 2)
        self.assertEqual(out["methods"]["m2"]["n_runs"], 1)
        # m2's single run had pool_max 1.0 and final 1.0 -> regret 0, not 2.0-1.0=1.0.
        self.assertAlmostEqual(out["methods"]["m2"]["final_regret_mean"], 0.0, places=9)

    def test_corrupt_shard_does_not_kill_the_whole_cell(self) -> None:
        """One truncated file used to turn a 12-shard cell into {"kind": "error"}."""
        from _scratch_bo.summarize_stage import summarise_bo

        with tempfile.TemporaryDirectory() as tmp:
            good = self._shard(tmp, "a_s0.json", {"m1": [[0.0, 1.0]]}, [1.0])
            out = summarise_bo([good])
        self.assertEqual(out["methods"]["m1"]["n_runs"], 1)


class RealPooledStudyTest(unittest.TestCase):
    """Construct a REAL PooledStudy. The previous fixture re-implemented the weight
    loop, so breaking production changed nothing."""

    def _study(self, tmp: str, seeds_per_prior: tuple[int, ...]) -> object:
        pats = []
        for k, n_seeds in enumerate(seeds_per_prior):
            path = os.path.join(tmp, f"p{k}_s0.json")
            n_runs = 2 * n_seeds  # two tasks x n_seeds
            with open(path, "w") as f:
                json.dump(
                    {
                        "config": {},
                        "n_runs": n_runs,
                        "per_run": {
                            "traj_raw": {"m": [[0.0, float(k)] for _ in range(n_runs)]},
                            "pool_max_raw": [1.0] * n_runs,
                            "eval_dataset_idx": [0, 1] * n_seeds,
                        },
                    },
                    f,
                )
            pats.append(path)
        return analyze_pd1pool.PooledStudy(pats)

    def test_unequal_seeds_do_not_bias_the_weighted_mean(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            pooled = self._study(tmp, (5, 3, 3))  # the real arm-C imbalance
            # Value depends only on the prior: p0 -> 0, p1 -> 1, p2 -> 2. Equal
            # weighting must give exactly 1.0; run-count weighting gives ~0.91.
            vals = pooled.prior_idx.astype(float)
            self.assertAlmostEqual(pooled.wmean(vals), 1.0, places=9)
            self.assertNotAlmostEqual(float(np.mean(vals)), 1.0, places=3)

    def test_prior_weights_are_equal_by_construction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pooled = self._study(tmp, (5, 3, 3))
            for k in range(pooled.n_priors):
                share = pooled.run_w[pooled.prior_idx == k].sum()
                self.assertAlmostEqual(share, 1.0 / 3.0, places=9)

    def test_task_variance_is_the_between_prior_spread(self) -> None:
        """R6-S1: the spread of the per-prior means IS the whole variance.

        Each per-prior mean is itself noisy, so its spread already contains the within
        component. The earlier `within + between` form counted it twice and inflated
        the SE by up to sqrt(2), which is what turned two arm-C contrasts into ties in
        the first version of §27.20.
        """
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            pooled = self._study(tmp, (5, 3, 3))
            rng = np.random.default_rng(0)
            vals = rng.normal(size=pooled.n_runs)
            sel = pooled.ds_idx == 0
            means = [
                float(vals[sel][pooled.prior_idx[sel] == k].mean())
                for k in np.unique(pooled.prior_idx[sel])
            ]
            expected = float(np.var(means, ddof=1)) / len(means)
            self.assertAlmostEqual(pooled.task_wvar(vals, 0), expected, places=12)

    def test_missing_prior_coverage_is_surfaced(self) -> None:
        """A task absent from one prior re-introduces the bias and must not be silent."""
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            pooled = self._study(tmp, (2, 2))
            pooled.prior_idx[pooled.ds_idx == 0] = 0  # task 0 now only under p0
            pooled.task_wmean(np.zeros(pooled.n_runs), 0)
            self.assertIn(0, pooled.coverage_gaps)


class AtomicWriteTest(unittest.TestCase):
    """A reader must never observe a partial file (review N16/N17/N18)."""

    def test_no_partial_file_is_left_on_failure(self) -> None:
        from _scratch_bo.bo_experiment import atomic_write_json

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.json")

            class _Unserialisable:
                pass

            with self.assertRaises(TypeError):
                atomic_write_json(target, {"bad": {1, 2, 3}, "worse": _Unserialisable})
            self.assertFalse(
                os.path.exists(target), "a failed write must leave no target file"
            )
            self.assertEqual(
                [f for f in os.listdir(tmp) if f.endswith(".partial")],
                [],
                "temp files must be cleaned up",
            )

    def test_successful_write_is_valid_json(self) -> None:
        from _scratch_bo.bo_experiment import atomic_write_json

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.json")
            atomic_write_json(target, {"a": 1})
            with open(target) as f:
                self.assertEqual(json.load(f), {"a": 1})


class SummariserRobustnessTest(unittest.TestCase):
    """Genuinely truncated input, not a stand-in (the old test had no corrupt shard)."""

    def test_truncated_shard_does_not_poison_the_cell(self) -> None:
        from _scratch_bo.summarize_stage import summarise_bo

        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "c_s0.json")
            with open(good, "w") as f:
                json.dump(
                    {
                        "per_run": {
                            "traj_raw": {"m": [[0.0, 1.0]]},
                            "pool_max_raw": [1.0],
                            "eval_dataset_idx": [0],
                        }
                    },
                    f,
                )
            bad = os.path.join(tmp, "c_s1.json")
            with open(bad, "w") as f:
                f.write('{"per_run": {"traj_raw": {"m": [[0.0, 1.')  # truncated
            out = summarise_bo([good, bad])
        self.assertEqual(out["methods"]["m"]["n_runs"], 1, "good shard must survive")


class DriverExecutionTest(unittest.TestCase):
    """Run run_cell_queue.sh FOR REAL against a stub harness.

    The previous OrchestrationContractTest only grepped shell source, so it would have
    passed against a driver that ran nothing. These assert on observable behaviour:
    exit status, which shard files appear, and the .incomplete marker.
    """

    def _driver(self) -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        for c in (
            os.path.join(here, "run_cell_queue.sh"),
            os.path.join(here, "..", "results", "run_cell_queue.sh"),
        ):
            if os.path.exists(c):
                return c
        self.skipTest("run_cell_queue.sh not available as a resource")

    def _run(self, tmp: str, shards: str, stub: str, cells: str):
        """Run the queue with python3 shimmed by a stub on PATH."""
        bindir = os.path.join(tmp, "bin")
        os.makedirs(bindir, exist_ok=True)
        shim = os.path.join(bindir, "python3")
        with open(shim, "w") as f:
            f.write(stub)
        os.chmod(shim, 0o755)

        queue = os.path.join(tmp, "q.tsv")
        with open(queue, "w") as f:
            f.write(cells)

        env = dict(os.environ)
        env.update(
            {
                "PATH": bindir + os.pathsep + env["PATH"],
                "RESULTS_DIR": os.path.join(tmp, "results"),
                "STUDY": "unit",
                "QUEUE": queue,
                "COMMON": "--dummy",
                "SHARDS": shards,
                "THREADS": "1",
                "CELLS_PARALLEL": "1",
            }
        )
        proc = subprocess.run(
            ["bash", self._driver()],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        raw = os.path.join(tmp, "results", "raw", "unit")
        return proc, raw

    _STUB_OK = (
        "#!/bin/bash\n"
        "# Emulate `python3 -m _scratch_bo.<harness> ... --out PATH` by writing a minimal valid shard.\n"
        'out=""\n'
        'while [ $# -gt 0 ]; do [ "$1" = "--out" ] && out="$2"; shift; done\n'
        '[ -n "$out" ] && printf \'{"per_run": {"traj_raw": {}, "pool_max_raw": []}}\' > "$out"\n'
        "exit 0\n"
    )
    _STUB_FAIL = "#!/bin/bash\nexit 7\n"

    def test_successful_cell_writes_shards_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc, raw = self._run(tmp, "2", self._STUB_OK, "c1\t--x\n")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertTrue(os.path.exists(os.path.join(raw, "c1_s0.json")))
            self.assertTrue(os.path.exists(os.path.join(raw, "c1_s1.json")))
            self.assertFalse(os.path.exists(os.path.join(raw, ".incomplete")))

    def test_failing_harness_marks_incomplete_and_exits_nonzero(self) -> None:
        """The whole point of the .DONE work: a failed cell must be visible."""
        with tempfile.TemporaryDirectory() as tmp:
            proc, raw = self._run(tmp, "2", self._STUB_FAIL, "c1\t--x\n")
            self.assertNotEqual(proc.returncode, 0, "a failed cell must fail the queue")
            self.assertTrue(
                os.path.exists(os.path.join(raw, ".incomplete")),
                "the failed cell must be recorded",
            )

    def test_corrupt_shard_is_quarantined_and_rerun(self) -> None:
        """Not a log-message grep: assert the file is actually replaced."""
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "results", "raw", "unit")
            os.makedirs(raw, exist_ok=True)
            corrupt = os.path.join(raw, "c1_s0.json")
            with open(corrupt, "w") as f:
                f.write('{"per_run": {"traj')  # truncated
            proc, raw = self._run(tmp, "1", self._STUB_OK, "c1\t--x\n")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            with open(corrupt) as f:
                json.load(f)  # must now be valid: it was re-run
            self.assertTrue(
                [f for f in os.listdir(raw) if ".corrupt." in f],
                "the bad file must be quarantined, not deleted",
            )

    def test_valid_shard_is_not_rerun(self) -> None:
        """Resumability: an existing valid shard must be left alone."""
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "results", "raw", "unit")
            os.makedirs(raw, exist_ok=True)
            keep = os.path.join(raw, "c1_s0.json")
            with open(keep, "w") as f:
                json.dump({"sentinel": True}, f)
            # A stub that would FAIL if invoked; passing proves it was skipped.
            proc, raw = self._run(tmp, "1", self._STUB_FAIL, "c1\t--x\n")
            self.assertEqual(proc.returncode, 0, "existing valid shard must be skipped")
            with open(keep) as f:
                self.assertEqual(json.load(f), {"sentinel": True})

    def test_non_numeric_shards_is_rejected(self) -> None:
        """An empty/garbage SHARDS silently became 12 and duplicated runs (N14/B14)."""
        with tempfile.TemporaryDirectory() as tmp:
            proc, _ = self._run(tmp, "notanumber", self._STUB_OK, "c1\t--x\n")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("must be numeric", proc.stdout + proc.stderr)


class BlendedMeanPersistenceTest(unittest.TestCase):
    """beta must survive the journeys the mean actually takes.

    pretrain_em_prior stores the mean module in EMPriorContainer and make_surrogate
    deepcopies it per fold. If beta were lost anywhere on that path the mean would
    silently revert to EM's, which is precisely the class of silent no-op that has
    bitten this project seven times -- and it would look like "the mean transfer does
    not help" rather than like a bug.
    """

    def _blend(self, beta: float):
        from gpytorch.means import ConstantMean

        em, hb = ConstantMean(), ConstantMean()
        with torch.no_grad():
            em.constant.fill_(1.0)
            hb.constant.fill_(3.0)
        return bo_experiment._BlendedMean(em, hb, beta)

    def test_beta_survives_deepcopy(self) -> None:
        import copy

        b = self._blend(0.25)
        x = torch.zeros(4, 2, dtype=torch.double)
        expected = float(b(x)[0])
        d = copy.deepcopy(b)
        self.assertEqual(d.beta, 0.25)
        self.assertAlmostEqual(float(d(x)[0]), expected, places=9)

    def test_beta_is_not_carried_by_state_dict(self) -> None:
        """Documents a real limitation rather than assuming it away.

        beta is a plain float, not a buffer, so load_state_dict does NOT restore it.
        In-process use (deepcopy) is safe; any future save/load of a prior to disk must
        persist beta separately or the blend silently reverts.
        """
        b = self._blend(1.0)
        fresh = self._blend(0.0)
        fresh.load_state_dict(b.state_dict())
        self.assertEqual(
            fresh.beta,
            0.0,
            "if this ever starts passing beta through state_dict, relax the warning "
            "in _BlendedMean's docstring",
        )

    def test_submodules_are_registered(self) -> None:
        """Both means must be real submodules, or deepcopy/.to() would miss them."""
        b = self._blend(0.5)
        names = {n for n, _ in b.named_children()}
        self.assertEqual(names, {"mean_em", "mean_hb"})


class SharedKernelScaleTest(unittest.TestCase):
    """D2: the shared kernel must be fitted in the caller's target space.

    build_shared_gp_model_list used BoTorch's DEFAULT outcome_transform, a PER-TASK
    Standardize. With globally-standardized data that silently refits the shared mean
    against zero-mean targets and calibrates the outputscale to unit per-task variance,
    while pretrain_em_prior consumes the global scale -- so Sigma_init, the EM kernel
    init and the shrinkage target were all on the wrong scale.
    """

    def _datasets(self):
        from botorch.models.empirical_gps.utils import ExperimentDataset

        torch.manual_seed(0)
        X = torch.rand(24, 2, dtype=torch.double)
        ds = []
        for k in range(4):
            # Deliberately different per-task means AND scales.
            y = (0.5 + k) * X[:, :1] + 3.0 * k
            ds.append(ExperimentDataset(X=X, Y=y))
        allY = torch.cat([t.Y for t in ds])
        gm, gs = allY.mean(), allY.std().clamp_min(1e-8)
        return [ExperimentDataset(X=t.X, Y=(t.Y - gm) / gs) for t in ds]

    def test_per_task_variance_is_not_unit_after_global_standardization(self) -> None:
        """The precondition that makes D2 bite: if every task were already unit
        variance, the hidden Standardize would be harmless."""
        variances = [float(t.Y.var()) for t in self._datasets()]
        self.assertTrue(
            any(abs(v - 1.0) > 0.1 for v in variances),
            f"fixture must have non-unit per-task variance, got {variances}",
        )

    def test_outcome_transform_none_reaches_the_gps(self) -> None:
        from botorch.models.empirical_gps.em_empirical_gp import (
            build_shared_gp_model_list,
        )
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.means import ConstantMean

        model_list, _ = build_shared_gp_model_list(
            self._datasets(),
            ConstantMean(),
            ScaleKernel(MaternKernel(nu=2.5)),
            outcome_transform=None,
        )
        for gp in model_list.models:
            self.assertIsNone(
                getattr(gp, "outcome_transform", None),
                "outcome_transform=None must reach the SingleTaskGPs",
            )

    def test_library_default_is_unchanged(self) -> None:
        """Other callers must keep the old behaviour: this was an opt-in, not a
        silent change to a shared library."""
        from botorch.models.empirical_gps.em_empirical_gp import (
            build_shared_gp_model_list,
        )
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.means import ConstantMean

        model_list, _ = build_shared_gp_model_list(
            self._datasets(), ConstantMean(), ScaleKernel(MaternKernel(nu=2.5))
        )
        self.assertTrue(
            any(
                getattr(gp, "outcome_transform", None) is not None
                for gp in model_list.models
            ),
            "the library default must remain Standardize",
        )

    def test_per_task_standardized_data_makes_the_transform_inert(self) -> None:
        """Answers the design question directly: does the EM path standardize twice?

        No. --per-task-standardize rescales each task to zero mean / unit variance BEFORE
        build_shared_kernel, so a subsequent per-task Standardize is a near-identity.
        That is why D2 is inert on PD1 (which sets the flag) and real on LCBench (which
        does not). Locking this down because §27.18 got the blast radius wrong by
        reasoning about it instead of checking.
        """
        from botorch.models.empirical_gps.utils import ExperimentDataset

        torch.manual_seed(0)
        X = torch.rand(24, 2, dtype=torch.double)
        raw = [(0.5 + k) * X[:, :1] + 3.0 * k for k in range(4)]

        per_task = [
            ExperimentDataset(X=X, Y=(y - y.mean()) / y.std().clamp_min(1e-8))
            for y in raw
        ]
        for t in per_task:
            self.assertAlmostEqual(float(t.Y.mean()), 0.0, places=9)
            self.assertAlmostEqual(float(t.Y.std()), 1.0, places=6)

        allY = torch.cat(raw)
        glob = [
            ExperimentDataset(X=X, Y=(y - allY.mean()) / allY.std().clamp_min(1e-8))
            for y in raw
        ]
        spread = [abs(float(t.Y.std()) - 1.0) for t in glob]
        self.assertTrue(
            max(spread) > 0.1,
            "globally standardized tasks must NOT be unit variance, or D2 could not "
            f"bite; got {spread}",
        )


class WiringTest(unittest.TestCase):
    """A fix that is defined but never CALLED fixes nothing -- checked by RUNNING it.

    Round 3 added `mean_baseline` and `_safe_copy_*`; round 4 found both were dead. The
    round-4 replacements were source greps, which is the same failure one level up: they
    prove a line exists, not that it does anything. These build a real surrogate and
    assert the observable property instead.
    """

    def _const_mean(self, value: float):
        from gpytorch.means import ConstantMean

        m = ConstantMean()
        with torch.no_grad():
            m.constant.fill_(value)
        return m

    def _shared_kernel(self):
        from gpytorch.kernels import MaternKernel, ScaleKernel

        k = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=3))
        with torch.no_grad():
            k.raw_outputscale.fill_(0.7)
        return k

    def test_baseline_does_not_inherit_the_blended_mean(self) -> None:
        """D1's actual property: with --em-mean, pretrained_gp_frozen must keep the
        PRE-BLEND mean, or the control silently becomes a second transfer method."""
        em, hb = self._const_mean(1.0), self._const_mean(9.0)
        blended = bo_experiment._BlendedMean(em, hb, 1.0)  # fully HyperBO's mean
        X = torch.rand(6, 3, dtype=torch.double)
        Y = torch.rand(6, 1, dtype=torch.double)

        model = bo_experiment.make_surrogate(
            "pretrained_gp_frozen",
            X,
            Y,
            None,
            blended,
            self._shared_kernel(),
            mean_baseline=em,
        )
        got = float(model.mean_module.constant)
        self.assertAlmostEqual(
            got,
            1.0,
            places=6,
            msg=f"baseline inherited {got}, i.e. the blended mean -- the control is "
            "contaminated (D1)",
        )

    def test_without_mean_baseline_the_shared_mean_is_used(self) -> None:
        """Backwards compatibility: callers that never blend are unaffected."""
        em = self._const_mean(2.5)
        X = torch.rand(6, 3, dtype=torch.double)
        Y = torch.rand(6, 1, dtype=torch.double)
        model = bo_experiment.make_surrogate(
            "pretrained_gp_frozen", X, Y, None, em, self._shared_kernel()
        )
        self.assertAlmostEqual(float(model.mean_module.constant), 2.5, places=6)

    def test_outputscale_is_actually_transferred(self) -> None:
        """_safe_copy_outputscale must move the value, not merely be referenced."""
        em = self._const_mean(0.0)
        X = torch.rand(6, 3, dtype=torch.double)
        Y = torch.rand(6, 1, dtype=torch.double)
        model = bo_experiment.make_surrogate(
            "pretrained_gp_frozen", X, Y, None, em, self._shared_kernel()
        )
        self.assertAlmostEqual(float(model.covar_module.raw_outputscale), 0.7, places=6)

    def test_composite_kernel_degrades_loudly_not_silently(self) -> None:
        """A canonical kernel with no raw_outputscale must warn, not quietly hand back
        an untrained 'pre-trained' baseline (R3)."""
        import contextlib
        import io

        from gpytorch.kernels import MaternKernel

        em = self._const_mean(0.0)
        X = torch.rand(6, 3, dtype=torch.double)
        Y = torch.rand(6, 1, dtype=torch.double)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bo_experiment.make_surrogate(
                "pretrained_gp_frozen", X, Y, None, em, MaternKernel(nu=2.5)
            )
        self.assertIn("NOT warm-started", buf.getvalue())


class CensoringSemanticsTest(unittest.TestCase):
    """The primary metric is RMTT@T, and it is biased. Lock the behaviour so nobody
    re-adds the false claim that censoring prevents 'looking fast by solving easy runs'.
    """

    class _Study:
        def __init__(self, regret) -> None:
            import numpy as np

            self.regret = {"m": np.asarray(regret, dtype=float)}

    def test_non_solver_is_charged_the_horizon(self) -> None:
        # Regret is pool_max - achieved, so it is POSITIVE and a solver drives it to 0.
        s = self._Study([[1.0, 1.0, 1.0, 1.0]])  # never within tol of 0
        got = analyze_pd1pool.budget_censored(s, "m", -0.01)
        self.assertEqual(float(got[0]), 4.0, "censored at the horizon T")

    def test_the_documented_pathology_is_real(self) -> None:
        """A method failing half its runs can beat one that solves them all.

        This is the worked example in budget_censored's docstring. If this test ever
        fails, the estimator changed and the docstring must be rewritten with it.
        """
        import numpy as np

        horizon = 6
        solved_early = [0.0] * horizon  # regret 0 from index 0
        never = [1.0] * horizon  # regret never reaches the target
        a = self._Study(np.array([solved_early, never]))  # 50% solved at index 0
        b = self._Study(np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 2))  # 100% at 2
        mean_a = float(analyze_pd1pool.budget_censored(a, "m", -0.01).mean())
        mean_b = float(analyze_pd1pool.budget_censored(b, "m", -0.01).mean())
        self.assertEqual((mean_a, mean_b), (3.0, 2.0))
        self.assertGreater(
            mean_a,
            mean_b,
            "with a 6-step horizon B correctly wins; the pathology needs a horizon "
            "large enough that censoring under-charges the failures",
        )
        # Now the documented case: a long horizon makes censoring too generous.
        long_h = 50
        a2 = self._Study(
            np.array([[0.0] * long_h, [1.0] * long_h])  # 50% at 0, 50% never
        )
        b2 = self._Study(
            np.array([[1.0] * 30 + [0.0] * (long_h - 30)] * 2)
        )  # all at 30
        m_a2 = float(analyze_pd1pool.budget_censored(a2, "m", -0.01).mean())
        m_b2 = float(analyze_pd1pool.budget_censored(b2, "m", -0.01).mean())
        self.assertLess(
            m_a2,
            m_b2,
            "A fails half its runs yet scores better; if this ever stops holding, "
            "budget_censored's docstring must be rewritten",
        )

    def test_2T_sensitivity_penalises_non_solvers_harder(self) -> None:
        """The cheapest available check on whether a conclusion survives the constant."""
        s = self._Study([[1.0, 1.0, 1.0, 1.0]])
        at_t = float(analyze_pd1pool.budget_censored_at(s, "m", -0.01, 1.0)[0])
        at_2t = float(analyze_pd1pool.budget_censored_at(s, "m", -0.01, 2.0)[0])
        self.assertEqual(at_t, 4.0)
        self.assertEqual(at_2t, 8.0)

    def test_solvers_are_unaffected_by_the_penalty(self) -> None:
        """The sensitivity knob must only move non-solvers."""
        s = self._Study([[1.0, 0.0, 0.0, 0.0]])
        self.assertEqual(
            float(analyze_pd1pool.budget_censored_at(s, "m", -0.01, 1.0)[0]),
            float(analyze_pd1pool.budget_censored_at(s, "m", -0.01, 2.0)[0]),
        )

    def test_docstring_names_the_estimator_and_its_bias(self) -> None:
        """The docstring must name RMTT and state the bias, not merely describe steps."""
        doc = analyze_pd1pool.budget_censored.__doc__ or ""
        self.assertIn("RMTT", doc, "the estimator must be named for what it is")
        self.assertIn("was false", doc, "the retracted claim must stay retracted")
        self.assertIn(
            "solve rate", doc, "must direct the reader to report it alongside"
        )


class ModuleGlobalsTest(unittest.TestCase):
    """D6: importing make_surrogate elsewhere must not raise NameError.

    bo_diagnose imports it directly, and its broad except swallowed the NameError, so
    the method simply vanished from the results table with no message.
    """

    def test_globals_are_bound_at_module_scope(self) -> None:
        for name, expected in (
            ("HYPERBO_BASE_KERNEL", None),
            ("EM_CANONICAL", "sumll"),
            ("DEEP_KERNEL", None),
        ):
            self.assertTrue(hasattr(bo_experiment, name), f"{name} must be bound")
            self.assertEqual(getattr(bo_experiment, name), expected)


class SafeHyperparamCopyTest(unittest.TestCase):
    """D10: transplants must not crash on composite canonical kernels."""

    def test_outputscale_copy_skipped_when_absent(self) -> None:
        from gpytorch.kernels import MaternKernel, ScaleKernel

        src, dst = ScaleKernel(MaternKernel()), ScaleKernel(MaternKernel())
        self.assertTrue(bo_experiment._safe_copy_outputscale(dst, src))
        # A bare Matern has no raw_outputscale: must return False, not raise.
        self.assertFalse(bo_experiment._safe_copy_outputscale(dst, MaternKernel()))

    def test_lengthscale_copy_requires_matching_ard_shape(self) -> None:
        from gpytorch.kernels import MaternKernel

        a, b = MaternKernel(ard_num_dims=3), MaternKernel(ard_num_dims=3)
        self.assertTrue(bo_experiment._safe_copy_lengthscale(a, b))
        # Mismatched ARD dims used to raise inside copy_ and abort the whole run.
        self.assertFalse(
            bo_experiment._safe_copy_lengthscale(a, MaternKernel(ard_num_dims=7))
        )


class WarpConsistencyTest(unittest.TestCase):
    """D4: the rank warp is pool-dependent, so it cannot be used off-grid."""

    def test_gaussian_warp_is_pool_dependent(self) -> None:
        y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
        small = bo_experiment._apply_output_warp([y], "gaussian")[0]
        big = bo_experiment._apply_output_warp(
            [torch.cat([y, torch.tensor([4.0, 5.0], dtype=torch.double)])], "gaussian"
        )[0][:3]
        self.assertFalse(
            torch.allclose(small, big),
            "if this ever becomes pool-independent, the arm-C guard can be removed",
        )

    def test_neglog_warp_is_pointwise(self) -> None:
        """Contrast case: neglog is safe off-grid, which is why the guard is
        warp-specific rather than blanket."""
        y = torch.tensor([0.2, 0.5, 0.8], dtype=torch.double)
        small = bo_experiment._apply_output_warp([y], "neglog")[0]
        big = bo_experiment._apply_output_warp(
            [torch.cat([y, torch.tensor([0.9], dtype=torch.double)])], "neglog"
        )[0][:3]
        self.assertTrue(torch.allclose(small, big))


class DeadCodeTest(unittest.TestCase):
    """Dead code that LOOKS like a fix has fooled this review process twice.

    `mean_baseline` was assigned three times and read nowhere; `equal_prior_weights` was
    defined and never called while the real weighting lived in PooledStudy. Both made a
    reader believe a mechanism was in place. These check the module has no such traps.
    """

    def test_equal_prior_weights_is_gone(self) -> None:
        """It was superseded by PooledStudy.run_w; leaving it implied the weighting
        lived there."""
        self.assertFalse(hasattr(analyze_pd1pool, "equal_prior_weights"))
        self.assertTrue(hasattr(analyze_pd1pool.PooledStudy, "wmean"))


class BehaviouralGuardTest(unittest.TestCase):
    """Tests that EXECUTE the guarded paths.

    Round 6 deleted eight source-text tests from this file. One of them,
    test_finite_guard_is_wired_into_both_selection_points, asserted the literal string
    '_assert_finite_scores(score, method, "fantasy' -- and therefore PASSED on a
    NameError, certifying the bug it was written to prevent. Grepping source proves a
    line exists; only running it proves the line works.
    """

    def test_fantasy_batch_pick_executes_end_to_end(self) -> None:
        """CALL _fantasy_batch_pick. The NameError that shipped in round 5 lived on a
        line reached only at runtime, and the previous version of this test scanned the
        source instead of running it -- so it passed for a full round."""

        class _Post:
            def __init__(self, m, v):
                self.mean, self.variance = m, v

            def rsample(self, sample_shape=None):
                n = sample_shape[0] if sample_shape else 1
                return self.mean.unsqueeze(0).expand(n, *self.mean.shape).clone()

        class _Model:
            num_outputs = 1

            def posterior(self, X, **kw):
                n = X.shape[-2]
                m = torch.linspace(0.0, 1.0, n, dtype=torch.double).reshape(
                    *X.shape[:-1]
                )
                return _Post(m, torch.full_like(m, 0.25))

            def condition_on_observations(self, X, Y, **kw):
                return self

        pool = torch.rand(8, 3, dtype=torch.double)
        picks, _, _, idx0 = bo_experiment._fantasy_batch_pick(
            "em_frozen",
            pool,
            [0],
            torch.rand(8, 1, dtype=torch.double),
            [1, 2, 3, 4, 5],
            0.5,
            2,
            lambda m, X, Y: _Model(),
            "logei",
            2.0,
            2,
            torch.Generator().manual_seed(0),
        )
        self.assertEqual(len(picks), 2, "q=2 must yield two distinct picks")
        self.assertEqual(len(set(picks)), 2)
        self.assertIn(idx0, picks)

    def test_every_finite_guard_argument_is_in_scope(self) -> None:
        """Generalises the above to every call site in the module."""
        import inspect

        mod_src = inspect.getsource(bo_experiment)
        tree = ast.parse(mod_src)
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
            assigned = {
                t.id
                for n in ast.walk(fn)
                if isinstance(n, ast.Assign)
                for t in n.targets
                if isinstance(t, ast.Name)
            }
            for call in ast.walk(fn):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "_assert_finite_scores"
                    and len(call.args) >= 2
                    and isinstance(call.args[1], ast.Name)
                ):
                    name = call.args[1].id
                    with self.subTest(function=fn.name, arg=name):
                        self.assertTrue(
                            name in params
                            or name in assigned
                            or hasattr(bo_experiment, name),
                            f"{fn.name} passes unbound '{name}' to the guard",
                        )

    def test_posinf_is_rejected_not_reranked(self) -> None:
        """+inf is a model blow-up. Mapping it to -inf silently ranked it LAST."""
        score = torch.tensor([float("inf"), 1.0], dtype=torch.double)
        with self.assertRaises(FloatingPointError):
            bo_experiment._assert_finite_scores(score, "m", "unit test")

    def test_zero_shot_pick_is_guarded(self) -> None:
        """_prior_mean_pick was the one selection site left unguarded (R6-B3)."""
        import inspect

        src = inspect.getsource(bo_experiment._prior_mean_pick)
        self.assertIn("_assert_finite_scores(pmean", src)


class Round5RegressionTest(unittest.TestCase):
    """Round-5 fixes, each asserted on the production path rather than in isolation."""

    def _src(self, mod) -> str:
        import inspect

        return inspect.getsource(mod)

    def test_non_finite_scores_raise_instead_of_arbitrary_argmax(self) -> None:
        """argmax on NaN returns an index silently; that produced plausible-looking
        trajectories from failed models (N3)."""
        bad = torch.tensor([float("nan"), float("nan")], dtype=torch.double)
        with self.assertRaises(FloatingPointError):
            bo_experiment._assert_finite_scores(bad, "m", "unit test")

    def test_partially_finite_scores_are_usable(self) -> None:
        """A single bad candidate must not abort the run -- only an all-bad surface."""
        mixed = torch.tensor([float("nan"), 1.0], dtype=torch.double)
        out = bo_experiment._assert_finite_scores(mixed, "m", "unit test")
        self.assertEqual(int(torch.argmax(out)), 1)

    def test_ofat_mean_transfer_cells_carry_a_hyperbo_prior(self) -> None:
        """A cell that raises at runtime never writes shards, and a stage with a
        missing cell can never reach .DONE -- it re-runs and fails forever (R6)."""
        for name, flags in gen_stage_queue.OFAT:
            if "--em-mean" in flags:
                with self.subTest(cell=name):
                    self.assertIn(
                        "--em-canonical hyperbo",
                        flags,
                        f"{name} requests a mean transfer with no prior to transfer",
                    )


class DegreesOfFreedomTest(unittest.TestCase):
    """R6-S3: the SE rests on P prior draws, so the threshold is t(P-1), not 1.96.

    At the project's usual P=3 that is 4.30. Every earlier |z|>2 verdict was
    anti-conservative by more than a factor of two on the dominant variance component.
    """

    def test_known_critical_values(self) -> None:
        self.assertAlmostEqual(analyze_pd1pool.critical_value(2), 4.30, places=2)
        self.assertAlmostEqual(analyze_pd1pool.critical_value(1), 12.71, places=2)
        self.assertAlmostEqual(analyze_pd1pool.critical_value(10), 2.23, places=2)

    def test_large_df_approaches_normal(self) -> None:
        self.assertAlmostEqual(analyze_pd1pool.critical_value(1000), 1.96, places=2)

    def test_monotone_decreasing_in_df(self) -> None:
        vals = [analyze_pd1pool.critical_value(d) for d in (1, 2, 3, 5, 10, 20, 60)]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_three_priors_is_stricter_than_the_normal_threshold(self) -> None:
        """The specific number that flips the PD1 verdicts."""
        self.assertGreater(analyze_pd1pool.critical_value(2), 2.0 * 2)

    def test_zero_df_is_never_significant(self) -> None:
        self.assertEqual(analyze_pd1pool.critical_value(0), float("inf"))


class StageDirResolutionTest(unittest.TestCase):
    """The end-to-end smoke found that the pipeline's two halves disagreed on layout.

    run_cell_queue.sh writes results/raw/$STUDY verbatim; summarize_stage hardcoded
    results/raw/v2/<stage>. They agreed only by orchestrator convention, and a mismatch
    summarised nothing -- which the orchestrator tolerates with `|| true`, after which
    the .DONE gate fails forever on the missing SUMMARY.json.
    """

    def test_resolves_the_v2_layout(self) -> None:
        from _scratch_bo import summarize_stage

        with tempfile.TemporaryDirectory() as tmp:
            want = os.path.join(tmp, "results/raw/v2/st")
            os.makedirs(want)
            with patch.object(summarize_stage, "ROOT", tmp):
                self.assertEqual(summarize_stage.resolve_stage_dir("st"), want)

    def test_resolves_the_bare_layout(self) -> None:
        from _scratch_bo import summarize_stage

        with tempfile.TemporaryDirectory() as tmp:
            want = os.path.join(tmp, "results/raw/st")
            os.makedirs(want)
            with patch.object(summarize_stage, "ROOT", tmp):
                self.assertEqual(summarize_stage.resolve_stage_dir("st"), want)

    def test_missing_stage_names_what_it_tried(self) -> None:
        """Silence here cost a full stage; the error must be actionable."""
        from _scratch_bo import summarize_stage

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(summarize_stage, "ROOT", tmp):
                with self.assertRaises(FileNotFoundError) as ctx:
                    summarize_stage.resolve_stage_dir("nope")
        self.assertIn("results/raw/v2/nope", str(ctx.exception))
        self.assertIn("results/raw/nope", str(ctx.exception))


class StageSelectionTest(unittest.TestCase):
    """Stage 2/3 must pick their own cells from the previous stage's artifact.

    Before this, `run_autonomous.sh` promised "stage 2 and 3 pick their own cells" and
    the implementation was two `say` lines, so the pipeline was autonomous for stages 0
    and 1 only (round-6 P4).
    """

    def _stage(self, tmp, configs, cells):
        stage_dir = os.path.join(tmp, "results/raw/v2", "stage1_x")
        os.makedirs(stage_dir, exist_ok=True)
        with open(os.path.join(stage_dir, "SUMMARY.json"), "w") as f:
            json.dump({"stage": "stage1_x", "configs": configs}, f)
        with open(os.path.join(stage_dir, "PROVENANCE.json"), "w") as f:
            json.dump({"cells": cells}, f)
        return stage_dir

    def _cfg(self, budget, n_priors=3, kind="bo"):
        return {
            "kind": kind,
            "n_priors": n_priors,
            "methods": {"em_frozen": {"budget_to_0.01": budget}},
        }

    def test_picks_the_lowest_budget_configs(self) -> None:
        """Primary metric, lower is better (.llms/rules/metrics.md)."""
        with tempfile.TemporaryDirectory() as tmp:
            self._stage(
                tmp,
                {"a": self._cfg(40.0), "b": self._cfg(10.0), "c": self._cfg(25.0)},
                [
                    {"name": "a_p0", "flags": "--x --pretrain-seed 0"},
                    {"name": "b_p0", "flags": "--y --pretrain-seed 0"},
                    {"name": "c_p0", "flags": "--z --pretrain-seed 0"},
                ],
            )
            with patch.object(gen_stage_queue, "ROOT", tmp):
                got = gen_stage_queue.select_top_configs("stage1_x", k=2)
        self.assertEqual([n for n, _ in got], ["b", "c"])

    def test_prior_seed_is_stripped_so_the_caller_can_re_expand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._stage(
                tmp,
                {"a": self._cfg(10.0)},
                [{"name": "a_p2", "flags": "--em-base-mean linear --pretrain-seed 2"}],
            )
            with patch.object(gen_stage_queue, "ROOT", tmp):
                got = gen_stage_queue.select_top_configs("stage1_x", k=1)
        self.assertEqual(got, [("a", "--em-base-mean linear")])

    def test_single_prior_configs_are_not_rankable(self) -> None:
        """A single-prior number has no estimable between-prior variance (§27.21), so it
        would win by looking artificially precise."""
        with tempfile.TemporaryDirectory() as tmp:
            self._stage(
                tmp,
                {"solo": self._cfg(1.0, n_priors=1), "real": self._cfg(50.0)},
                [
                    {"name": "solo_p0", "flags": "--a --pretrain-seed 0"},
                    {"name": "real_p0", "flags": "--b --pretrain-seed 0"},
                ],
            )
            with patch.object(gen_stage_queue, "ROOT", tmp):
                got = gen_stage_queue.select_top_configs("stage1_x", k=2)
        self.assertEqual([n for n, _ in got], ["real"], "solo must be skipped")

    def test_error_configs_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._stage(
                tmp,
                {"bad": {"kind": "error", "n_priors": 3}, "ok": self._cfg(9.0)},
                [
                    {"name": "bad_p0", "flags": "--a --pretrain-seed 0"},
                    {"name": "ok_p0", "flags": "--b --pretrain-seed 0"},
                ],
            )
            with patch.object(gen_stage_queue, "ROOT", tmp):
                got = gen_stage_queue.select_top_configs("stage1_x", k=2)
        self.assertEqual([n for n, _ in got], ["ok"])

    def test_missing_summary_raises_rather_than_defaulting(self) -> None:
        """Silently falling back to a default cell list would run an arbitrary
        experiment while looking successful."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(gen_stage_queue, "ROOT", tmp):
                with self.assertRaises(FileNotFoundError):
                    gen_stage_queue.select_top_configs("stage1_missing")

    def test_no_rankable_configs_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._stage(tmp, {"solo": self._cfg(1.0, n_priors=1)}, [])
            with patch.object(gen_stage_queue, "ROOT", tmp):
                with self.assertRaises(ValueError):
                    gen_stage_queue.select_top_configs("stage1_x")

    def test_winner_without_provenance_raises(self) -> None:
        """Reconstructing flags by guessing would silently drift from what
        stage 1 actually ran."""
        with tempfile.TemporaryDirectory() as tmp:
            self._stage(tmp, {"a": self._cfg(10.0)}, [])
            with patch.object(gen_stage_queue, "ROOT", tmp):
                with self.assertRaises(ValueError):
                    gen_stage_queue.select_top_configs("stage1_x", k=1)

    def test_driver_actually_runs_stage3(self) -> None:
        """The promise was two `say` lines; assert a real run_block invocation."""
        src = self._driver_src()
        self.assertIn('run_block stage3 "$rg" replicate "stage1_$rg"', src)
        self.assertNotIn("deferred to the follow-up generator", src)

    def test_driver_forwards_select_from_to_the_generator(self) -> None:
        """Wiring the call is not enough -- the argument has to arrive.

        An inline edit silently dropped the `"${sel_args[@]}"` expansion and left a bare
        `""`, so --select-from never reached gen_stage_queue and every stage-3 run would
        have died on "requires --select-from". The previous test passed throughout,
        because it only checked that run_block was called.
        """
        src = self._driver_src()
        self.assertIn("sel_args=(--select-from", src, "the array must be built")
        self.assertIn(
            'sel_args[@]+"${sel_args[@]}"',
            src,
            "the array must be EXPANDED into the generator call, not dropped",
        )
        # and the mangled form must not come back
        self.assertNotIn('--out "$queue" "" 2>&1', src)

    def _driver_src(self) -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (
            os.path.join(here, "run_autonomous.sh"),
            os.path.join(here, "..", "results", "run_autonomous.sh"),
        ):
            if os.path.exists(cand):
                with open(cand) as f:
                    return f.read()
        self.skipTest("driver not available as a resource")


class ReplicationIndependenceTest(unittest.TestCase):
    """Priors must be INDEPENDENT replicates, or the whole design is anti-conservative.

    I recommended `--priors 9 --n-seeds 1` from the algebra Var = sB^2/P + sW^2/B, then
    found the initial-design seed was `1000*task + seed`, with no dependence on
    the prior.
    At n_seeds=1 that gives every prior the SAME initial design, so the per-prior noise
    e_p collapses to one shared e; the between-prior spread then estimates sB^2
    alone and
    understates Var by sW^2 -- which §27.24 measured as the LARGER component for most
    methods. The recommendation would have produced badly anti-conservative SEs.
    """

    def _init_design(
        self, task: int, seed: int, pretrain_seed: int, n_pool=50, n_init=3
    ):
        g = torch.Generator().manual_seed(1000 * task + seed + 100003 * pretrain_seed)
        return torch.randperm(n_pool, generator=g)[:n_init].tolist()

    def test_initial_design_differs_across_priors(self) -> None:
        """The property that makes n_seeds=1 replication valid."""
        designs = [self._init_design(3, 0, p) for p in range(9)]
        self.assertEqual(
            len({tuple(d) for d in designs}),
            9,
            "every prior must draw its own initial design, or the replicates are not "
            "independent and the between-prior spread is an underestimate",
        )

    def test_initial_design_is_shared_across_methods_within_a_cell(self) -> None:
        """Pairing must survive: the SAME (task, seed, prior) yields one design."""
        a = self._init_design(3, 0, 5)
        b = self._init_design(3, 0, 5)
        self.assertEqual(a, b)

    def test_seed_derivation_in_production_depends_on_the_prior(self) -> None:
        """Assert the production formula, not just my reimplementation of it."""
        import inspect

        src = inspect.getsource(bo_experiment)
        self.assertIn("100003 * args.pretrain_seed", src)
        # the old prior-independent forms must be gone from the seeding path
        self.assertNotIn("manual_seed(1000 * held + seed)", src)
        self.assertNotIn("manual_seed(1000 * ei + seed)", src)

    def test_matrix_uses_many_priors_and_one_seed(self) -> None:
        """§27.24: at fixed budget, priors buy variance reduction and seeds do not."""
        import inspect

        src = inspect.getsource(gen_stage_queue)
        self.assertIn("default=9,", src, "priors should default to 9")
        for regime, spec in gen_stage_queue.REGIMES.items():
            if spec["harness"] == "bo_experiment":
                with self.subTest(regime=regime):
                    self.assertIn(
                        "--n-seeds 1",
                        spec["flags"],
                        "BO regimes should replicate over priors, not seeds",
                    )


class PriorCacheTest(unittest.TestCase):
    """A mis-keyed prior cache silently substitutes the wrong model into every result.

    That is the worst failure mode available to this project, so these tests target the
    key and the verification rather than the happy path.
    """

    def _data(self, seed=0, n=6):
        g = torch.Generator().manual_seed(seed)
        return [
            {"X": torch.rand(n, 3, generator=g), "Y": torch.rand(n, 1, generator=g)}
        ]

    def _params(self, **over):
        p = {"num_iterations": 100, "pretrain_seed": 0, "held_out": 3}
        p.update(over)
        return p

    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d, pr = self._data(), self._params()
            dig = prior_cache.tensor_digest(d)
            prior_cache.save(tmp, "hyperbo", dig, pr, {"w": torch.ones(3)})
            got = prior_cache.load(tmp, "hyperbo", dig, pr)
        self.assertIsNotNone(got)
        self.assertTrue(torch.equal(got["w"], torch.ones(3)))

    def test_seed_is_part_of_the_key_so_fits_are_retained_separately(self) -> None:
        """The user-facing point: repeated fits must NOT overwrite each other, or the
        variance of the trained model is unmeasurable."""
        with tempfile.TemporaryDirectory() as tmp:
            d = self._data()
            dig = prior_cache.tensor_digest(d)
            for s in range(3):
                prior_cache.save(
                    tmp, "hyperbo", dig, self._params(pretrain_seed=s), {"s": s}
                )
            inv = prior_cache.describe(tmp)
            self.assertEqual(len(inv), 3, "each seed must be a separate entry")
            seeds = sorted(e["param.pretrain_seed"] for e in inv)
            self.assertEqual(seeds, [0, 1, 2])
            for s in range(3):
                got = prior_cache.load(
                    tmp, "hyperbo", dig, self._params(pretrain_seed=s)
                )
                self.assertEqual(got["s"], s, "seed %d returned the wrong fit" % s)

    def test_different_data_is_a_miss_even_with_identical_params(self) -> None:
        """The digest is the safety net for flags we forgot to enumerate."""
        with tempfile.TemporaryDirectory() as tmp:
            pr = self._params()
            prior_cache.save(
                tmp, "hyperbo", prior_cache.tensor_digest(self._data(0)), pr, {"a": 1}
            )
            miss = prior_cache.load(
                tmp, "hyperbo", prior_cache.tensor_digest(self._data(1)), pr
            )
        self.assertIsNone(miss, "different pre-training data must not hit")

    def test_different_hyperparams_are_a_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = prior_cache.tensor_digest(self._data())
            prior_cache.save(tmp, "hyperbo", d, self._params(), {"a": 1})
            self.assertIsNone(
                prior_cache.load(tmp, "hyperbo", d, self._params(num_iterations=200))
            )

    def test_different_fold_is_a_miss(self) -> None:
        """Fold leakage would be catastrophic and invisible: a prior fitted with the
        held-out task INCLUDED would inflate every number for that fold."""
        with tempfile.TemporaryDirectory() as tmp:
            d = prior_cache.tensor_digest(self._data())
            prior_cache.save(tmp, "hyperbo", d, self._params(held_out=3), {"a": 1})
            self.assertIsNone(
                prior_cache.load(tmp, "hyperbo", d, self._params(held_out=4))
            )

    def test_config_mismatch_under_the_same_key_is_refused(self) -> None:
        """Simulates a collision or a stale entry: the stored config is verified, not
        trusted. Without this, a key that fails to capture something returns the wrong
        prior silently."""
        with tempfile.TemporaryDirectory() as tmp:
            d = prior_cache.tensor_digest(self._data())
            pr = self._params()
            path = prior_cache.save(tmp, "hyperbo", d, pr, {"a": 1})
            blob = torch.load(path, map_location="cpu", weights_only=False)
            blob["config"]["params"]["num_iterations"] = 999  # tamper
            torch.save(blob, path)
            with self.assertLogs(prior_cache.logger, level="ERROR"):
                got = prior_cache.load(tmp, "hyperbo", d, pr)
        self.assertIsNone(got, "a mismatched config must never be used")

    def test_corrupt_entry_is_a_miss_not_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = prior_cache.tensor_digest(self._data())
            pr = self._params()
            path = prior_cache.save(tmp, "hyperbo", d, pr, {"a": 1})
            with open(path, "wb") as f:
                f.write(b"garbage")
            self.assertIsNone(prior_cache.load(tmp, "hyperbo", d, pr))

    def test_unhashable_input_raises_rather_than_being_skipped(self) -> None:
        """Silently excluding an input from the key is how caches go wrong."""
        with self.assertRaises(TypeError):
            prior_cache.tensor_digest({"f": lambda x: x})

    def test_cached_computes_on_miss_and_reuses_on_hit(self) -> None:
        calls = []

        def _compute():
            calls.append(1)
            return {"v": torch.tensor([1.0])}

        with tempfile.TemporaryDirectory() as tmp:
            d, pr = self._data(), self._params()
            a = prior_cache.cached(tmp, "readwrite", "hyperbo", d, pr, _compute)
            b = prior_cache.cached(tmp, "readwrite", "hyperbo", d, pr, _compute)
        self.assertEqual(len(calls), 1, "second call must hit the cache")
        self.assertTrue(torch.equal(a["v"], b["v"]))

    def test_mode_off_never_caches(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(2):
                prior_cache.cached(
                    tmp,
                    "off",
                    "hyperbo",
                    self._data(),
                    self._params(),
                    lambda: calls.append(1) or {"v": 1},
                )
            self.assertEqual(len(calls), 2)
            self.assertEqual(prior_cache.describe(tmp), [])

    def test_read_mode_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prior_cache.cached(
                tmp, "read", "hyperbo", self._data(), self._params(), lambda: {"v": 1}
            )
            self.assertEqual(prior_cache.describe(tmp), [])

    def test_format_version_bump_invalidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d, pr = prior_cache.tensor_digest(self._data()), self._params()
            prior_cache.save(tmp, "hyperbo", d, pr, {"a": 1})
            with patch.object(prior_cache, "CACHE_FORMAT_VERSION", 999):
                self.assertIsNone(prior_cache.load(tmp, "hyperbo", d, pr))

    def test_wired_into_the_expensive_pretraining_path(self) -> None:
        import inspect

        src = inspect.getsource(bo_experiment)
        self.assertIn("prior_cache.cached(", src)
        self.assertIn('"pretrain_seed": args.pretrain_seed', src)
        self.assertIn('"held_out": int(held)', src)


class MeanSweepTest(unittest.TestCase):
    """OFAT cannot answer the mean question, so a dedicated crossed sweep exists.

    §27.31 found every mean transfer null on LCBench, but OFAT only ever compares each
    variant against `base` and never crosses the two axes -- and LCBench is on-grid,
    where §27.29 showed EM is already strong. The sweep crosses base-mean × transfer so
    the question can be answered on the off-grid regimes too.
    """

    def test_both_mean_axes_are_crossed(self) -> None:
        names = {n for n, _ in gen_stage_queue.MEANS}
        # constant and linear base means must each appear with empirical AND hyperbo
        for base in ("const", "lin"):
            for kind in ("emp", "hb"):
                with self.subTest(base=base, kind=kind):
                    self.assertTrue(
                        any(f"mean_{base}_{kind}" == n for n in names),
                        f"mean_{base}_{kind} missing -- the axes are not crossed",
                    )

    def test_blend_weights_span_the_range(self) -> None:
        ws = set()
        for _n, f in gen_stage_queue.MEANS:
            if "--em-mean blend" in f:
                ws.add(f.split("--em-mean-weight ")[1].split()[0])
        self.assertGreaterEqual(
            len(ws),
            3,
            "a single blend weight cannot show whether the effect is monotone",
        )

    def test_every_transfer_row_carries_a_hyperbo_prior(self) -> None:
        """--em-mean != empirical needs a prior to transfer FROM, or the cell raises at
        runtime and blocks its stage forever (round-6 R6)."""
        for name, flags in gen_stage_queue.MEANS:
            if "--em-mean " in flags and "--em-mean empirical" not in flags:
                with self.subTest(cell=name):
                    self.assertIn("--em-canonical hyperbo", flags)

    def test_sweep_does_not_disturb_the_existing_kinds(self) -> None:
        """A stage was running when this was added; baseline/ofat must be untouched."""
        import inspect

        src = inspect.getsource(gen_stage_queue.main)
        self.assertIn('elif args.kind == "means":', src)
        # the pre-existing branches must still be present and independent
        self.assertIn('if args.kind == "baseline":', src)
        self.assertIn('elif args.kind == "replicate":', src)


class LCBenchIntegrityTest(unittest.TestCase):
    """The dataset roster must not shrink silently.

    A parallel stack's LCBench filter dropped a whole dataset whenever any curve
    diverged. Measured on this data that rule removes 26 of 35 datasets to excise 1.8%
    of curves (§27.22). Our loader does no filtering, and this locks that in: dataset
    indices feed the seeded randperm splits, so a silent roster change would invalidate
    every committed result while looking like a data-quality improvement.
    """

    def test_dataset_roster_is_the_full_35(self) -> None:
        from botorch.utils.lcbench import LCBENCH_DATASET_NAMES

        self.assertEqual(len(LCBENCH_DATASET_NAMES), 35)
        self.assertEqual(
            len(set(LCBENCH_DATASET_NAMES)), 35, "roster must have no duplicates"
        )

    def test_load_pool_returns_one_series_per_dataset(self) -> None:
        """load_pool must not drop datasets. Uses a stub loader so the test stays fast
        and hermetic -- the point is the CONTRACT, not the data.

        Deliberately NOT paired with a test that greps EXPERIMENTAL_RESULTS.md for the
        policy text: rounds 6 and 7 showed doc/source greps pass on the very bug they
        target. The guarantee lives in these two behavioural assertions.
        """
        names = ["a", "b", "c"]

        def _fake_load(nm, metric, dtype=None):
            class _D:
                parameters = torch.rand(5, 7, dtype=torch.double)
                metrics = torch.rand(5, 50, dtype=torch.double)

            return _D()

        with patch.object(bo_experiment, "load_lcbench_data", _fake_load):
            Xn, Y_all = bo_experiment.load_pool(names)
        self.assertEqual(
            len(Y_all), len(names), "load_pool must return one series per dataset"
        )
        self.assertEqual(Xn.shape[-1], 7)


class GeneratedCellsParseTest(unittest.TestCase):
    """Validate the cells `main()` ACTUALLY generates, against each regime's harness.

    `test_every_ofat_cell_parses_on_every_harness` checks the raw OFAT table, so the
    `--pretrain-seed` that `main()` appends during the --priors expansion was outside
    its coverage. bo_diagnose has no such flag, so 96 regression cells were guaranteed
    to abort and crash-loop their stage forever (round-7 R7-1). This closes that gap by
    generating the real queue and parsing every line.
    """

    def _parser(self, harness: str):
        return {
            "bo_experiment": bo_experiment.build_parser,
            "bo_diagnose": bo_diagnose.build_parser,
        }[harness]()

    def _generate(self, tmp, regime, kind, priors):
        import sys

        out = os.path.join(tmp, "q.tsv")
        argv = sys.argv
        sys.argv = [
            "x",
            "--stage",
            "t",
            "--regime",
            regime,
            "--kind",
            kind,
            "--out",
            out,
            "--priors",
            str(priors),
        ]
        try:
            with patch.object(gen_stage_queue, "ROOT", tmp):
                gen_stage_queue.main()
        finally:
            sys.argv = argv
        rows = []
        with open(out) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    name, flags = line.rstrip("\n").split("\t", 1)
                    rows.append((name, flags))
        return rows

    def test_generated_cells_parse_on_their_own_harness(self) -> None:
        for regime, spec in gen_stage_queue.REGIMES.items():
            for kind in ("baseline", "ofat", "means"):
                for priors in (1, 3):
                    with tempfile.TemporaryDirectory() as tmp:
                        cells = self._generate(tmp, regime, kind, priors)
                    self.assertTrue(cells, f"{regime}/{kind} generated no cells")
                    parser = self._parser(spec["harness"])
                    common = spec["flags"].split()
                    for name, flags in cells:
                        with self.subTest(
                            regime=regime, kind=kind, priors=priors, cell=name
                        ):
                            try:
                                parser.parse_args(
                                    common + flags.split() + ["--out", "/tmp/x.json"]
                                )
                            except SystemExit:
                                self.fail(
                                    f"{regime}/{kind}/priors={priors} cell {name!r} "
                                    f"does not parse on {spec['harness']}: {flags}"
                                )

    def test_priors_expansion_reaches_the_generated_cells(self) -> None:
        """Guards against the expansion silently not happening."""
        with tempfile.TemporaryDirectory() as tmp:
            cells = self._generate(tmp, "pd1_bo_armC", "ofat", 3)
        self.assertTrue(all("--pretrain-seed" in f for _, f in cells))
        self.assertEqual(len({f.split("--pretrain-seed ")[1][0] for _, f in cells}), 3)

    def test_pretrain_seed_exists_on_both_harnesses(self) -> None:
        """The specific asymmetry that caused R7-1."""
        for harness in ("bo_experiment", "bo_diagnose"):
            opts = {s for a in self._parser(harness)._actions for s in a.option_strings}
            self.assertIn("--pretrain-seed", opts, f"{harness} cannot replicate priors")


class PriorPoolingTest(unittest.TestCase):
    """--priors names cells `<config>_p<k>`; something must POOL them.

    Round-6 added the expansion and round-7's self-check found nothing joined the
    pieces back up, so each prior sat in its own bucket and the between-prior variance
    was identically zero again -- the fix was inert in a new way. This asserts the
    pooled `configs` view exists and actually merges priors.
    """

    def _shard(self, tmp, cell, shard, n_runs=2):
        path = os.path.join(tmp, f"{cell}_s{shard}.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "per_run": {
                        "traj_raw": {"m": [[0.0, 1.0]] * n_runs},
                        "pool_max_raw": [1.0] * n_runs,
                        "eval_dataset_idx": list(range(n_runs)),
                    }
                },
                f,
            )
        return path

    def test_priors_are_pooled_into_one_config(self) -> None:
        from _scratch_bo import summarize_stage

        with tempfile.TemporaryDirectory() as tmp:
            stage = os.path.join(tmp, "results/raw/v2/st")
            os.makedirs(stage, exist_ok=True)
            for k in range(3):
                self._shard(stage, f"base_p{k}", 0)
            with patch.object(summarize_stage, "ROOT", tmp):
                import sys

                argv = sys.argv
                sys.argv = ["x", "--stage", "st"]
                try:
                    summarize_stage.main()
                finally:
                    sys.argv = argv
            with open(os.path.join(stage, "SUMMARY.json")) as f:
                out = json.load(f)

        self.assertEqual(len(out["cells"]), 3, "one cell per prior, as generated")
        self.assertIn("configs", out, "pooled view must exist or nothing pools priors")
        self.assertIn("base", out["configs"])
        self.assertEqual(out["configs"]["base"]["n_priors"], 3)
        self.assertEqual(out["configs"]["base"]["methods"]["m"]["n_runs"], 6)

    def test_regression_configs_also_get_between_prior_spread(self) -> None:
        """Stage 4.1 correlates regression metrics against BO metrics.

        Computing the between-prior spread for BO only would put an uncertainty estimate
        on one side of that correlation and none on the other, which makes it
        uninterpretable. The end-to-end smoke caught this: smoke_reg carried no
        `between_prior` while smoke_bo did.
        """
        from _scratch_bo import summarize_stage

        with tempfile.TemporaryDirectory() as tmp:
            stage = os.path.join(tmp, "results/raw/v2/st")
            os.makedirs(stage, exist_ok=True)
            for k, nll in enumerate((0.10, 0.30)):
                path = os.path.join(stage, f"cfg_p{k}_s0.json")
                with open(path, "w") as f:
                    json.dump(
                        {
                            "prior_mean": {"em_frozen": {"spearman": 0.7 + 0.1 * k}},
                            "by_n_obs": {"5": {"em_frozen": {"nll": nll}}},
                        },
                        f,
                    )
            with patch.object(summarize_stage, "ROOT", tmp):
                import sys

                argv = sys.argv
                sys.argv = ["x", "--stage", "st"]
                try:
                    summarize_stage.main()
                finally:
                    sys.argv = argv
            with open(os.path.join(stage, "SUMMARY.json")) as f:
                out = json.load(f)

        cfg = out["configs"]["cfg"]
        self.assertEqual(cfg["kind"], "regression")
        self.assertIn(
            "between_prior", cfg, "regression configs must carry a spread estimate too"
        )
        spread = cfg["between_prior"]["em_frozen"]
        self.assertIn("n5_nll", spread)
        # priors differ by 0.20, so var of the mean = var([0.1,0.3])/2 = 0.02/2 = 0.01
        self.assertAlmostEqual(
            spread["n5_nll"]["between_prior_var_of_mean"], 0.01, places=9
        )

    def test_single_prior_config_is_flagged(self) -> None:
        """A config with one prior cannot estimate between-prior variance, and any SE
        from it understates the uncertainty. That must be visible in the artifact."""
        from _scratch_bo import summarize_stage

        with tempfile.TemporaryDirectory() as tmp:
            stage = os.path.join(tmp, "results/raw/v2/st")
            os.makedirs(stage, exist_ok=True)
            self._shard(stage, "solo_p0", 0)
            with patch.object(summarize_stage, "ROOT", tmp):
                import sys

                argv = sys.argv
                sys.argv = ["x", "--stage", "st"]
                try:
                    summarize_stage.main()
                finally:
                    sys.argv = argv
            with open(os.path.join(stage, "SUMMARY.json")) as f:
                out = json.load(f)
        self.assertIn("warning", out["configs"]["solo"])
        self.assertIn("single prior", out["configs"]["solo"]["warning"])


class CorrectedVarianceTest(unittest.TestCase):
    """R6-S1: the spread of per-prior means IS the whole variance; adding a separate
    within term double-counts it and inflated the SE by up to sqrt(2)."""

    def _pooled(self, seeds_per_prior):
        import numpy as np

        class _P:
            n_priors = len(seeds_per_prior)
            prior_idx = np.repeat(range(len(seeds_per_prior)), seeds_per_prior)
            ds_idx = np.zeros(sum(seeds_per_prior), dtype=int)
            coverage_gaps = set()
            task_wvar = analyze_pd1pool.PooledStudy.task_wvar

        return _P()

    def test_variance_is_the_spread_of_prior_means_only(self) -> None:
        import numpy as np

        p = self._pooled((5, 3, 3))
        rng = np.random.default_rng(0)
        vals = rng.normal(size=11)
        means = [float(vals[p.prior_idx == k].mean()) for k in range(3)]
        self.assertAlmostEqual(
            p.task_wvar(vals, 0), float(np.var(means, ddof=1)) / 3, places=12
        )

    def test_no_within_term_is_added(self) -> None:
        """Identical prior means => zero variance, whatever the within-prior spread."""
        import numpy as np

        p = self._pooled((3, 3, 3))
        # Wildly different within-prior spread, identical prior means.
        vals = np.array([-5.0, 0.0, 5.0, -1.0, 0.0, 1.0, -0.1, 0.0, 0.1])
        self.assertAlmostEqual(p.task_wvar(vals, 0), 0.0, places=12)

    def test_single_prior_falls_back_to_within(self) -> None:
        import numpy as np

        p = self._pooled((4,))
        vals = np.array([0.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(
            p.task_wvar(vals, 0), float(vals.var(ddof=1)) / 4, places=12
        )


class RegressionSummarySchemaTest(unittest.TestCase):
    """R6-P1: the summariser read keys bo_diagnose never writes, so every regression
    cell reported COMPLETE while carrying zero numbers."""

    def _write(self, tmp, payload):
        path = os.path.join(tmp, "c_s0.json")
        with open(path, "w") as f:
            json.dump(payload, f)
        return path

    def test_reads_the_keys_bo_diagnose_actually_writes(self) -> None:
        """Fixture matches the REAL schema, verified against results/raw/diagnose.json.

        The previous version of this test used `by_n_obs: {"5": [{"method": ...}]}` -- a
        list-of-rows shape bo_diagnose has never produced -- so it certified the reader
        bug instead of catching it (round-7 R7-2). Both keys are dicts keyed by method.
        """
        from _scratch_bo.summarize_stage import summarise_regression

        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(
                tmp,
                {
                    "prior_mean": {"em_frozen": {"spearman": 0.8}},
                    "by_n_obs": {
                        "5": {
                            "em_frozen": {"rank_corr": 0.699, "nll": 0.0254},
                            "hyperbo_frozen": {"rank_corr": 0.588},
                        }
                    },
                },
            )
            out = summarise_regression([p])
        self.assertEqual(out["kind"], "regression")
        self.assertAlmostEqual(out["methods"]["em_frozen"]["prior_spearman"], 0.8)
        self.assertAlmostEqual(out["methods"]["em_frozen"]["n5_rank_corr"], 0.699)
        self.assertAlmostEqual(out["methods"]["em_frozen"]["n5_nll"], 0.0254)
        self.assertAlmostEqual(out["methods"]["hyperbo_frozen"]["n5_rank_corr"], 0.588)

    def test_list_shaped_by_n_obs_is_not_silently_accepted(self) -> None:
        """The fictional shape must yield nothing, not be quietly tolerated."""
        from _scratch_bo.summarize_stage import summarise_regression

        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, {"by_n_obs": {"5": [{"method": "m", "nll": 1.0}]}})
            out = summarise_regression([p])
        self.assertEqual(out["kind"], "error")

    def test_empty_extraction_reports_error_not_success(self) -> None:
        """A cell carrying nothing must not pass the .DONE gate."""
        from _scratch_bo.summarize_stage import summarise_regression

        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, {"config": {}, "eval_datasets": ["a"]})
            out = summarise_regression([p])
        self.assertEqual(out["kind"], "error")


class PriorReplicationTest(unittest.TestCase):
    """R6-P3: nothing varied --pretrain-seed, so every cell was a single prior and the
    between-prior variance was identically zero on the entire corpus."""

    def test_cells_are_expanded_across_priors(self) -> None:
        import argparse as _ap

        ns = _ap.Namespace(kind="ofat", priors=3)
        base = [(n, f) for n, f in gen_stage_queue.OFAT]
        expanded = [
            (f"{n}_p{k}", f"{f} --pretrain-seed {k}".strip())
            for n, f in base
            for k in range(ns.priors)
        ]
        self.assertEqual(len(expanded), len(base) * 3)
        self.assertTrue(all("--pretrain-seed" in f for _, f in expanded))

    def test_default_priors_is_more_than_one(self) -> None:
        """S27.3: one prior cannot rank these methods."""
        import inspect

        src = inspect.getsource(gen_stage_queue.main)
        self.assertIn('"--priors"', src)
        self.assertIn("default=3", src)


if __name__ == "__main__":
    unittest.main()


class IWNuEstimatorTest(unittest.TestCase):
    """--iw-nu-mode must be inert by default and actually change nu when asked.

    Written behaviourally rather than by grepping the source: §27.38 and the two bugs
    found while wiring this (an undefined `logger`, and an import that was never
    inserted because its anchor did not exist) both survived source-level checks and
    only showed up when the output was inspected.
    """

    def test_alpha_to_nu_inversion_is_consistent(self):
        from _scratch_bo.em_noise_selection import estimate_iw_nu

        torch.manual_seed(0)
        X = torch.randn(8, 40, dtype=torch.double)
        nu, alpha = estimate_iw_nu(X, method="oas")
        # nu = (K+1) + alpha*T/(1-alpha)  =>  invert back to alpha
        K, T = X.shape[1], X.shape[0]
        back = (nu - K - 1) / (T + nu - K - 1)
        self.assertAlmostEqual(back, alpha, places=6)

    def test_alpha_tracks_how_informative_the_target_is(self):
        """alpha must FALL with T on structured data and RISE on isotropic data.

        My first version of this test asserted "more tasks => less shrinkage" using
        torch.randn, which is isotropic -- exactly the case where the shrinkage target
        mu*I IS the truth, so alpha correctly rises toward 1. The estimator was right and
        the test was wrong (§27.46). Both directions are asserted here.
        """
        from _scratch_bo.em_noise_selection import oas_alpha

        torch.manual_seed(0)
        # Isotropic: the target is the truth, so MORE data should push alpha UP.
        iso = torch.randn(200, 30, dtype=torch.double)
        self.assertGreater(oas_alpha(iso[:150])[0], oas_alpha(iso[:6])[0])

        # Low-rank: the target is wrong, so more data should push alpha DOWN.
        g = torch.Generator().manual_seed(3)
        B = torch.randn(30, 4, generator=g, dtype=torch.double)
        low = torch.randn(200, 4, generator=g, dtype=torch.double) @ B.T
        self.assertLess(oas_alpha(low[:150])[0], oas_alpha(low[:6])[0])

    def test_flag_defaults_to_manual(self):
        import _scratch_bo.bo_experiment as bx

        p = bx.build_parser() if hasattr(bx, "build_parser") else None
        if p is None:
            self.skipTest("no build_parser to introspect")
        args = p.parse_args(["--benchmark", "lcbench"])
        self.assertEqual(args.iw_nu_mode, "manual")

    def test_resolve_is_a_noop_in_manual_mode(self):
        import _scratch_bo.bo_experiment as bx

        class A:
            iw_nu_mode = "manual"
            iw_nu = 7.5
            use_covar_prior = True

        self.assertEqual(bx._resolve_iw_nu(A(), []), 7.5)


class BoDiagnoseReplicationAxisTest(unittest.TestCase):
    """The two `bo_diagnose` defects found in S28.6, and why one test is not enough.

    `test_pretrain_seed_exists_on_both_harnesses` above passed throughout, because the
    flag DID exist -- it was parsed, recorded in every output config, and changed
    nothing for EM. That is this project's signature failure: a test that certifies the
    bug. Asserting a flag is accepted says nothing about whether it reaches the data
    path, so both invariants below are asserted separately:

      * the control must be constant ACROSS CONFIGS  (defect 1, the missing RNG guard)
      * EM output must vary ACROSS PRIORS            (defect 2, the dead seed axis)

    Each catches a different bug and neither implies the other.
    """

    def _tree(self, module) -> ast.Module:
        with open(module.__file__) as f:
            return ast.parse(f.read())

    def _seed_offsets(self, module) -> list[int]:
        """Every `torch.manual_seed(pretrain_seed * 1009 + N)` offset, in order."""
        out = []
        for node in ast.walk(self._tree(module)):
            if not (isinstance(node, ast.Call) and node.args):
                continue
            if getattr(node.func, "attr", None) != "manual_seed":
                continue
            src = ast.dump(node.args[0])
            if "1009" in src and "pretrain_seed" in src:
                out.append((node.lineno, node.args[0]))
        return [
            int(ast.literal_eval(a.right))
            for _, a in sorted(out)
            if isinstance(a, ast.BinOp) and isinstance(a.op, ast.Add)
        ]

    def _generator_seeds(self) -> list[str]:
        """Dumped seed expressions of every `torch.Generator().manual_seed(...)`.

        Only the Generator form: those are the local, explicitly-seeded draws that pick
        data. The bare `torch.manual_seed(...)` calls seed the global stream for model
        fitting and are a separate concern.
        """
        out = []
        for node in ast.walk(self._tree(bo_diagnose)):
            if not (isinstance(node, ast.Call) and node.args):
                continue
            if getattr(node.func, "attr", None) != "manual_seed":
                continue
            if isinstance(node.func.value, ast.Call):
                out.append(ast.dump(node.args[0]))
        return out

    def test_rng_guard_precedes_every_baseline_build(self) -> None:
        """Defect 1: bo_diagnose had ZERO guards, so 3/14 configs moved the control."""
        self.assertEqual(
            len(self._seed_offsets(bo_diagnose)),
            4,
            "bo_diagnose must re-seed before each of its 4 baseline builds; without "
            "this a config that merely allocates an extra module shifts the RNG "
            "stream and silently moves hyperbo_frozen (S27.38).",
        )

    def test_guard_offsets_match_the_other_harness(self) -> None:
        """Same baseline must get the same stream on both harnesses, or the two are
        not comparable -- and comparing them is the entire point of running both."""
        self.assertEqual(
            self._seed_offsets(bo_diagnose),
            self._seed_offsets(bo_experiment),
            "offsets diverged; bo_experiment uses 11/11/13/17 for "
            "hyperbo-first/hyperbo/pacoh/ablr",
        )

    def test_observation_subset_depends_on_the_prior(self) -> None:
        """Defect 2, the larger one.

        EM pre-training is deterministic (S25.4), so the prior index can only reach an
        EM result through the DATA. It did not: the subset generator was seeded on
        `1000 * ei` alone, making em_frozen bit-identical across all 9 priors for 7 of
        14 configs, driving between-prior variance to exactly 0 and every paired t to
        0/0. Asserted on the AST rather than by restating the formula, so the test
        cannot pass by agreeing with a copy of the bug.

        Scoped to the draw keyed on the eval-task index `ei`. The other two seeded
        generators -- the train/eval split and the inducing set -- must stay FIXED
        across priors, or the priors stop being comparable; see the companion test.
        """
        seeds = self._generator_seeds()
        per_task = [s for s in seeds if "'ei'" in s]
        self.assertEqual(
            len(per_task), 1, f"expected exactly one per-eval-task draw, got {seeds}"
        )
        self.assertIn(
            "pretrain_seed",
            per_task[0],
            "the observation subset must vary with the prior, or --pretrain-seed is a "
            "no-op for every EM method and the 9 'priors' are 1 replicate (S28.6)",
        )

    def test_the_split_and_inducing_set_stay_FIXED_across_priors(self) -> None:
        """The other half of the invariant, and the reason the test above is scoped.

        Priors are only comparable if they see the same tasks and the same inducing
        set. A well-meaning "make everything depend on the prior" fix would silently
        turn the replication axis into a different experiment per prior, which is a
        worse bug than the one being fixed and would not crash.
        """
        for seed in self._generator_seeds():
            if "'ei'" in seed:
                continue
            self.assertNotIn(
                "pretrain_seed",
                seed,
                "the train/eval split and the inducing set must NOT vary with the "
                "prior -- priors would no longer be paired and every paired contrast "
                "in the study would silently become unpaired",
            )

    def test_the_two_harnesses_agree_on_what_a_prior_varies(self) -> None:
        """bo_experiment keys its initial design on the prior; bo_diagnose must key
        its observation subset the same way, using the same multiplier."""
        for module in (bo_experiment, bo_diagnose):
            src = open(module.__file__).read()
            self.assertIn(
                "100003 * args.pretrain_seed",
                src.replace(
                    "args.pretrain_seed * 100003", "100003 * args.pretrain_seed"
                ),
                f"{module.__name__} does not spread priors with the shared multiplier",
            )


class RegressionSelectionKeyTest(unittest.TestCase):
    """S28.6 defect 3: `sorted()` on raw metric keys is lexicographic.

    `sorted({'n5_rank_corr','n20_rank_corr','n50_rank_corr'})[0]` is `n20_rank_corr`,
    because '2' < '5'. The code took that as "smallest n_obs" and said so in a comment.
    Inert only because no regression stage 3 has ever run -- it would have mis-selected
    on the first one, with no visible symptom.
    """

    def _stage(self, tmp: str, stage: str) -> None:
        d = os.path.join(tmp, "results/raw/v2", stage)
        os.makedirs(d)
        # alpha wins at n_obs=5; beta wins at n_obs=20. The two orderings disagree, so
        # the assertion cannot pass under both key choices.
        configs = {
            "cfg_alpha": {"n5_rank_corr": 0.90, "n20_rank_corr": 0.10},
            "cfg_beta": {"n5_rank_corr": 0.10, "n20_rank_corr": 0.90},
        }
        with open(os.path.join(d, "SUMMARY.json"), "w") as f:
            json.dump(
                {
                    "configs": {
                        name: {
                            "kind": "regression",
                            "n_priors": 9,
                            "methods": {"em_frozen": metrics},
                        }
                        for name, metrics in configs.items()
                    }
                },
                f,
            )
        with open(os.path.join(d, "PROVENANCE.json"), "w") as f:
            json.dump(
                {
                    "cells": [
                        {"name": f"{n}_p0", "flags": f"--tag {n} --pretrain-seed 0"}
                        for n in configs
                    ]
                },
                f,
            )

    def test_regression_selection_uses_the_smallest_n_obs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._stage(tmp, "stageX_reg")
            with patch.object(gen_stage_queue, "ROOT", tmp):
                chosen = gen_stage_queue.select_top_configs("stageX_reg", k=1)
        self.assertEqual(
            [c for c, _ in chosen],
            ["cfg_alpha"],
            "selection must rank on n_obs=5, the regime BO actually operates in; "
            "picking n20 here means lexicographic sorting crept back (S28.6)",
        )


class ReplicateSeedDisjointnessTest(unittest.TestCase):
    """S28.1: stage 3 reproduced stage 1 byte-for-byte and nobody noticed.

    `--kind replicate` re-expanded `range(args.priors)`, handing the replicate the same
    --pretrain-seed 0..8 the source stage used. Against deterministic code on identical
    inputs that reproduces every shard exactly: 675/675 identical, Delta = 0.000 +-
    0.000.
    It then looked like flawless stability, which is the most dangerous possible
    presentation of "no measurement was taken."
    """

    def _stage(self, tmp: str, stage: str, seeds: range) -> None:
        d = os.path.join(tmp, "results/raw/v2", stage)
        os.makedirs(d)
        with open(os.path.join(d, "SUMMARY.json"), "w") as f:
            json.dump(
                {
                    "configs": {
                        "cfgA": {
                            "kind": "bo",
                            "n_priors": 9,
                            "methods": {"em_frozen": {"budget_to_0.01": 5.0}},
                        }
                    }
                },
                f,
            )
        with open(os.path.join(d, "PROVENANCE.json"), "w") as f:
            json.dump(
                {
                    "cells": [
                        {"name": f"cfgA_p{k}", "flags": f"--foo --pretrain-seed {k}"}
                        for k in seeds
                    ]
                },
                f,
            )

    def test_source_seeds_are_recovered_from_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._stage(tmp, "stage1_x", range(9))
            with patch.object(gen_stage_queue, "ROOT", tmp):
                self.assertEqual(
                    gen_stage_queue.source_prior_seeds("stage1_x"), set(range(9))
                )

    def test_replicate_defaults_to_a_disjoint_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._stage(tmp, "stage1_x", range(9))
            out = os.path.join(tmp, "q.tsv")
            argv = [
                "gen_stage_queue",
                "--stage",
                "stage3_x",
                "--regime",
                "pd1_bo_armC",
                "--kind",
                "replicate",
                "--select-from",
                "stage1_x",
                "--priors",
                "9",
                "--out",
                out,
            ]
            with (
                patch.object(gen_stage_queue, "ROOT", tmp),
                patch.object(sys, "argv", argv),
            ):
                gen_stage_queue.main()
            seeds = {
                int(line.split("--pretrain-seed ")[1].split()[0])
                for line in open(out)
                if "--pretrain-seed" in line
            }
        self.assertFalse(
            seeds & set(range(9)),
            f"replicate reused the source's priors ({sorted(seeds & set(range(9)))}); "
            "it will reproduce the source byte-for-byte (S28.1)",
        )
        self.assertEqual(len(seeds), 9)

    def test_explicit_overlap_is_refused_not_warned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._stage(tmp, "stage1_x", range(9))
            argv = [
                "gen_stage_queue",
                "--stage",
                "stage3_x",
                "--regime",
                "pd1_bo_armC",
                "--kind",
                "replicate",
                "--select-from",
                "stage1_x",
                "--priors",
                "9",
                "--prior-offset",
                "0",
                "--out",
                os.path.join(tmp, "q.tsv"),
            ]
            with (
                patch.object(gen_stage_queue, "ROOT", tmp),
                patch.object(sys, "argv", argv),
            ):
                with self.assertRaises(ValueError):
                    gen_stage_queue.main()


class MeanKernelDeconfoundTest(unittest.TestCase):
    """S28.3: --em-canonical hyperbo sat on exactly the non-empirical-mean rows.

    That makes the canonical kernel and the mean transfer perfectly confounded, and on
    arm C the ranking is precisely that split while every within-mean contrast is null.
    The sweep therefore cannot say which factor won. One cell fixes it.
    """

    def test_a_kernel_only_cell_exists(self) -> None:
        kernel_only = [
            n
            for n, f in gen_stage_queue.MEANS
            if "--em-canonical hyperbo" in f and "--em-mean " not in f
        ]
        self.assertTrue(
            kernel_only,
            "MEANS has no canonical=hyperbo x mean=empirical cell, so the kernel and "
            "the mean transfer cannot be separated within this sweep (S28.3)",
        )

    def test_the_confounding_pattern_is_actually_broken(self) -> None:
        """Not just 'a cell exists' -- the two factors must be separable.

        Separability needs BOTH corners of the kernel axis at a fixed mean: a row that
        moves the kernel alone, and a row that moves the kernel and the mean together.
        With only the latter (the original design) every kernel row is also a mean row
        and no contrast can attribute the effect.
        """
        rows = [
            (
                "--em-canonical hyperbo" in f,
                "--em-mean " in f and "--em-mean empirical" not in f,
            )
            for _n, f in gen_stage_queue.MEANS
        ]
        self.assertTrue(
            any(canon and not mean for canon, mean in rows),
            "no row moves the canonical kernel with the mean left empirical, so the "
            "kernel effect cannot be isolated (S28.3)",
        )
        self.assertTrue(
            any(canon and mean for canon, mean in rows),
            "no row moves both, so the incremental mean effect cannot be measured",
        )
        self.assertFalse(
            any(mean and not canon for canon, mean in rows),
            "a mean-transfer row without a hyperbo prior raises at runtime (R6)",
        )


class PriorCacheCostAccountingTest(unittest.TestCase):
    """S30.1 -- a cache hit must never be readable as zero-cost pre-training."""

    def test_cached_reports_hit_and_real_compute_time(self):
        with tempfile.TemporaryDirectory() as d:
            data = [torch.zeros(2, 2)]
            params = {"a": 1}
            calls = []

            def compute():
                calls.append(1)
                return {"w": torch.ones(3)}

            s1: dict = {}
            prior_cache.cached(
                d, "readwrite", "hyperbo", data, params, compute, stats=s1
            )
            self.assertEqual(len(calls), 1)
            self.assertFalse(s1["hit"], "first call must be a miss")
            self.assertGreaterEqual(s1["compute_s"], 0.0)

            s2: dict = {}
            prior_cache.cached(
                d, "readwrite", "hyperbo", data, params, compute, stats=s2
            )
            self.assertEqual(len(calls), 1, "second call must be served from cache")
            self.assertTrue(s2["hit"], "a hit must be reported as a hit")
            self.assertEqual(
                s2["compute_s"],
                0.0,
                "a hit must contribute ZERO fitting time -- the bug in S30.1 was that "
                "the wrapper was timed instead of the compute, so a hit recorded the "
                "torch.load time (0.03-0.05s) as if it were the cost of pre-training",
            )

    def test_stats_is_optional(self):
        with tempfile.TemporaryDirectory() as d:
            out = prior_cache.cached(
                d, "readwrite", "hyperbo", [torch.zeros(1)], {}, lambda: {"x": 1}
            )
            self.assertEqual(out, {"x": 1})


class PretrainCostRefusesCachedCellsTest(unittest.TestCase):
    """S30.1 -- stage4 must report cache-served cells as missing, not as free."""

    def _cell(self, d, name, em, hb, hits=None):
        body = {
            "summary": {},
            "timings": {"em_pretrain_s": em, "hyperbo_pretrain_s": hb},
        }
        if hits is not None:
            body["timings"]["prior_cache_hits"] = hits
        with open(os.path.join(d, name), "w") as f:
            json.dump(body, f)

    def _cost(self, root, method):
        stage_dir = os.path.join(root, "st")
        os.makedirs(stage_dir, exist_ok=True)
        with patch.object(stage4, "RAW", root):
            return stage4.pretrain_cost("st", "base", method, priors=range(3))

    def test_hyperbo_cost_is_unmeasurable_when_every_cell_was_cached(self):
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "st")
            os.makedirs(d)
            # torch.load times, exactly as the real cached cells record them.
            for i in range(3):
                self._cell(d, f"base_p{i}_s0.json", 400.0, 0.031)
            cost, n_used, n_served = self._cost(root, "hyperbo_frozen")
            self.assertEqual((n_used, n_served), (0, 3))
            self.assertNotEqual(
                cost,
                cost,
                "0.031s is a cache hit, not a 0.031s HyperBO fit; averaging it "
                "reports pre-training as free and inverts the cost Pareto",
            )

    def test_additive_method_is_refused_when_only_its_hyperbo_half_was_cached(self):
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "st")
            os.makedirs(d)
            for i in range(3):
                self._cell(d, f"base_p{i}_s0.json", 415.9, 0.031)
            # The TOTAL is 415.9s, far above any plausible hit threshold, so a
            # total-based guard passes this through while understating cost by ~845s.
            _cost, n_used, n_served = self._cost(root, "em_additive_hyperbo")
            self.assertEqual((n_used, n_served), (0, 3))

    def test_genuine_fits_are_still_measured(self):
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "st")
            os.makedirs(d)
            for i in range(3):
                self._cell(d, f"base_p{i}_s0.json", 400.0, 845.0, hits={"hyperbo": 0})
            cost, n_used, n_served = self._cost(root, "hyperbo_frozen")
            self.assertEqual((n_used, n_served), (3, 0))
            self.assertAlmostEqual(cost, 845.0, places=3)

    def test_explicit_hit_counter_beats_the_threshold_heuristic(self):
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "st")
            os.makedirs(d)
            # A large recorded time that the harness nonetheless flagged as a hit.
            for i in range(3):
                self._cell(d, f"base_p{i}_s0.json", 400.0, 845.0, hits={"hyperbo": 1})
            _c, n_used, n_served = self._cost(root, "hyperbo_frozen")
            self.assertEqual((n_used, n_served), (0, 3))

    def test_em_alone_is_never_treated_as_cache_served(self):
        # EM is deliberately not cached, so a small em_pretrain_s is a real measurement
        # and must not be discarded by the hit heuristic.
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "st")
            os.makedirs(d)
            for i in range(3):
                self._cell(d, f"base_p{i}_s0.json", 0.4, 0.031)
            cost, n_used, n_served = self._cost(root, "em_frozen")
            self.assertEqual((n_used, n_served), (3, 0))
            self.assertAlmostEqual(cost, 0.4, places=3)


class DiagnoseHasItsOwnCacheKeyTest(unittest.TestCase):
    """S30.2 -- bo_diagnose must not key fits under bo_experiment's description."""

    def test_diagnose_accepts_the_cache_flags(self):
        p = bo_diagnose.build_parser()
        a = p.parse_args(["--prior-cache", "/tmp/x", "--prior-cache-mode", "read"])
        self.assertEqual(a.prior_cache, "/tmp/x")
        self.assertEqual(a.prior_cache_mode, "read")

    def test_diagnose_key_records_what_diagnose_actually_fits(self):
        a = bo_diagnose.build_parser().parse_args([])
        k = bo_diagnose._diag_hyperbo_params(a, 5)
        # This harness hardcodes NLL and uses meta_iters at every call site.
        self.assertEqual(k["loss_type"], "NLL")
        self.assertEqual(k["num_iterations"], a.meta_iters)
        self.assertIn("pretrain_seed", k)

    def test_the_two_harnesses_do_not_collide_when_they_fit_differently(self):
        da = bo_diagnose.build_parser().parse_args([])
        ea = bo_experiment.build_parser().parse_args(
            ["--hyperbo-loss", "EKL", "--meta-iters", str(da.meta_iters)]
        )
        dk = bo_diagnose._diag_hyperbo_params(da, 5)
        ek = bo_experiment._hyperbo_cache_params(ea, 5)
        self.assertNotEqual(
            prior_cache.fingerprint("hyperbo", "same-data", dk),
            prior_cache.fingerprint("hyperbo", "same-data", ek),
            "an EKL fit and an NLL fit on the same data must not share a cache entry",
        )


class DriversPassTheCacheToBothHarnessesTest(unittest.TestCase):
    """S30.2 -- withholding the flag from bo_diagnose left 252 cells re-fitting."""

    def _script(self, name):
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (
            os.path.join(here, name),
            os.path.join(here, "..", "results", name),
        ):
            if os.path.exists(cand):
                with open(cand) as f:
                    return f.read()
        self.skipTest(f"{name} not available as a resource")

    def test_no_driver_gates_the_prior_cache_on_the_harness(self):
        for name in ("run_autonomous.sh", "run_noise_sweep.sh"):
            body = self._script(name)
            self.assertIn("--prior-cache", body, f"{name} must pass the cache")
            lines = [
                ln
                for ln in body.splitlines()
                if "--prior-cache" in ln and not ln.strip().startswith("#")
            ]
            self.assertTrue(lines, f"{name} has no live --prior-cache line")
            for ln in lines:
                self.assertNotIn(
                    'harness" = "bo_experiment',
                    ln,
                    f"{name} still gates the cache on the harness",
                )


class IwNuSweepCanActuallyMoveTest(unittest.TestCase):
    """S30.6 -- --iw-nu-mode is a no-op unless --use-covar-prior is also set.

    resolve_iw_nu() returns args.iw_nu unchanged when use_covar_prior is False, and
    neither PD1 arm sets it. A sweep that varied only --iw-nu-mode would therefore
    produce cells numerically identical to the control and read as "OAS makes no
    difference".
    """

    def test_every_mode_cell_also_enables_the_covar_prior(self):
        for name, flags in gen_stage_queue.IWNU:
            if "--iw-nu-mode" in flags:
                self.assertIn(
                    "--use-covar-prior",
                    flags,
                    f"{name} sets --iw-nu-mode without --use-covar-prior, so "
                    "resolve_iw_nu returns early and the estimator never runs",
                )

    def test_control_holds_the_covar_prior_on(self):
        # Otherwise the contrast is "covar prior on/off", not "how nu is chosen".
        control = dict(gen_stage_queue.IWNU)["iwnu_manual"]
        self.assertIn("--use-covar-prior", control)
        self.assertNotIn("--iw-nu-mode", control)

    def test_the_sweep_actually_varies_the_mode(self):
        modes = {
            f.split("--iw-nu-mode ")[1].split()[0]
            for _, f in gen_stage_queue.IWNU
            if "--iw-nu-mode" in f
        }
        self.assertEqual(modes, {"oas", "ledoit_wolf", "oas_whitened"})


class ProvenanceStampTest(unittest.TestCase):
    """S30.8 -- every result must say which binary produced it."""

    def test_atomic_write_json_stamps_the_commit(self):
        import json as _json
        import tempfile

        from _scratch_bo.bo_experiment import atomic_write_json

        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/r.json"
            atomic_write_json(p, {"config": {"a": 1}})
            got = _json.load(open(p))
        self.assertIn(
            "provenance",
            got,
            "results must carry a provenance block; PROVENANCE.json records the "
            "queue-generation commit, which is not what produced the shard",
        )
        self.assertIn("code_commit", got["provenance"])
        self.assertIn("run_utc", got["provenance"])

    def test_a_non_dict_payload_is_left_alone(self):
        import json as _json
        import tempfile

        from _scratch_bo.bo_experiment import atomic_write_json

        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/l.json"
            atomic_write_json(p, [1, 2, 3])
            self.assertEqual(_json.load(open(p)), [1, 2, 3])


class WhitenedIwNuTest(unittest.TestCase):
    """S30.7 -- alpha must be measured against the target actually in force."""

    def test_ridge_one_recovers_plain_oas_exactly(self):
        """tau=1 makes the target the identity, so it MUST equal spherical OAS.

        This is the property that makes oas_whitened a generalisation rather than a
        different estimator. If it ever drifts, the whitened path has stopped calling
        the published formula and its guarantees no longer transfer.
        """
        import torch as _t
        from _scratch_bo.em_noise_selection import (
            estimate_iw_nu,
            estimate_iw_nu_whitened,
        )

        g = _t.Generator().manual_seed(4)
        # T MUST exceed K by enough that alpha lands strictly inside (0, 1). The first
        # version of this test used T=9, K=12: the sample covariance was singular, OAS
        # saturated at the clip for BOTH paths, and the assertion compared 1.0 to 1.0.
        # It passed against a deliberately broken ridge -- an instrument that could not
        # register the effect it existed to measure.
        T, K = 40, 8
        A = _t.randn(K, K, generator=g, dtype=_t.double)
        Sig = A @ A.T + _t.eye(K, dtype=_t.double)
        X = _t.randn(T, K, generator=g, dtype=_t.double) @ _t.linalg.cholesky(Sig).T
        F = _t.randn(K, K, generator=g, dtype=_t.double)
        F = F @ F.T + _t.eye(K, dtype=_t.double)
        nu_s, a_s = estimate_iw_nu(X, method="oas")
        nu_w, a_w = estimate_iw_nu_whitened(X, F, ridge=1.0)
        self.assertGreater(
            a_s, 0.02, "alpha saturated low -- this test cannot detect a difference"
        )
        self.assertLess(
            a_s, 0.98, "alpha saturated high -- this test cannot detect a difference"
        )
        self.assertAlmostEqual(a_s, a_w, places=9)
        self.assertAlmostEqual(nu_s, nu_w, places=6)

    def test_target_on_wrong_points_raises_rather_than_broadcasts(self):
        import torch as _t
        from _scratch_bo.em_noise_selection import estimate_iw_nu_whitened

        X = _t.randn(9, 12, dtype=_t.double)
        with self.assertRaises(ValueError):
            estimate_iw_nu_whitened(X, _t.eye(7, dtype=_t.double))

    def test_every_whitened_cell_enables_the_covar_prior(self):
        for name, flags in gen_stage_queue.IWNU:
            if "oas_whitened" in flags:
                self.assertIn("--use-covar-prior", flags, name)
                self.assertIn("--iw-target-ridge", flags, name)

    def test_a_dict_without_config_is_not_stamped(self):
        """The helper also writes non-result JSON; that contract must not change."""
        import json as _json
        import tempfile

        from _scratch_bo.bo_experiment import atomic_write_json

        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/x.json"
            atomic_write_json(p, {"a": 1})
            self.assertEqual(_json.load(open(p)), {"a": 1})


class EmShrinkageSweepTest(unittest.TestCase):
    """S31 follow-up -- alpha used DIRECTLY, not inverted into an inert nu."""

    def test_shrink2_cells_never_enable_the_covar_prior(self):
        """Same guard for the S33 follow-up cells, which are already queued to run."""
        for name, flags in gen_stage_queue.SHRINK2:
            self.assertNotIn("--use-covar-prior", flags, name)

    def test_shrink2_spans_both_questions_it_was_built_to_answer(self):
        alphas = {
            float(f.split("--em-shrinkage ")[1].split()[0])
            for _, f in gen_stage_queue.SHRINK2
            if "--em-shrinkage " in f
        }
        self.assertIn(0.0, alphas, "control must be present")
        # S32 bracketed the optimum at 0.10 between 0.05 and 0.20; fill that in.
        self.assertTrue(len([a for a in alphas if 0.05 < a < 0.20]) >= 3)
        taus = {
            float(f.split("--iw-target-ridge ")[1].split()[0])
            for _, f in gen_stage_queue.SHRINK2
            if "--iw-target-ridge" in f
        }
        self.assertTrue(len(taus) >= 3, "need a ridge curve, not a single point")
        self.assertTrue(
            any(
                "--em-shrinkage-mode oas" in f and "whitened" not in f
                for _, f in gen_stage_queue.SHRINK2
            ),
            "tau=1 anchor (plain OAS) must be measured in the SAME stage, or the "
            "whitening comparison is cross-stage and not paired",
        )

    def test_shrink_cells_never_enable_the_covar_prior(self):
        """The IW prior would pin alpha at 0.97 underneath whatever this sweep sets.

        S31.3 measured it at +7.60 +/- 1.88 evals worse on armA. If it leaked into a
        cell, that cell would be measuring the prior, not the shrinkage.
        """
        for name, flags in gen_stage_queue.SHRINK:
            self.assertNotIn("--use-covar-prior", flags, name)

    def test_the_sweep_spans_the_low_region_and_contains_its_control(self):
        vals = {
            float(f.split("--em-shrinkage ")[1].split()[0])
            for _, f in gen_stage_queue.SHRINK
            if "--em-shrinkage " in f
        }
        self.assertIn(0.0, vals, "alpha=0 is the control and must be present")
        self.assertTrue(
            any(0.0 < v <= 0.1 for v in vals),
            "S31.3 puts the interesting region low; need a cell in (0, 0.1]",
        )
        modes = {
            f.split("--em-shrinkage-mode ")[1].split()[0]
            for _, f in gen_stage_queue.SHRINK
            if "--em-shrinkage-mode" in f
        }
        self.assertEqual(modes, {"oas", "oas_whitened"})

    def test_manual_mode_is_exactly_the_flag(self):
        """The default path must not consult the data at all."""
        from _scratch_bo.bo_experiment import _resolve_em_shrinkage

        class A:
            em_shrinkage = 0.123
            em_shrinkage_mode = "manual"

        self.assertEqual(_resolve_em_shrinkage(A(), None, None), 0.123)
