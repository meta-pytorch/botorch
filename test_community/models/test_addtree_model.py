# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Tests for :class:`AddTreeGP` and :func:`optimize_addtree_acqf`."""

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.utils.testing import BotorchTestCase
from botorch_community.acquisition.addtree import optimize_addtree_acqf
from botorch_community.models.addtree.encoding import decode, encode
from botorch_community.models.addtree.model import AddTreeGP
from botorch_community.models.addtree.space import AddTreeSpace
from gpytorch.mlls import ExactMarginalLogLikelihood


def _jenatton_small_spec() -> dict:
    leaf = lambda name: {  # noqa: E731
        "name": name,
        "continuous": [{"name": "v", "lo": 0.0, "hi": 1.0}],
    }
    return {
        "name": "root",
        "choices": [
            {
                "name": "L1",
                "options": {
                    "left": {
                        "name": "L1_left",
                        "continuous": [{"name": "v", "lo": 0.0, "hi": 1.0}],
                        "choices": [
                            {
                                "name": "L2",
                                "options": {
                                    "left": leaf("L2_ll"),
                                    "right": leaf("L2_lr"),
                                },
                            }
                        ],
                    },
                    "right": {
                        "name": "L1_right",
                        "continuous": [{"name": "v", "lo": 0.0, "hi": 1.0}],
                        "choices": [
                            {
                                "name": "L2",
                                "options": {
                                    "left": leaf("L2_rl"),
                                    "right": leaf("L2_rr"),
                                },
                            }
                        ],
                    },
                },
            }
        ],
    }


def _make_train_data(space: AddTreeSpace, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    Xs, Ys = [], []
    choices = ["left", "right"]
    for _ in range(n):
        L1 = rng.choice(choices)
        L2 = rng.choice(choices)
        p = {
            "L1": L1,
            f"L1_{L1}.v": float(rng.random()),
            "L2": L2,
            f"L2_{L1[0]}{L2[0]}.v": float(rng.random()),
        }
        Xs.append(encode(space, p))
        # Toy objective: sum of squared on-path continuous values.
        Ys.append(-sum(p[k] ** 2 for k in p if k.endswith(".v")))
    return torch.stack(Xs), torch.tensor(Ys, dtype=torch.float64).unsqueeze(-1)


class TestAddTreeGP(BotorchTestCase):
    def test_init_validates_dim(self) -> None:
        torch.set_default_dtype(torch.float64)
        space = AddTreeSpace.from_dict(_jenatton_small_spec())
        bad_X = torch.zeros(3, space.dim - 1, dtype=torch.float64)
        bad_Y = torch.zeros(3, 1, dtype=torch.float64)
        with self.assertRaises(ValueError):
            AddTreeGP(space, bad_X, bad_Y)

    def test_init_rejects_non_space(self) -> None:
        torch.set_default_dtype(torch.float64)
        bad_X = torch.zeros(3, 5, dtype=torch.float64)
        bad_Y = torch.zeros(3, 1, dtype=torch.float64)
        with self.assertRaises(TypeError):
            AddTreeGP("not a space", bad_X, bad_Y)  # type: ignore[arg-type]

    def test_fit_runs(self) -> None:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        space = AddTreeSpace.from_dict(_jenatton_small_spec())
        train_X, train_Y = _make_train_data(space, n=8, seed=0)
        model = AddTreeGP(space, train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        # Should now be able to predict.
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(train_X)
            self.assertEqual(posterior.mean.shape, train_Y.shape)

    def test_addtree_space_property(self) -> None:
        torch.set_default_dtype(torch.float64)
        space = AddTreeSpace.from_dict(_jenatton_small_spec())
        train_X, train_Y = _make_train_data(space, n=5, seed=1)
        model = AddTreeGP(space, train_X, train_Y)
        self.assertIs(model.addtree_space, space)


class TestOptimizeAddTreeAcqf(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        self.space = AddTreeSpace.from_dict(_jenatton_small_spec())
        train_X, train_Y = _make_train_data(self.space, n=5, seed=0)
        self.model = AddTreeGP(self.space, train_X, train_Y)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def test_q1_returns_valid_encoding(self) -> None:
        candidates, vals = optimize_addtree_acqf(
            self.model,
            beta=1.0,
            q=1,
            num_restarts=2,
            raw_samples=64,
        )
        self.assertEqual(candidates.shape, (1, self.space.dim))
        # Decoded user dict must round-trip.
        p = decode(self.space, candidates[0])
        self.assertIn("L1", p)
        self.assertIn("L2", p)
        # Re-encoding gives back the same vector.
        x_re = encode(self.space, p, dtype=candidates.dtype, device=candidates.device)
        self.assertTrue(torch.allclose(x_re, candidates[0], atol=1e-9))

    def test_q2_supports_batch(self) -> None:
        candidates, vals = optimize_addtree_acqf(
            self.model,
            beta=1.0,
            q=2,
            num_restarts=2,
            raw_samples=64,
        )
        self.assertEqual(candidates.shape, (2, self.space.dim))
        # Each candidate must be a valid encoding.
        for i in range(2):
            p = decode(self.space, candidates[i])
            self.assertIn("L1", p)

    def test_bo_loop_does_not_regress(self) -> None:
        """A short BO loop should never decrease the running best."""
        torch.manual_seed(1)
        np.random.seed(1)
        space = self.space
        train_X, train_Y = _make_train_data(space, n=5, seed=1)
        best_init = train_Y.max().item()

        for _ in range(5):
            model = AddTreeGP(space, train_X, train_Y)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            candidates, _ = optimize_addtree_acqf(
                model,
                beta=2.0,
                q=1,
                num_restarts=2,
                raw_samples=64,
            )
            x_new = candidates[0]
            p = decode(space, x_new)
            y = sum(p[k] ** 2 for k in p if k.endswith(".v"))
            train_X = torch.cat([train_X, x_new.unsqueeze(0)])
            train_Y = torch.cat(
                [train_Y, torch.tensor([[-y]], dtype=torch.float64)],
            )

        self.assertGreaterEqual(train_Y.max().item(), best_init - 1e-6)
