#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models import SingleTaskGP, SingleTaskVariationalGP
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch_community.acquisition.local_entropy_search import LocalEntropySearch
from torch.optim import Optimizer


class NoOpOptimizer(Optimizer):
    def __init__(self, params, lr: float = 1.0) -> None:
        super().__init__(params, {"lr": lr})

    def step(self, closure=None):
        if closure is not None:
            closure()
        return None


class TestLocalEntropySearch(BotorchTestCase):
    def _get_model(
        self,
        *,
        n: int = 8,
        d: int = 2,
        dtype: torch.dtype = torch.double,
    ) -> tuple[torch.Tensor, SingleTaskGP]:
        train_X = torch.rand(n, d, device=self.device, dtype=dtype)
        train_Y = torch.sin(train_X[:, :1] * 2.7)
        if d > 1:
            train_Y = train_Y + 0.1 * train_X[:, 1:2]
        return train_X, SingleTaskGP(train_X, train_Y)

    def test_forward_is_finite_for_maximize_and_minimize(self) -> None:
        train_X, model = self._get_model()
        X = torch.rand(5, 1, 2, device=self.device, dtype=torch.double)

        for maximize in (True, False):
            with self.subTest(maximize=maximize):
                acqf = LocalEntropySearch(
                    model=model,
                    x_incumbent=train_X[0],
                    num_path_samples=3,
                    num_descent_steps=4,
                    learning_rate=0.05,
                    maximize=maximize,
                )
                values = acqf(X)
                self.assertEqual(values.shape, torch.Size([5]))
                self.assertTrue(torch.isfinite(values).all())

    def test_q_gt_1_raises(self) -> None:
        train_X, model = self._get_model(d=3)
        acqf = LocalEntropySearch(
            model=model,
            x_incumbent=train_X[0],
            num_path_samples=2,
            num_descent_steps=2,
            learning_rate=0.05,
        )

        with self.assertRaises(AssertionError):
            acqf(torch.rand(4, 2, 3, device=self.device, dtype=torch.double))

    def test_invalid_arguments_raise(self) -> None:
        train_X, model = self._get_model()
        cases = [
            {"x_incumbent": train_X[:2]},
            {"x_incumbent": torch.rand(3, device=self.device, dtype=torch.double)},
            {"num_path_samples": 0},
            {"num_descent_steps": 0},
            {"learning_rate": 0.0},
            {"min_variance": 0.0},
            {"sequence_subsample_size": 0},
            {"sequence_discretization_size": 0},
            {"conditional_model_chunk_size": 0},
            {"convergence_tol": -1e-6},
            {"virtual_observation_noise": 0.0},
            {"bounds": torch.rand(3, 2, device=self.device, dtype=torch.double)},
        ]

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    init_kwargs = {
                        "model": model,
                        "x_incumbent": train_X[0],
                        "num_path_samples": 2,
                        "num_descent_steps": 2,
                        "learning_rate": 0.05,
                    }
                    init_kwargs.update(kwargs)
                    LocalEntropySearch(
                        **init_kwargs,
                    )

    def test_multi_output_exact_model_raises(self) -> None:
        train_X = torch.rand(8, 2, device=self.device, dtype=torch.double)
        train_Y = torch.cat(
            [torch.sin(train_X[:, :1]), torch.cos(train_X[:, 1:2])],
            dim=-1,
        )
        model = SingleTaskGP(train_X, train_Y)

        with self.assertRaisesRegex(ValueError, "single-output models"):
            LocalEntropySearch(
                model=model,
                x_incumbent=train_X[0],
                num_path_samples=2,
                num_descent_steps=2,
                learning_rate=0.05,
            )

    def test_approximate_gp_model_raises(self) -> None:
        train_X = torch.rand(8, 2, device=self.device, dtype=torch.double)
        train_Y = torch.sin(train_X[:, :1] * 2.7)
        model = SingleTaskVariationalGP(train_X, train_Y)

        with self.assertRaisesRegex(ValueError, "exact single-output BoTorch GP"):
            LocalEntropySearch(
                model=model,
                x_incumbent=train_X[0],
                num_path_samples=2,
                num_descent_steps=2,
                learning_rate=0.05,
            )

    def test_batched_exact_model_raises(self) -> None:
        train_X = torch.rand(2, 8, 2, device=self.device, dtype=torch.double)
        train_Y = torch.sin(train_X[..., :1] * 2.7)
        model = SingleTaskGP(train_X, train_Y)

        with self.assertRaisesRegex(ValueError, "batched GP models"):
            LocalEntropySearch(
                model=model,
                x_incumbent=train_X[0, 0],
                num_path_samples=2,
                num_descent_steps=2,
                learning_rate=0.05,
            )

    def test_incumbent_dimension_mismatch_raises(self) -> None:
        train_X, model = self._get_model(d=2)
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            LocalEntropySearch(
                model=model,
                x_incumbent=torch.rand(3, device=self.device, dtype=torch.double),
                num_path_samples=2,
                num_descent_steps=2,
                learning_rate=0.05,
            )

    def test_out_of_bounds_incumbent_raises(self) -> None:
        train_X, model = self._get_model(d=2)
        bounds = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]],
            device=self.device,
            dtype=torch.double,
        )

        with self.assertRaisesRegex(ValueError, "within bounds"):
            LocalEntropySearch(
                model=model,
                x_incumbent=torch.tensor(
                    [1.1, 0.5], device=self.device, dtype=torch.double
                ),
                num_path_samples=2,
                num_descent_steps=2,
                learning_rate=0.05,
                bounds=bounds,
            )

    def test_chunking_tracks_path_counts_and_sequence_shapes(self) -> None:
        dtype = torch.double
        torch.manual_seed(1234)
        train_X, model = self._get_model(d=3, dtype=dtype)
        acqf = LocalEntropySearch(
            model=model,
            x_incumbent=train_X[0],
            num_path_samples=5,
            num_descent_steps=4,
            learning_rate=0.03,
            sequence_subsample_size=3,
            conditional_model_chunk_size=2,
        )

        self.assertEqual(acqf.conditional_model_num_paths, [2, 2, 1])
        self.assertEqual(len(acqf.conditional_models), 3)
        self.assertEqual(len(acqf.sequence_X_per_path), 5)
        self.assertEqual(len(acqf.sequence_Y_per_path), 5)
        self.assertEqual(sum(acqf.conditional_model_num_paths), 5)
        for X_sequence, Y_sequence in zip(
            acqf.sequence_X_per_path, acqf.sequence_Y_per_path, strict=True
        ):
            self.assertEqual(X_sequence.shape[0], Y_sequence.shape[0])
            self.assertEqual(X_sequence.shape[-1], train_X.shape[-1])
            self.assertEqual(Y_sequence.shape[-1], 1)

    def test_chunked_forward_matches_unchunked(self) -> None:
        dtype = torch.double
        torch.manual_seed(12345)
        train_X, model = self._get_model(n=9, d=2, dtype=dtype)
        X = torch.rand(7, 1, 2, device=self.device, dtype=dtype)

        torch.manual_seed(999)
        acqf_full = LocalEntropySearch(
            model=model,
            x_incumbent=train_X[0],
            num_path_samples=5,
            num_descent_steps=5,
            learning_rate=0.03,
            sequence_subsample_size=3,
            conditional_model_chunk_size=None,
        )
        torch.manual_seed(999)
        acqf_chunked = LocalEntropySearch(
            model=model,
            x_incumbent=train_X[0],
            num_path_samples=5,
            num_descent_steps=5,
            learning_rate=0.03,
            sequence_subsample_size=3,
            conditional_model_chunk_size=2,
        )

        self.assertAllClose(acqf_full(X), acqf_chunked(X), atol=1e-7, rtol=1e-5)

    def test_sequence_discretization_includes_terminal_point(self) -> None:
        train_X, model = self._get_model(d=1)
        acqf = LocalEntropySearch(
            model=model,
            x_incumbent=train_X[0],
            num_path_samples=1,
            num_descent_steps=2,
            learning_rate=0.1,
            sequence_subsample_size=None,
            sequence_discretization_size=None,
        )
        X_paths = torch.tensor(
            [[[0.0], [1.0], [2.0], [3.0], [4.0]]],
            device=self.device,
            dtype=torch.double,
        )

        expected = {
            1: [[4]],
            4: [[1, 2, 3, 4]],
        }
        for discretization_size, indices in expected.items():
            with self.subTest(discretization_size=discretization_size):
                acqf.sequence_discretization_size = discretization_size
                support_indices = acqf._get_sequence_support_indices_batch(X_paths)
                self.assertEqual(support_indices.tolist(), indices)

    def test_sequence_subsampling_preserves_path_order(self) -> None:
        train_X, model = self._get_model(d=1)
        acqf = LocalEntropySearch(
            model=model,
            x_incumbent=train_X[0],
            num_path_samples=1,
            num_descent_steps=2,
            learning_rate=0.1,
            sequence_subsample_size=3,
            sequence_discretization_size=5,
        )
        X_paths = torch.tensor(
            [[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]],
            device=self.device,
            dtype=torch.double,
        )

        torch.manual_seed(0)
        support_indices = acqf._get_sequence_support_indices_batch(X_paths)

        self.assertEqual(support_indices.shape, torch.Size([1, 3]))
        self.assertTrue(
            torch.equal(support_indices, torch.sort(support_indices, dim=1).values)
        )
        self.assertEqual(torch.unique(support_indices).numel(), 3)
        self.assertTrue(
            torch.isin(
                support_indices,
                torch.tensor(
                    [1, 2, 3, 4, 5],
                    device=self.device,
                    dtype=torch.long,
                ),
            ).all()
        )

    def test_bounds_are_respected_in_sequence(self) -> None:
        train_X, model = self._get_model()
        bounds = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]],
            device=self.device,
            dtype=torch.double,
        )
        acqf = LocalEntropySearch(
            model=model,
            x_incumbent=train_X[0],
            num_path_samples=2,
            num_descent_steps=5,
            learning_rate=1.0,
            bounds=bounds,
        )

        for X_sequence in acqf.sequence_X_per_path:
            self.assertTrue((X_sequence >= 0.0).all())
            self.assertTrue((X_sequence <= 1.0).all())

    def test_predictive_entropy_clamps_variance(self) -> None:
        train_X, model = self._get_model()
        acqf = LocalEntropySearch(
            model=model,
            x_incumbent=train_X[0],
            num_path_samples=2,
            num_descent_steps=2,
            learning_rate=0.05,
            min_variance=1e-4,
        )
        X = torch.rand(3, 1, 2, device=self.device, dtype=torch.double)
        posterior = MockPosterior(
            mean=torch.zeros(3, 1, 1, device=self.device, dtype=torch.double),
            variance=torch.zeros(3, 1, 1, device=self.device, dtype=torch.double),
        )
        entropy = acqf._predictive_entropy(model=MockModel(posterior), X=X)
        expected = torch.full(
            (3,),
            0.5 * torch.log(torch.tensor(2.0 * torch.pi * torch.e * 1e-4)).item(),
            device=self.device,
            dtype=torch.double,
        )
        self.assertAllClose(entropy, expected)

    def test_early_stopping_breaks_when_paths_stop_moving(self) -> None:
        train_X, model = self._get_model()
        acqf = LocalEntropySearch(
            model=model,
            x_incumbent=train_X[0],
            num_path_samples=3,
            num_descent_steps=10,
            learning_rate=0.05,
            optimizer_cls=NoOpOptimizer,
            convergence_tol=1e-12,
        )
        self.assertTrue(all(path.shape[0] == 2 for path in acqf.sequence_X_per_path))
