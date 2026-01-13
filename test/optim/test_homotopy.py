# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest.mock as mock

import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models import GenericDeterministicModel
from botorch.optim.homotopy import (
    FixedHomotopySchedule,
    Homotopy,
    HomotopyParameter,
    LinearHomotopySchedule,
    LogLinearHomotopySchedule,
)
from botorch.optim.optimize_homotopy import optimize_acqf_homotopy, prune_candidates
from botorch.utils.testing import BotorchTestCase
from torch.nn import Parameter


PRUNE_CANDIDATES_PATH = f"{prune_candidates.__module__}"


class TestHomotopy(BotorchTestCase):
    def _test_schedule(self, schedule, values):
        self.assertEqual(schedule.num_steps, len(values))
        self.assertEqual(schedule.value, values[0])
        self.assertFalse(schedule.should_stop)
        for i in range(len(values) - 1):
            schedule.step()
            self.assertEqual(schedule.value, values[i + 1])
            self.assertFalse(schedule.should_stop)
        schedule.step()
        self.assertTrue(schedule.should_stop)
        schedule.restart()
        self.assertEqual(schedule.value, values[0])
        self.assertFalse(schedule.should_stop)

    def test_fixed_schedule(self):
        values = [1, 3, 7]
        fixed = FixedHomotopySchedule(values=values)
        self.assertEqual(fixed._values, values)
        self._test_schedule(schedule=fixed, values=values)

    def test_linear_schedule(self):
        values = [1, 2, 3, 4, 5]
        linear = LinearHomotopySchedule(start=1, end=5, num_steps=5)
        self.assertEqual(linear._values, values)
        self._test_schedule(schedule=linear, values=values)

    def test_log_linear_schedule(self):
        values = [0.01, 0.1, 1, 10, 100]
        linear = LogLinearHomotopySchedule(start=0.01, end=100, num_steps=5)
        self.assertEqual(linear._values, values)
        self._test_schedule(schedule=linear, values=values)

    def test_homotopy(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        p1 = Parameter(-2 * torch.ones(1, **tkwargs))
        v1 = [1, 2, 3, 4, 5]
        p2 = -3 * torch.ones(1, **tkwargs)
        v2 = [0.01, 0.1, 1, 10, 100]
        callback = mock.Mock()
        homotopy_parameters = [
            HomotopyParameter(
                parameter=p1,
                schedule=LinearHomotopySchedule(start=1, end=5, num_steps=5),
            ),
            HomotopyParameter(
                parameter=p2,
                schedule=LogLinearHomotopySchedule(start=0.01, end=100, num_steps=5),
            ),
        ]
        homotopy = Homotopy(
            homotopy_parameters=homotopy_parameters, callbacks=[callback]
        )
        self.assertEqual(homotopy._original_values, [-2, -3])
        self.assertEqual(homotopy._homotopy_parameters, homotopy_parameters)
        self.assertEqual(homotopy._callbacks, [callback])
        self.assertEqual(
            [h.parameter.item() for h in homotopy._homotopy_parameters], [v1[0], v2[0]]
        )
        for i in range(4):
            homotopy.step()
            self.assertEqual(
                [h.parameter.item() for h in homotopy._homotopy_parameters],
                [v1[i + 1], v2[i + 1]],
            )
            self.assertFalse(homotopy.should_stop)
        homotopy.step()
        self.assertTrue(homotopy.should_stop)
        # Restart the schedules
        homotopy.restart()
        self.assertEqual(
            [h.parameter.item() for h in homotopy._homotopy_parameters], [v1[0], v2[0]]
        )
        # Reset the parameters to their original values
        homotopy.reset()
        self.assertEqual(
            [h.parameter.item() for h in homotopy._homotopy_parameters], [-2, -3]
        )
        # Expect the call count to be 8: init (1), step (5), restart (1), reset (1).
        self.assertEqual(callback.call_count, 8)

    def test_optimize_acqf_homotopy(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        p = Parameter(-2 * torch.ones(1, **tkwargs))
        hp = HomotopyParameter(
            parameter=p,
            schedule=LinearHomotopySchedule(start=4, end=0, num_steps=5),
        )
        model = GenericDeterministicModel(f=lambda x: 5 - (x - p) ** 2)
        acqf = PosteriorMean(model=model)
        candidate, acqf_val = optimize_acqf_homotopy(
            q=1,
            acq_function=acqf,
            bounds=torch.tensor([[-10], [5]], **tkwargs),
            homotopy=Homotopy(homotopy_parameters=[hp]),
            num_restarts=2,
            raw_samples=16,
            post_processing_func=lambda x: x.round(),
        )
        self.assertEqual(candidate, torch.zeros(1, **tkwargs))
        self.assertEqual(acqf_val, 5 * torch.ones(1, **tkwargs))

        # test fixed feature
        fixed_features = {0: 1.0}
        model = GenericDeterministicModel(
            f=lambda x: 5 - (x - p).sum(dim=-1, keepdims=True) ** 2
        )
        acqf = PosteriorMean(model=model)
        # test raise warning on using ``fixed_features`` argument
        message = (
            "The `fixed_features` argument is deprecated, "
            "use `fixed_features_list` instead."
        )
        with self.assertWarnsRegex(DeprecationWarning, message):
            optimize_acqf_homotopy(
                q=1,
                acq_function=acqf,
                bounds=torch.tensor([[-10, -10], [5, 5]]).to(**tkwargs),
                homotopy=Homotopy(homotopy_parameters=[hp]),
                num_restarts=2,
                raw_samples=16,
                fixed_features=fixed_features,
            )

        candidate, acqf_val = optimize_acqf_homotopy(
            q=1,
            acq_function=acqf,
            bounds=torch.tensor([[-10, -10], [5, 5]], **tkwargs),
            homotopy=Homotopy(homotopy_parameters=[hp]),
            num_restarts=2,
            raw_samples=16,
            fixed_features_list=[fixed_features],
        )
        self.assertEqual(candidate[0, 0], torch.tensor(1, **tkwargs))

        # test fixed feature list
        fixed_features_list = [{0: 1.0}, {1: 3.0}]
        model = GenericDeterministicModel(
            f=lambda x: 5 - (x - p).sum(dim=-1, keepdims=True) ** 2
        )
        acqf = PosteriorMean(model=model)
        # test raise error when fixed_features and fixed_features_list are both provided
        with self.assertRaisesRegex(
            ValueError,
            "Either `fixed_feature` or `fixed_features_list` can be provided",
        ):
            optimize_acqf_homotopy(
                q=1,
                acq_function=acqf,
                bounds=torch.tensor([[-10, -10, -10], [5, 5, 5]], **tkwargs),
                homotopy=Homotopy(homotopy_parameters=[hp]),
                num_restarts=2,
                raw_samples=16,
                fixed_features_list=fixed_features_list,
                fixed_features=fixed_features,
            )
        candidate, acqf_val = optimize_acqf_homotopy(
            q=1,
            acq_function=acqf,
            bounds=torch.tensor([[-10, -10, -10], [5, 5, 5]], **tkwargs),
            homotopy=Homotopy(homotopy_parameters=[hp]),
            num_restarts=2,
            raw_samples=16,
            fixed_features_list=fixed_features_list,
        )
        self.assertEqual(candidate[0, 0], torch.tensor(1, **tkwargs))

        # With q > 1.
        acqf = qExpectedImprovement(model=model, best_f=0.0)
        candidate, acqf_val = optimize_acqf_homotopy(
            q=3,
            acq_function=acqf,
            bounds=torch.tensor([[-10, -10], [5, 5]], **tkwargs),
            homotopy=Homotopy(homotopy_parameters=[hp]),
            num_restarts=2,
            raw_samples=16,
            fixed_features_list=[fixed_features],
        )
        self.assertEqual(candidate.shape, torch.Size([3, 2]))
        self.assertEqual(acqf_val.shape, torch.Size([3]))

        # with linear constraints
        constraints = [
            (  # X[..., 0] + X[..., 1] >= 2.
                torch.tensor([0, 1], device=self.device),
                torch.ones(2, device=self.device, dtype=torch.double),
                2.0,
            )
        ]

        acqf = PosteriorMean(model=model)
        candidate, acqf_val = optimize_acqf_homotopy(
            q=1,
            acq_function=acqf,
            bounds=torch.tensor([[-10, -10], [5, 5]], **tkwargs),
            homotopy=Homotopy(homotopy_parameters=[hp]),
            num_restarts=2,
            raw_samples=16,
            inequality_constraints=constraints,
        )
        self.assertEqual(candidate.shape, torch.Size([1, 2]))
        self.assertGreaterEqual(candidate.sum().item(), 2.0 - 1e-6)

    def test_prune_candidates(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        # no pruning
        X = torch.rand(6, 3, **tkwargs)
        vals = X.sum(dim=-1)
        X_pruned = prune_candidates(candidates=X, acq_values=vals, prune_tolerance=1e-6)
        self.assertTrue((X[vals.argsort(descending=True), :] == X_pruned).all())
        # pruning
        X[1, :] = X[0, :] + 1e-10
        X[4, :] = X[2, :] - 1e-10
        vals = torch.tensor([1, 6, 3, 4, 2, 5], **tkwargs)
        X_pruned = prune_candidates(candidates=X, acq_values=vals, prune_tolerance=1e-6)
        self.assertTrue((X[[1, 5, 3, 2]] == X_pruned).all())
        # invalid shapes
        with self.assertRaisesRegex(
            ValueError, "`candidates` must be of size `n x d`."
        ):
            prune_candidates(
                candidates=torch.zeros(3, 2, 1),
                acq_values=torch.zeros(2, 1),
                prune_tolerance=1e-6,
            )
        with self.assertRaisesRegex(ValueError, "`acq_values` must be of size `n`."):
            prune_candidates(
                candidates=torch.zeros(3, 2),
                acq_values=torch.zeros(3, 1),
                prune_tolerance=1e-6,
            )
        with self.assertRaisesRegex(ValueError, "`prune_tolerance` must be >= 0."):
            prune_candidates(
                candidates=torch.zeros(3, 2),
                acq_values=torch.zeros(3),
                prune_tolerance=-1.2345,
            )

    @mock.patch(f"{PRUNE_CANDIDATES_PATH}.prune_candidates", wraps=prune_candidates)
    def test_optimize_acqf_homotopy_pruning(self, prune_candidates_mock):
        tkwargs = {"device": self.device, "dtype": torch.double}
        p = Parameter(torch.zeros(1, **tkwargs))
        hp = HomotopyParameter(
            parameter=p,
            schedule=LinearHomotopySchedule(start=4, end=0, num_steps=5),
        )
        model = GenericDeterministicModel(f=lambda x: 5 - (x - p) ** 2)
        acqf = PosteriorMean(model=model)
        candidate, acqf_val = optimize_acqf_homotopy(
            q=1,
            acq_function=acqf,
            bounds=torch.tensor([[-10], [5]]).to(**tkwargs),
            homotopy=Homotopy(homotopy_parameters=[hp]),
            num_restarts=4,
            raw_samples=16,
            post_processing_func=lambda x: x.round(),
            return_full_tree=True,
        )
        # First time we expect to call ``prune_candidates`` with 4 candidates
        self.assertEqual(
            prune_candidates_mock.call_args_list[0][1]["candidates"].shape,
            torch.Size([4, 1]),
        )
        for i in range(1, 5):  # The paths should have been pruned to just one path
            self.assertEqual(
                prune_candidates_mock.call_args_list[i][1]["candidates"].shape,
                torch.Size([1, 1]),
            )
