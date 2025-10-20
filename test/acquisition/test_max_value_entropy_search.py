#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from unittest import mock

import torch
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.max_value_entropy_search import (
    _sample_max_value_Gumbel,
    _sample_max_value_Thompson,
    qLowerBoundMaxValueEntropy,
    qMaxValueEntropy,
    qMultiFidelityLowerBoundMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase


class TestMaxValueEntropySearch(BotorchTestCase):
    def test_q_max_value_entropy(self):
        for dtype in (torch.float, torch.double):
            torch.manual_seed(7)
            model = SingleTaskGP(
                train_X=torch.rand(10, 2, device=self.device, dtype=dtype),
                train_Y=torch.rand(10, 1, device=self.device, dtype=dtype),
            )
            candidate_set = torch.rand(100, 2, device=self.device, dtype=dtype)

            # test error in case of batch GP model
            with self.assertRaises(NotImplementedError):
                qMaxValueEntropy(
                    model=SingleTaskGP(
                        train_X=torch.rand(5, 10, 2, device=self.device, dtype=dtype),
                        train_Y=torch.rand(5, 10, 1, device=self.device, dtype=dtype),
                    ),
                    candidate_set=candidate_set,
                )

            # test that init works if batch_shape is not implemented on the model
            with mock.patch.object(
                SingleTaskGP, "batch_shape", side_effect=NotImplementedError
            ):
                qMaxValueEntropy(model=model, candidate_set=candidate_set)

            # test error when number of outputs > 1.
            with self.assertRaisesRegex(
                UnsupportedError, "Multi-output models are not supported by"
            ):
                qMaxValueEntropy(
                    model=SingleTaskGP(
                        train_X=torch.rand(10, 2, device=self.device, dtype=dtype),
                        train_Y=torch.rand(10, 2, device=self.device, dtype=dtype),
                    ),
                    candidate_set=candidate_set,
                )

            # test with X_pending is None
            qMVE = qMaxValueEntropy(
                model=model, candidate_set=candidate_set, num_mv_samples=5
            )

            # test initialization
            self.assertEqual(qMVE.num_fantasies, 16)
            self.assertEqual(qMVE.num_mv_samples, 5)
            self.assertIsInstance(qMVE.sampler, SobolQMCNormalSampler)
            self.assertEqual(qMVE.sampler.sample_shape, torch.Size([128]))
            self.assertIsInstance(qMVE.fantasies_sampler, SobolQMCNormalSampler)
            self.assertEqual(qMVE.fantasies_sampler.sample_shape, torch.Size([16]))
            self.assertEqual(qMVE.use_gumbel, True)
            self.assertEqual(qMVE.posterior_max_values.shape, torch.Size([5, 1]))

            # test evaluation
            X = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            self.assertEqual(qMVE(X).shape, torch.Size([3]))

            # test set X pending to None in case of _init_model exists
            qMVE.set_X_pending(None)
            self.assertEqual(qMVE.model, qMVE._init_model)

            # test with use_gumbel = False
            qMVE = qMaxValueEntropy(
                model=model,
                candidate_set=candidate_set,
                num_mv_samples=5,
                use_gumbel=False,
            )
            self.assertEqual(qMVE(X).shape, torch.Size([3]))

            # test with X_pending is not None
            qMVE = qMaxValueEntropy(
                model=model,
                candidate_set=candidate_set,
                num_mv_samples=5,
                X_pending=torch.rand(1, 2, device=self.device, dtype=dtype),
            )

            X = torch.rand(7, 1, 2, device=self.device, dtype=dtype)
            self.assertEqual(qMVE(X).shape, torch.Size([7]))

    def test_q_lower_bound_max_value_entropy(self):
        for dtype in (torch.float, torch.double):
            torch.manual_seed(7)
            model = SingleTaskGP(
                train_X=torch.rand(10, 2, device=self.device, dtype=dtype),
                train_Y=torch.rand(10, 1, device=self.device, dtype=dtype),
            )
            candidate_set = torch.rand(100, 2, device=self.device, dtype=dtype)

            # test with X_pending is None
            qGIBBON = qLowerBoundMaxValueEntropy(
                model=model, candidate_set=candidate_set, num_mv_samples=5
            )

            # test initialization
            self.assertEqual(qGIBBON.num_mv_samples, 5)
            self.assertEqual(qGIBBON.use_gumbel, True)
            self.assertEqual(qGIBBON.posterior_max_values.shape, torch.Size([5, 1]))

            # test evaluation
            X = torch.rand(5, 1, 2, device=self.device, dtype=dtype)
            self.assertEqual(qGIBBON(X).shape, torch.Size([5]))

            # test with use_gumbel = False
            qGIBBON = qLowerBoundMaxValueEntropy(
                model=model,
                candidate_set=candidate_set,
                num_mv_samples=5,
                use_gumbel=False,
            )
            self.assertEqual(qGIBBON(X).shape, torch.Size([5]))

            # test with X_pending is not None
            qGIBBON = qLowerBoundMaxValueEntropy(
                model=model,
                candidate_set=candidate_set,
                num_mv_samples=5,
                use_gumbel=False,
                X_pending=torch.rand(1, 2, device=self.device, dtype=dtype),
            )
            self.assertEqual(qGIBBON(X).shape, torch.Size([5]))

            # Test posterior transform with X_pending.
            pt = ScalarizedPosteriorTransform(
                weights=torch.ones(1, device=self.device, dtype=dtype)
            )
            qGIBBON = qLowerBoundMaxValueEntropy(
                model=model,
                candidate_set=candidate_set,
                num_mv_samples=10,
                use_gumbel=False,
                X_pending=torch.rand(1, 2, device=self.device, dtype=dtype),
                posterior_transform=pt,
            )
            with self.assertRaisesRegex(UnsupportedError, "X_pending is not None"):
                qGIBBON(X)

    def test_fantasy_max_values(self):
        """Test that max values are computed correctly with fantasies."""
        for dtype in (torch.float, torch.double):
            torch.manual_seed(7)
            model = SingleTaskGP(
                train_X=torch.rand(10, 2, device=self.device, dtype=dtype),
                train_Y=torch.rand(10, 1, device=self.device, dtype=dtype),
            )
            candidate_set = torch.rand(20, 2, device=self.device, dtype=dtype)
            X_pending = torch.rand(2, 2, device=self.device, dtype=dtype)

            # Create acquisition function with X_pending to trigger fantasizing
            qGIBBON = qLowerBoundMaxValueEntropy(
                model=model,
                candidate_set=candidate_set,
                num_mv_samples=5,
                X_pending=X_pending,
            )

            # Check that posterior_max_values has the right shape
            # Shape should be (num_mv_samples, 1) after taking mean across fantasies
            self.assertEqual(len(qGIBBON.posterior_max_values.shape), 2)
            self.assertEqual(qGIBBON.posterior_max_values.shape[0], 5)  # num_mv_samples
            self.assertEqual(
                qGIBBON.posterior_max_values.shape[1], 1
            )  # fantasy dimension removed

            # Test evaluation with fantasized model
            X = torch.rand(5, 1, 2, device=self.device, dtype=dtype)
            acq_value = qGIBBON(X)
            self.assertEqual(acq_value.shape, torch.Size([5]))

            # Test multi-fidelity version with fantasies
            qMF_GIBBON = qMultiFidelityLowerBoundMaxValueEntropy(
                model=model,
                candidate_set=candidate_set,
                num_mv_samples=5,
                X_pending=X_pending,
            )
            self.assertEqual(
                qMF_GIBBON.posterior_max_values.shape[0], 5
            )  # num_mv_samples

            # Test evaluation with fantasized model
            acq_value = qMF_GIBBON(X)
            self.assertEqual(acq_value.shape, torch.Size([5]))

    def test_q_multi_fidelity_max_value_entropy(
        self, acqf_class=qMultiFidelityMaxValueEntropy
    ):
        for dtype in (torch.float, torch.double):
            torch.manual_seed(7)
            model = SingleTaskGP(
                train_X=torch.rand(10, 2, device=self.device, dtype=dtype),
                train_Y=torch.rand(10, 1, device=self.device, dtype=dtype),
            )
            candidate_set = torch.rand(10, 2, device=self.device, dtype=dtype)
            qMF_MVE = acqf_class(
                model=model, candidate_set=candidate_set, num_mv_samples=5
            )

            # test initialization
            self.assertEqual(qMF_MVE.num_fantasies, 16)
            self.assertEqual(qMF_MVE.num_mv_samples, 5)
            self.assertIsInstance(qMF_MVE.sampler, SobolQMCNormalSampler)
            self.assertIsInstance(qMF_MVE.cost_sampler, SobolQMCNormalSampler)
            self.assertEqual(qMF_MVE.sampler.sample_shape, torch.Size([128]))
            self.assertIsInstance(qMF_MVE.fantasies_sampler, SobolQMCNormalSampler)
            self.assertEqual(qMF_MVE.fantasies_sampler.sample_shape, torch.Size([16]))
            self.assertIsInstance(qMF_MVE.expand, Callable)
            self.assertIsInstance(qMF_MVE.project, Callable)
            self.assertIsNone(qMF_MVE.X_pending)
            self.assertEqual(qMF_MVE.posterior_max_values.shape, torch.Size([5, 1]))
            self.assertIsInstance(
                qMF_MVE.cost_aware_utility, InverseCostWeightedUtility
            )

            # test evaluation
            X = torch.rand(3, 1, 2, device=self.device, dtype=dtype)
            self.assertEqual(qMF_MVE(X).shape, torch.Size([3]))

            # Test with multi-output model.
            with self.assertRaisesRegex(
                UnsupportedError, "Multi-output models are not supported"
            ):
                acqf_class(
                    model=ModelListGP(model, model),
                    candidate_set=candidate_set,
                    num_mv_samples=5,
                )

            # Test with expand.
            if acqf_class is qMultiFidelityMaxValueEntropy:
                qMF_MVE = acqf_class(
                    model=model,
                    candidate_set=candidate_set,
                    num_mv_samples=5,
                    expand=lambda X: X.repeat(1, 2, 1),
                )
                X = torch.rand(4, 1, 2, device=self.device, dtype=dtype)
                self.assertEqual(qMF_MVE(X).shape, torch.Size([4]))
            else:
                with self.assertRaisesRegex(UnsupportedError, "does not support trace"):
                    acqf_class(
                        model=model,
                        candidate_set=candidate_set,
                        num_mv_samples=5,
                        expand=lambda X: X.repeat(1, 2, 1),
                    )

    def test_q_multi_fidelity_lower_bound_max_value_entropy(self):
        # Same test as for MF-MES since GIBBON only changes in the way it computes the
        # information gain.
        self.test_q_multi_fidelity_max_value_entropy(
            acqf_class=qMultiFidelityLowerBoundMaxValueEntropy
        )

    def _test_max_value_sampler_base(self, sampler) -> None:
        for dtype in (torch.float, torch.double):
            torch.manual_seed(7)
            model = SingleTaskGP(
                train_X=torch.rand(10, 2, device=self.device, dtype=dtype),
                train_Y=torch.rand(10, 1, device=self.device, dtype=dtype),
            )
            candidate_set = torch.rand(3, 10, 2, device=self.device, dtype=dtype)
            samples = sampler(model=model, candidate_set=candidate_set, num_samples=5)
            self.assertEqual(samples.shape, torch.Size([5, 3]))

            # Test with multi-output model w/ transform.
            model = ModelListGP(model, model)
            pt = ScalarizedPosteriorTransform(
                weights=torch.ones(2, device=self.device, dtype=dtype)
            )
            samples = sampler(
                model=model,
                candidate_set=candidate_set,
                num_samples=5,
                posterior_transform=pt,
            )
            self.assertEqual(samples.shape, torch.Size([5, 3]))

    def test_sample_max_value_Gumbel(self):
        self._test_max_value_sampler_base(sampler=_sample_max_value_Gumbel)

    def test_sample_max_value_Thompson(self):
        self._test_max_value_sampler_base(sampler=_sample_max_value_Thompson)
