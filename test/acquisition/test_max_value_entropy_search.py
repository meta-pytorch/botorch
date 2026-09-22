#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from types import SimpleNamespace
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
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.exceptions.errors import UnsupportedError
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase
from torch import Tensor


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

    def test_q_lower_bound_max_value_entropy_no_underflow(self):
        # With max values far above the posterior mean, 1 - u rounds to 1 and the
        # information gain used to be exactly 0. It is small but positive.
        for dtype in (torch.float, torch.double):
            torch.manual_seed(7)
            model = SingleTaskGP(
                train_X=torch.rand(10, 2, device=self.device, dtype=dtype),
                train_Y=torch.rand(10, 1, device=self.device, dtype=dtype),
            )
            candidate_set = torch.rand(100, 2, device=self.device, dtype=dtype)
            X = torch.rand(1, 1, 2, device=self.device, dtype=dtype)
            with torch.no_grad():
                posterior = model.posterior(X)
            # 10 posterior standard deviations above the mean at X
            max_values = (posterior.mean + 10 * posterior.variance.sqrt()).view(1, 1)
            for acqf_class, X_pending in (
                (qLowerBoundMaxValueEntropy, None),
                (qLowerBoundMaxValueEntropy, candidate_set[:2]),
                (qMultiFidelityLowerBoundMaxValueEntropy, None),
            ):
                acqf = acqf_class(
                    model=model,
                    candidate_set=candidate_set,
                    num_mv_samples=5,
                    X_pending=X_pending,
                )
                acqf.posterior_max_values = max_values.expand(5, 1)
                X_grad = X.clone().requires_grad_(True)
                acq = acqf(X_grad)
                (grad,) = torch.autograd.grad(acq.sum(), X_grad)
                self.assertTrue(torch.isfinite(acq).all())
                self.assertTrue(torch.isfinite(grad).all())
                if X_pending is None:
                    self.assertTrue((acq > 0).all())
                    self.assertTrue((grad != 0).any())

    def test_q_lower_bound_max_value_entropy_information_gain_values(self):
        def information_gain(gamma: Tensor, covar: Tensor) -> Tensor:
            # Unit variances, so that rho^2 = covar^2 and gamma = m* - mean.
            n = gamma.shape[0]
            variance = torch.ones(n, 1, 1, device=self.device, dtype=gamma.dtype)
            stand_in = SimpleNamespace(
                model=mock.Mock(
                    posterior=mock.Mock(return_value=SimpleNamespace(variance=variance))
                ),
                posterior_max_values=torch.zeros_like(gamma[:1]).view(1, 1),
                posterior_transform=None,
                X_pending=None,
            )
            return qLowerBoundMaxValueEntropy._compute_information_gain(
                stand_in,
                X=torch.zeros_like(variance),
                mean_M=-gamma.view(n, 1),
                variance_M=variance.view(n, 1),
                covar_mM=covar.view(n, 1, 1),
            ).view(n)

        # -0.5 * log(1 - rho^2 * r * (gamma + r)) with r = phi(gamma) / Phi(gamma),
        # computed with mpmath at 60 digits of precision.
        gammas = [0.5, 2.0, 5.0, 8.0, 12.0]
        references = {
            1.0: [
                3.605928707099e-01,
                6.026417937969e-02,
                3.716814772108e-06,
                2.020908433415e-14,
                1.287830241398e-31,
            ],
            0.1: [
                2.637478576985e-02,
                5.709881580377e-03,
                3.716802338892e-07,
                2.020908433415e-15,
                1.287830241398e-32,
            ],
        }
        for dtype, rtol in ((torch.float, 1e-4), (torch.double, 1e-10)):
            tkwargs = {"device": self.device, "dtype": dtype}
            gamma = torch.tensor(gammas, **tkwargs)
            for rho_squared, reference in references.items():
                covar = torch.full_like(gamma, rho_squared**0.5)
                self.assertAllClose(
                    information_gain(gamma, covar),
                    torch.tensor(reference, **tkwargs),
                    rtol=rtol,
                    atol=0.0,
                )

            # Finite, non-zero gradients where the gain used to be exactly 0, and
            # finite gradients at zero correlation.
            gamma = torch.tensor([12.0, 12.0, 1.0], **tkwargs).requires_grad_(True)
            covar = torch.tensor([1.0, 0.0, 0.0], **tkwargs).requires_grad_(True)
            ig = information_gain(gamma, covar)
            self.assertEqual(ig[1:].tolist(), [0.0, 0.0])
            grad_gamma, grad_covar = torch.autograd.grad(ig.sum(), (gamma, covar))
            self.assertTrue(torch.isfinite(grad_gamma).all())
            self.assertTrue(torch.isfinite(grad_covar).all())
            self.assertLess(grad_gamma[0].item(), 0.0)

        # Agrees with the direct expression where that is accurate.
        tkwargs = {"device": self.device, "dtype": torch.double}
        gamma = torch.linspace(0.0, 3.0, 31, **tkwargs)
        normal = torch.distributions.Normal(0.0, 1.0)
        ratio = normal.log_prob(gamma).exp() / normal.cdf(gamma)
        for rho_squared in (1.0, 0.1):
            direct = -0.5 * torch.log(1 - rho_squared * ratio * (gamma + ratio))
            covar = torch.full_like(gamma, rho_squared**0.5)
            self.assertAllClose(information_gain(gamma, covar), direct, rtol=1e-8)

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
        is_mes = acqf_class is qMultiFidelityMaxValueEntropy
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
            if is_mes:
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

            # Test candidate_set dimension mismatch with MF model.
            torch.manual_seed(7)
            train_x = torch.rand(10, 3, device=self.device, dtype=dtype)
            train_fidelity = torch.ones(10, 1, device=self.device, dtype=dtype)
            train_x_full = torch.cat((train_x, train_fidelity), dim=1)
            train_y = torch.rand(10, 1, device=self.device, dtype=dtype)
            mf_model = SingleTaskMultiFidelityGP(
                train_X=train_x_full,
                train_Y=train_y,
                data_fidelities=[3],
            )
            candidate_set_wrong_dim = torch.rand(
                100, 3, device=self.device, dtype=dtype
            )
            with self.assertRaisesRegex(RuntimeError, "Sizes of tensors must match"):
                acqf_class(
                    model=mf_model,
                    candidate_set=candidate_set_wrong_dim,
                    num_mv_samples=5,
                )

            # Test with SingleTaskMultiFidelityGP models.
            # MES produces NaN with torch.float for MF GP models due to
            # numerical issues in _compute_information_gain. GIBBON does not.
            if is_mes and dtype == torch.float:
                continue

            # Configs: (d, fidelity_dims, iteration_fidelity)
            mf_configs = [
                (3, [3], None),  # single fidelity dim
                (2, [3], 2),  # iteration + data fidelity dims
            ]
            for d, fidelity_dims, iteration_fidelity in mf_configs:
                torch.manual_seed(7)
                n_fidelity = len(fidelity_dims) + (iteration_fidelity is not None)
                d_full = d + n_fidelity
                train_x = torch.rand(16, d, device=self.device, dtype=dtype)
                fidelity_cols = [
                    torch.tensor([0.5, 0.75, 1.0], device=self.device, dtype=dtype)[
                        torch.randint(3, (16, 1))
                    ]
                    for _ in range(n_fidelity)
                ]
                train_x_full = torch.cat([train_x] + fidelity_cols, dim=1)
                train_y = torch.rand(16, 1, device=self.device, dtype=dtype)

                mf_model = SingleTaskMultiFidelityGP(
                    train_X=train_x_full,
                    train_Y=train_y,
                    data_fidelities=fidelity_dims,
                    iteration_fidelity=iteration_fidelity,
                )

                all_fidelity_dims = sorted(
                    fidelity_dims + ([iteration_fidelity] if iteration_fidelity else [])
                )
                target_fidelities = dict.fromkeys(all_fidelity_dims, 1.0)

                def project(X, tf=target_fidelities, _d=d_full):
                    return project_to_target_fidelity(X=X, target_fidelities=tf, d=_d)

                cost_model = AffineFidelityCostModel(
                    fidelity_weights={all_fidelity_dims[0]: 1.0}, fixed_cost=0.25
                )
                cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

                mf_candidate_set = torch.rand(
                    50, d_full, device=self.device, dtype=dtype
                )
                for dim in all_fidelity_dims:
                    mf_candidate_set[..., dim] = 1.0

                qMF_MVE = acqf_class(
                    model=mf_model,
                    candidate_set=mf_candidate_set,
                    num_mv_samples=5,
                    project=project,
                    cost_aware_utility=cost_aware_utility,
                )
                self.assertEqual(qMF_MVE.posterior_max_values.shape, torch.Size([5, 1]))

                X = torch.rand(3, 1, d_full, device=self.device, dtype=dtype)
                for dim in all_fidelity_dims:
                    X[..., dim] = 0.5
                acq_values = qMF_MVE(X)
                self.assertEqual(acq_values.shape, torch.Size([3]))
                self.assertTrue(torch.isfinite(acq_values).all())

                # Test with candidate_set of size n x d (without fidelity dims).
                # project should insert fidelity dims automatically.
                mf_candidate_set_no_fidelity = torch.rand(
                    50, d, device=self.device, dtype=dtype
                )
                qMF_MVE_no_f = acqf_class(
                    model=mf_model,
                    candidate_set=mf_candidate_set_no_fidelity,
                    num_mv_samples=5,
                    project=project,
                    cost_aware_utility=cost_aware_utility,
                )
                self.assertEqual(
                    qMF_MVE_no_f.posterior_max_values.shape, torch.Size([5, 1])
                )
                acq_values_no_f = qMF_MVE_no_f(X)
                self.assertEqual(acq_values_no_f.shape, torch.Size([3]))
                self.assertTrue(torch.isfinite(acq_values_no_f).all())

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
            for batch_shape in ((), (3,), (2, 3)):
                candidate_set = torch.rand(
                    *batch_shape, 10, 2, device=self.device, dtype=dtype
                )
                samples = sampler(
                    model=model, candidate_set=candidate_set, num_samples=5
                )
                expected_batch_shape = batch_shape or (1,)
                self.assertEqual(samples.shape, torch.Size([5, *expected_batch_shape]))

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
            self.assertEqual(samples.shape, torch.Size([5, 2, 3]))

    def test_sample_max_value_Gumbel(self):
        self._test_max_value_sampler_base(sampler=_sample_max_value_Gumbel)

    def test_sample_max_value_Gumbel_batched_matches_loop(self):
        for dtype in (torch.float, torch.double):
            torch.manual_seed(7)
            model = SingleTaskGP(
                train_X=torch.rand(10, 2, device=self.device, dtype=dtype),
                train_Y=torch.rand(10, 1, device=self.device, dtype=dtype),
            )
            candidate_set = torch.rand(2, 3, 10, 2, device=self.device, dtype=dtype)

            torch.manual_seed(11)
            batched_samples = _sample_max_value_Gumbel(
                model=model, candidate_set=candidate_set, num_samples=5
            )
            loop_samples = []
            for candidates in candidate_set.reshape(-1, 10, 2):
                torch.manual_seed(11)
                loop_samples.append(
                    _sample_max_value_Gumbel(
                        model=model, candidate_set=candidates, num_samples=5
                    )[:, 0]
                )
            expected = torch.stack(loop_samples, dim=-1).reshape(5, 2, 3)

            self.assertAllClose(batched_samples, expected)

    def test_sample_max_value_Thompson(self):
        self._test_max_value_sampler_base(sampler=_sample_max_value_Thompson)
