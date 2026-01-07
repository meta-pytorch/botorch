#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch.acquisition import LearnedObjective
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    DEFAULT_NUM_PREF_SAMPLES,
    ExpectationPosteriorTransform,
    GenericMCObjective,
    IdentityMCObjective,
    LEARNED_OBJECTIVE_PREF_MODEL_MIXED_DTYPE_WARN,
    LinearMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import _get_single_precision_warning, InputDataWarning
from botorch.models.deterministic import PosteriorMeanModel
from botorch.models.pairwise_gp import PairwiseGP
from botorch.models.transforms.input import Normalize
from botorch.posteriors import GPyTorchPosterior
from botorch.posteriors.gpytorch import scalarize_posterior
from botorch.utils import apply_constraints
from botorch.utils.testing import BotorchTestCase, get_test_posterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from linear_operator.operators.dense_linear_operator import to_linear_operator
from torch import Tensor


def generic_obj_deprecated(samples: Tensor) -> Tensor:
    return torch.log(torch.sum(samples**2, dim=-1))


def generic_obj(samples: Tensor, X=None) -> Tensor:
    return generic_obj_deprecated(samples)


def infeasible_con(samples: Tensor) -> Tensor:
    return torch.ones(samples.shape[0:-1], device=samples.device, dtype=samples.dtype)


def feasible_con(samples: Tensor) -> Tensor:
    return -(
        torch.ones(samples.shape[0:-1], device=samples.device, dtype=samples.dtype)
    )


class InputDependentPosteriorTransform(PosteriorTransform):
    """A posterior transform that uses input X for testing purposes.

    This transform scales the outcomes by the mean of the input features.
    """

    scalarize = True

    def evaluate(self, Y: Tensor, X: Tensor | None = None) -> Tensor:
        if X is None:
            return Y.sum(dim=-1)
        scale = X.mean(dim=-1, keepdim=True)
        return (Y * scale).sum(dim=-1)

    def forward(
        self, posterior: GPyTorchPosterior, X: Tensor | None = None
    ) -> GPyTorchPosterior:
        m = posterior.mean.shape[-1]
        dtype, device = posterior.mean.dtype, posterior.mean.device
        weights = torch.ones(m, dtype=dtype, device=device)
        if X is not None:
            scale = X.mean()
            weights = scale * weights
        return scalarize_posterior(posterior=posterior, weights=weights, offset=0.0)


class TestPosteriorTransform(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            PosteriorTransform()


class TestInputDependentPosteriorTransform(BotorchTestCase):
    def test_input_dependent_posterior_transform(self):
        """Test posterior transform that uses input X in evaluate() and forward()."""
        from botorch.models.gp_regression import SingleTaskGP

        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            transform = InputDependentPosteriorTransform()

            # Test evaluate() with and without X
            Y = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], **tkwargs)
            X = torch.tensor([[[0.5, 0.5], [1.0, 1.0]]], **tkwargs)
            # Without X: just sum over m dimension
            self.assertAllClose(transform.evaluate(Y=Y, X=None), Y.sum(dim=-1))
            # With X: scale by mean of X, then sum
            # X.mean(dim=-1, keepdim=True) = [[[0.5], [1.0]]]
            # Y * scale = [[[0.5, 1.0], [3.0, 4.0]]], sum = [[1.5, 7.0]]
            expected = torch.tensor([[1.5, 7.0]], **tkwargs)
            self.assertAllClose(transform.evaluate(Y=Y, X=X), expected)

            # Test forward() with and without X
            posterior = get_test_posterior(batch_shape=[], m=2, **tkwargs)
            X_forward = torch.tensor([[2.0, 2.0]], **tkwargs)
            self.assertEqual(transform(posterior=posterior, X=None).mean.shape[-1], 1)
            self.assertEqual(
                transform(posterior=posterior, X=X_forward).mean.shape[-1], 1
            )

            # Test end-to-end: X is passed from model.posterior() to transform
            train_X = torch.rand(5, 2, **tkwargs)
            train_Y = torch.rand(5, 2, **tkwargs)
            model = SingleTaskGP(train_X, train_Y)
            test_X = torch.rand(3, 2, **tkwargs)
            posterior = model.posterior(X=test_X, posterior_transform=transform)
            self.assertEqual(posterior.mean.shape[-1], 1)
            self.assertEqual(posterior.mean.shape[-2], 3)


class TestScalarizedPosteriorTransform(BotorchTestCase):
    def test_scalarized_posterior_transform(self):
        for batch_shape, m, dtype in itertools.product(
            ([], [3]), (1, 2), (torch.float, torch.double)
        ):
            offset = torch.rand(1).item()
            weights = torch.randn(m, device=self.device, dtype=dtype)
            obj = ScalarizedPosteriorTransform(weights=weights, offset=offset)
            posterior = get_test_posterior(
                batch_shape, m=m, device=self.device, dtype=dtype
            )
            mean, covar = (
                posterior.distribution.mean,
                posterior.distribution.covariance_matrix,
            )
            new_posterior = obj(posterior)
            exp_size = torch.Size(batch_shape + [1, 1])
            self.assertEqual(new_posterior.mean.shape, exp_size)
            new_mean_exp = offset + mean @ weights
            self.assertAllClose(new_posterior.mean[..., -1], new_mean_exp)
            self.assertEqual(new_posterior.variance.shape, exp_size)
            new_covar_exp = ((covar @ weights) @ weights).unsqueeze(-1)
            self.assertTrue(
                torch.allclose(new_posterior.variance[..., -1], new_covar_exp)
            )
            # test error
            with self.assertRaises(ValueError):
                ScalarizedPosteriorTransform(weights=torch.rand(2, m))
            # test evaluate
            Y = torch.rand(2, m, device=self.device, dtype=dtype)
            val = obj.evaluate(Y)
            val_expected = offset + Y @ weights
            self.assertTrue(torch.equal(val, val_expected))


class TestExpectationPosteriorTransform(BotorchTestCase):
    def test_init(self):
        # Without weights.
        tf = ExpectationPosteriorTransform(n_w=5)
        self.assertEqual(tf.n_w, 5)
        self.assertAllClose(tf.weights, torch.ones(5, 1) * 0.2)
        # Errors with weights.
        with self.assertRaisesRegex(ValueError, "a tensor of size"):
            ExpectationPosteriorTransform(n_w=3, weights=torch.ones(5, 1))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ExpectationPosteriorTransform(n_w=3, weights=-torch.ones(3, 1))
        # Successful init with weights.
        weights = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        tf = ExpectationPosteriorTransform(n_w=3, weights=weights)
        self.assertAllClose(tf.weights, weights / torch.tensor([6.0, 12.0]))

    def test_evaluate(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"dtype": dtype, "device": self.device}
            # Without weights.
            tf = ExpectationPosteriorTransform(n_w=3)
            Y = torch.rand(3, 6, 2, **tkwargs)
            self.assertTrue(
                torch.allclose(tf.evaluate(Y), Y.view(3, 2, 3, 2).mean(dim=-2))
            )
            # With weights - weights intentionally doesn't use tkwargs.
            weights = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
            tf = ExpectationPosteriorTransform(n_w=2, weights=weights)
            expected = (Y.view(3, 3, 2, 2) * weights.to(Y)).sum(dim=-2) / 3.0
            self.assertAllClose(tf.evaluate(Y), expected)

    def test_expectation_posterior_transform(self):
        tkwargs = {"dtype": torch.float, "device": self.device}
        # Without weights, simple expectation, single output, no batch.
        # q = 2, n_w = 3.
        org_loc = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], **tkwargs)
        org_covar = torch.tensor(
            [
                [1.0, 0.8, 0.7, 0.3, 0.2, 0.1],
                [0.8, 1.0, 0.9, 0.25, 0.15, 0.1],
                [0.7, 0.9, 1.0, 0.2, 0.2, 0.05],
                [0.3, 0.25, 0.2, 1.0, 0.7, 0.6],
                [0.2, 0.15, 0.2, 0.7, 1.0, 0.7],
                [0.1, 0.1, 0.05, 0.6, 0.7, 1.0],
            ],
            **tkwargs,
        )
        org_mvn = MultivariateNormal(org_loc, to_linear_operator(org_covar))
        org_post = GPyTorchPosterior(distribution=org_mvn)
        tf = ExpectationPosteriorTransform(n_w=3)
        tf_post = tf(org_post)
        self.assertIsInstance(tf_post, GPyTorchPosterior)
        self.assertEqual(tf_post.sample().shape, torch.Size([1, 2, 1]))
        tf_mvn = tf_post.distribution
        self.assertIsInstance(tf_mvn, MultivariateNormal)
        expected_loc = torch.tensor([2.0, 5.0], **tkwargs)
        # This is the average of each 3 x 3 block.
        expected_covar = torch.tensor([[0.8667, 0.1722], [0.1722, 0.7778]], **tkwargs)
        self.assertAllClose(tf_mvn.loc, expected_loc)
        self.assertAllClose(tf_mvn.covariance_matrix, expected_covar, atol=1e-3)

        # With weights, 2 outputs, batched.
        tkwargs = {"dtype": torch.double, "device": self.device}
        # q = 2, n_w = 2, m = 2, leading to 8 values for loc and 8x8 cov.
        org_loc = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], **tkwargs)
        # We have 2 4x4 matrices with 0s as filler. Each block is for one outcome.
        # Each 2x2 sub block corresponds to `n_w`.
        org_covar = torch.tensor(
            [
                [1.0, 0.8, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
                [0.8, 1.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.3, 0.2, 1.2, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.2, 0.1, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.7, 0.4, 0.3],
                [0.0, 0.0, 0.0, 0.0, 0.7, 0.8, 0.3, 0.2],
                [0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 1.4, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.5, 1.2],
            ],
            **tkwargs,
        )
        # Making it batched by adding two more batches, mostly the same.
        org_loc = org_loc.repeat(3, 1)
        org_loc[1] += 100
        org_loc[2] += 1000
        org_covar = org_covar.repeat(3, 1, 1)
        # Construct the transform with weights.
        weights = torch.tensor([[1.0, 3.0], [2.0, 1.0]])
        tf = ExpectationPosteriorTransform(n_w=2, weights=weights)
        # Construct the posterior.
        org_mvn = MultitaskMultivariateNormal(
            # The return of mvn.loc and the required input are different.
            # We constructed it according to the output of mvn.loc,
            # reshaping here to have the required `b x n x t` shape.
            org_loc.view(3, 2, 4).transpose(-2, -1),
            to_linear_operator(org_covar),
            interleaved=True,  # To test the error.
        )
        org_post = GPyTorchPosterior(distribution=org_mvn)
        # Error if interleaved.
        with self.assertRaisesRegex(UnsupportedError, "interleaved"):
            tf(org_post)
        # Construct the non-interleaved posterior.
        org_mvn = MultitaskMultivariateNormal(
            org_loc.view(3, 2, 4).transpose(-2, -1),
            to_linear_operator(org_covar),
            interleaved=False,
        )
        org_post = GPyTorchPosterior(distribution=org_mvn)
        self.assertTrue(torch.equal(org_mvn.loc, org_loc))
        tf_post = tf(org_post)
        self.assertIsInstance(tf_post, GPyTorchPosterior)
        self.assertEqual(tf_post.sample().shape, torch.Size([1, 3, 2, 2]))
        tf_mvn = tf_post.distribution
        self.assertIsInstance(tf_mvn, MultitaskMultivariateNormal)
        expected_loc = torch.tensor([[1.6667, 3.6667, 5.25, 7.25]], **tkwargs).repeat(
            3, 1
        )
        expected_loc[1] += 100
        expected_loc[2] += 1000
        # This is the weighted average of each 2 x 2 block.
        expected_covar = torch.tensor(
            [
                [1.0889, 0.1667, 0.0, 0.0],
                [0.1667, 0.8, 0.0, 0.0],
                [0.0, 0.0, 0.875, 0.35],
                [0.0, 0.0, 0.35, 1.05],
            ],
            **tkwargs,
        ).repeat(3, 1, 1)
        self.assertAllClose(tf_mvn.loc, expected_loc, atol=1e-3)
        self.assertAllClose(tf_mvn.covariance_matrix, expected_covar, atol=1e-3)


class TestMCAcquisitionObjective(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            MCAcquisitionObjective()

    def test_verify_output_shape(self):
        obj = IdentityMCObjective()
        self.assertTrue(obj._verify_output_shape)
        samples = torch.zeros(2, 3, 1)
        X = torch.ones(2, 1)
        # No error if X is not given.
        obj(samples=samples)
        # Error if X is given, 2 != 3
        with self.assertRaises(RuntimeError):
            obj(samples=samples, X=X)
        # No error if _verify_output_shape=False
        obj._verify_output_shape = False
        obj(samples=samples, X=X)


class TestGenericMCObjective(BotorchTestCase):
    def test_generic_mc_objective(self):
        for dtype in (torch.float, torch.double):
            obj = GenericMCObjective(generic_obj)
            samples = torch.randn(1, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))
            samples = torch.randn(2, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))
            samples = torch.randn(3, 1, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))
            samples = torch.randn(3, 2, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), generic_obj(samples)))


class TestConstrainedMCObjective(BotorchTestCase):
    def test_constrained_mc_objective(self):
        for dtype in (torch.float, torch.double):
            # one feasible constraint
            obj = ConstrainedMCObjective(
                objective=generic_obj, constraints=[feasible_con]
            )
            samples = torch.randn(1, device=self.device, dtype=dtype)
            constrained_obj = apply_constraints(
                obj=generic_obj(samples),
                constraints=[feasible_con],
                samples=samples,
                infeasible_cost=0.0,
            )
            self.assertTrue(torch.equal(obj(samples), constrained_obj))
            # one infeasible constraint
            obj = ConstrainedMCObjective(
                objective=generic_obj, constraints=[infeasible_con]
            )
            samples = torch.randn(2, device=self.device, dtype=dtype)
            constrained_obj = apply_constraints(
                obj=generic_obj(samples),
                constraints=[infeasible_con],
                samples=samples,
                infeasible_cost=0.0,
            )
            self.assertTrue(torch.equal(obj(samples), constrained_obj))
            # one feasible, one infeasible
            obj = ConstrainedMCObjective(
                objective=generic_obj, constraints=[feasible_con, infeasible_con]
            )
            samples = torch.randn(2, 1, device=self.device, dtype=dtype)
            constrained_obj = apply_constraints(
                obj=generic_obj(samples),
                constraints=[feasible_con, infeasible_con],
                samples=samples,
                infeasible_cost=torch.tensor([0.0], device=self.device, dtype=dtype),
            )
            # one feasible, one infeasible different etas
            obj = ConstrainedMCObjective(
                objective=generic_obj,
                constraints=[feasible_con, infeasible_con],
                eta=torch.tensor([1, 10]),
            )
            samples = torch.randn(2, 1, device=self.device, dtype=dtype)
            constrained_obj = apply_constraints(
                obj=generic_obj(samples),
                constraints=[feasible_con, infeasible_con],
                samples=samples,
                eta=torch.tensor([1, 10]),
                infeasible_cost=torch.tensor([0.0], device=self.device, dtype=dtype),
            )
            self.assertTrue(torch.equal(obj(samples), constrained_obj))
            # one feasible, one infeasible, infeasible_cost
            obj = ConstrainedMCObjective(
                objective=generic_obj,
                constraints=[feasible_con, infeasible_con],
                infeasible_cost=5.0,
            )
            samples = torch.randn(3, 2, device=self.device, dtype=dtype)
            constrained_obj = apply_constraints(
                obj=generic_obj(samples),
                constraints=[feasible_con, infeasible_con],
                samples=samples,
                infeasible_cost=5.0,
            )
            self.assertTrue(torch.equal(obj(samples), constrained_obj))
            # one feasible, one infeasible, infeasible_cost, different eta
            obj = ConstrainedMCObjective(
                objective=generic_obj,
                constraints=[feasible_con, infeasible_con],
                infeasible_cost=5.0,
                eta=torch.tensor([1, 10]),
            )
            samples = torch.randn(3, 2, device=self.device, dtype=dtype)
            constrained_obj = apply_constraints(
                obj=generic_obj(samples),
                constraints=[feasible_con, infeasible_con],
                samples=samples,
                infeasible_cost=5.0,
                eta=torch.tensor([1, 10]),
            )
            self.assertTrue(torch.equal(obj(samples), constrained_obj))
            # one feasible, one infeasible, infeasible_cost, higher dimension
            obj = ConstrainedMCObjective(
                objective=generic_obj,
                constraints=[feasible_con, infeasible_con],
                infeasible_cost=torch.tensor([5.0], device=self.device, dtype=dtype),
            )
            samples = torch.randn(4, 3, 2, device=self.device, dtype=dtype)
            constrained_obj = apply_constraints(
                obj=generic_obj(samples),
                constraints=[feasible_con, infeasible_con],
                samples=samples,
                infeasible_cost=5.0,
            )
            self.assertTrue(torch.equal(obj(samples), constrained_obj))


class TestIdentityMCObjective(BotorchTestCase):
    def test_identity_mc_objective(self):
        for dtype in (torch.float, torch.double):
            obj = IdentityMCObjective()
            # single-element tensor
            samples = torch.randn(1, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), samples[0]))
            # single-dimensional non-squeezable tensor
            samples = torch.randn(2, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), samples))
            # two-dimensional squeezable tensor
            samples = torch.randn(3, 1, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), samples.squeeze(-1)))
            # two-dimensional non-squeezable tensor
            samples = torch.randn(3, 2, device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(obj(samples), samples))


class TestLinearMCObjective(BotorchTestCase):
    def test_linear_mc_objective(self) -> None:
        # Test passes for each seed
        torch.manual_seed(torch.randint(high=1000, size=(1,)))
        for dtype in (torch.float, torch.double):
            weights = torch.rand(3, device=self.device, dtype=dtype)
            obj = LinearMCObjective(weights=weights)
            samples = torch.randn(4, 2, 3, device=self.device, dtype=dtype)
            atol = 1e-7 if dtype == torch.double else 3e-7
            rtol = 1e-4 if dtype == torch.double else 4e-4
            self.assertAllClose(obj(samples), samples @ weights, atol=atol, rtol=rtol)
            samples = torch.randn(5, 4, 2, 3, device=self.device, dtype=dtype)
            self.assertAllClose(
                obj(samples),
                samples @ weights,
                atol=atol,
                rtol=rtol,
            )
            # make sure this errors if sample output dimensions are incompatible
            shape_mismatch_msg = "Output shape of samples not equal to that of weights"
            with self.assertRaisesRegex(RuntimeError, shape_mismatch_msg):
                obj(samples=torch.randn(2, device=self.device, dtype=dtype))
            with self.assertRaisesRegex(RuntimeError, shape_mismatch_msg):
                obj(samples=torch.randn(1, device=self.device, dtype=dtype))
            # make sure we can't construct objectives with multi-dim. weights
            weights_1d_msg = "weights must be a one-dimensional tensor."
            with self.assertRaisesRegex(ValueError, expected_regex=weights_1d_msg):
                LinearMCObjective(
                    weights=torch.rand(2, 3, device=self.device, dtype=dtype)
                )
            with self.assertRaisesRegex(ValueError, expected_regex=weights_1d_msg):
                LinearMCObjective(
                    weights=torch.tensor(1.0, device=self.device, dtype=dtype)
                )


class TestLearnedObjective(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = False) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        self.x_dim = 2

    def _get_pref_model(
        self,
        dtype: torch.dtype | None = None,
        input_transform: Normalize | None = None,
    ) -> PairwiseGP:
        train_X = torch.rand((2, self.x_dim), dtype=dtype)
        train_comps = torch.LongTensor([[0, 1]])
        pref_model = PairwiseGP(train_X, train_comps, input_transform=input_transform)
        return pref_model

    def test_learned_preference_objective(self) -> None:
        seed = torch.randint(low=0, high=10, size=torch.Size([1]))
        torch.manual_seed(seed)
        pref_model = self._get_pref_model(dtype=torch.float64)

        og_sample_shape = 3
        large_sample_shape = 256
        batch_size = 2
        q = 8
        test_X = torch.rand(
            torch.Size((og_sample_shape, batch_size, q, self.x_dim)),
            dtype=torch.float64,
        )
        large_X = torch.rand(
            torch.Size((large_sample_shape, batch_size, q, self.x_dim)),
            dtype=torch.float64,
        )

        # test default setting where sampler =
        # IIDNormalSampler(sample_shape=torch.Size([1]))
        with self.subTest("default sampler"):
            pref_obj = LearnedObjective(pref_model=pref_model)
            first_call_output = pref_obj(test_X)
            self.assertEqual(
                first_call_output.shape,
                torch.Size([og_sample_shape * DEFAULT_NUM_PREF_SAMPLES, batch_size, q]),
            )
            # Making sure the sampler has correct base_samples shape
            self.assertEqual(
                pref_obj.sampler.base_samples.shape,
                torch.Size([DEFAULT_NUM_PREF_SAMPLES, og_sample_shape, 1, q]),
            )
            # Passing through a same-shaped X again shouldn't change the base sample
            previous_base_samples = pref_obj.sampler.base_samples
            another_test_X = torch.rand_like(test_X)
            pref_obj(another_test_X)
            self.assertIs(pref_obj.sampler.base_samples, previous_base_samples)

            with self.assertRaisesRegex(
                ValueError, "samples should have at least 3 dimensions."
            ):
                pref_obj(torch.rand(q, self.x_dim, dtype=torch.float64))

        # test when sampler has multiple preference samples
        with self.subTest("Multiple samples"):
            num_samples = 256
            pref_obj = LearnedObjective(
                pref_model=pref_model,
                sample_shape=torch.Size([num_samples]),
            )
            self.assertEqual(
                pref_obj(test_X).shape,
                torch.Size([num_samples * og_sample_shape, batch_size, q]),
            )

            avg_obj_val = pref_obj(large_X).mean(dim=0)
            flipped_avg_obj_val = pref_obj(large_X.flip(dims=[0])).mean(dim=0)
            # Check if they are approximately close.
            # The variance is large hence the loose atol.
            self.assertAllClose(avg_obj_val, flipped_avg_obj_val, atol=1e-2)

        # test posterior mean
        with self.subTest("PosteriorMeanModel"):
            mean_pref_model = PosteriorMeanModel(model=pref_model)
            pref_obj = LearnedObjective(pref_model=mean_pref_model)
            self.assertEqual(
                pref_obj(test_X).shape,
                torch.Size([og_sample_shape, batch_size, q]),
            )

            # the order of samples shouldn't matter
            avg_obj_val = pref_obj(large_X).mean(dim=0)
            flipped_avg_obj_val = pref_obj(large_X.flip(dims=[0])).mean(dim=0)
            # When we use the posterior mean objective, they should be very close
            self.assertAllClose(avg_obj_val, flipped_avg_obj_val)

        # cannot use a deterministic model together with a sampler
        with self.subTest("deterministic model"), self.assertRaises(AssertionError):
            LearnedObjective(
                pref_model=mean_pref_model,
                sample_shape=torch.Size([num_samples]),
            )

    def test_dtype_compatibility_with_PairwiseGP(self) -> None:
        og_sample_shape = 3
        batch_size = 2
        n = 8

        test_X = torch.rand(
            torch.Size((og_sample_shape, batch_size, n, self.x_dim)),
        )

        for pref_model_dtype, test_x_dtype, expected_output_dtype in [
            (torch.float64, torch.float64, torch.float64),
            (torch.float32, torch.float32, torch.float32),
            (torch.float64, torch.float32, torch.float64),
        ]:
            with self.subTest(
                "numerical behavior",
                pref_model_dtype=pref_model_dtype,
                test_x_dtype=test_x_dtype,
                expected_output_dtype=expected_output_dtype,
            ):
                # Ignore a single-precision warning in PairwiseGP
                # and mixed-precision warning tested below
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=InputDataWarning,
                        message=_get_single_precision_warning(str(torch.float32)),
                    )
                    pref_model = self._get_pref_model(
                        dtype=pref_model_dtype,
                        input_transform=Normalize(d=2),
                    )
                pref_obj = LearnedObjective(pref_model=pref_model)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=InputDataWarning,
                        message=LEARNED_OBJECTIVE_PREF_MODEL_MIXED_DTYPE_WARN,
                    )
                    first_call_output = pref_obj(test_X.to(dtype=test_x_dtype))
                    second_call_output = pref_obj(test_X.to(dtype=test_x_dtype))

                self.assertEqual(first_call_output.dtype, expected_output_dtype)
                self.assertTrue(torch.equal(first_call_output, second_call_output))

        with self.subTest("mixed precision warning"):
            # should warn and test should pass
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=InputDataWarning)
                pref_model = self._get_pref_model(
                    dtype=torch.float64, input_transform=Normalize(d=2)
                )
            pref_obj = LearnedObjective(pref_model=pref_model)
            with self.assertWarnsRegex(
                InputDataWarning, LEARNED_OBJECTIVE_PREF_MODEL_MIXED_DTYPE_WARN
            ):
                first_call_output = pref_obj(test_X)
