#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from copy import deepcopy
from random import randint

import torch
from botorch.models.transforms.outcome import (
    Bilog,
    ChainedOutcomeTransform,
    Log,
    OutcomeTransform,
    Power,
    Standardize,
    StratifiedStandardize,
)
from botorch.models.transforms.utils import (
    norm_to_lognorm_mean,
    norm_to_lognorm_variance,
)
from botorch.posteriors import GPyTorchPosterior, TransformedPosterior
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.settings import min_variance
from linear_operator.operators import (
    BlockDiagLinearOperator,
    DenseLinearOperator,
    DiagLinearOperator,
)


def _get_test_posterior(shape, device, dtype, interleaved=True, lazy=False):
    mean = torch.rand(shape, device=device, dtype=dtype)
    n_covar = shape[-2:].numel()
    diag = torch.rand(shape, device=device, dtype=dtype)
    diag = diag.view(*diag.shape[:-2], n_covar)
    a = torch.rand(*shape[:-2], n_covar, n_covar, device=device, dtype=dtype)
    covar = a @ a.transpose(-1, -2) + torch.diag_embed(diag)
    if lazy:
        covar = DenseLinearOperator(covar)
    if shape[-1] == 1:
        mvn = MultivariateNormal(mean.squeeze(-1), covar)
    else:
        mvn = MultitaskMultivariateNormal(mean, covar, interleaved=interleaved)
    return GPyTorchPosterior(mvn)


class NotSoAbstractOutcomeTransform(OutcomeTransform):
    def forward(self, Y, Yvar):
        pass


class TestOutcomeTransforms(BotorchTestCase):
    def test_abstract_base_outcome_transform(self):
        with self.assertRaises(TypeError):
            OutcomeTransform()
        oct = NotSoAbstractOutcomeTransform()
        with self.assertRaises(NotImplementedError):
            oct.subset_output(None)
        with self.assertRaises(NotImplementedError):
            oct.untransform(None, None)
        with self.assertRaises(NotImplementedError):
            oct.untransform_posterior(None)

    def test_standardize_raises_when_mean_not_set(self) -> None:
        posterior = _get_test_posterior(
            shape=torch.Size([1, 1]), device=self.device, dtype=torch.float64
        )
        for transform in [
            Standardize(m=1),
            ChainedOutcomeTransform(
                chained=ChainedOutcomeTransform(stand=Standardize(m=1))
            ),
        ]:
            with self.assertRaises(
                RuntimeError,
                msg="`Standardize` transforms must be called on outcome data "
                "(e.g. `transform(Y)`) before calling `untransform_posterior`, since "
                "means and standard deviations need to be computed.",
            ):
                transform.untransform_posterior(posterior)

            new_tf = transform.subset_output([0])
            assert isinstance(new_tf, type(transform))

            y = torch.arange(3, device=self.device, dtype=torch.float64)
            with self.assertRaises(
                RuntimeError,
                msg="`Standardize` transforms must be called on outcome data "
                "(e.g. `transform(Y)`) before calling `untransform`, since "
                "means and standard deviations need to be computed.",
            ):
                transform.untransform(y)

    def test_is_linear(self) -> None:
        posterior = _get_test_posterior(
            shape=torch.Size([1, 1]), device=self.device, dtype=torch.float64
        )
        y = torch.arange(2, dtype=torch.float64, device=self.device)[:, None]
        standardize_tf = Standardize(m=1)
        standardize_tf(y)

        for transform in [
            standardize_tf,
            Power(power=0.5),
            Log(),
            ChainedOutcomeTransform(
                chained=ChainedOutcomeTransform(stand=standardize_tf)
            ),
            ChainedOutcomeTransform(log=Log()),
        ]:
            posterior_is_gpt = isinstance(
                transform.untransform_posterior(posterior), GPyTorchPosterior
            )
            self.assertEqual(posterior_is_gpt, transform._is_linear)

    def test_standardize_raises_when_no_observations(self) -> None:
        tf = Standardize(m=1)
        with self.assertRaisesRegex(
            ValueError, "Can't standardize with no observations."
        ):
            tf(torch.zeros(0, 1, device=self.device), None)

    def test_standardize(self) -> None:
        # test error on incompatible dim
        tf = Standardize(m=1)
        with self.assertRaisesRegex(
            RuntimeError, r"Wrong output dimension. Y.size\(-1\) is 2; expected 1."
        ):
            tf(torch.zeros(3, 2, device=self.device), None)
        # test error on incompatible batch shape
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected Y.shape\[:-2\] to be torch.Size\(\[\]\), matching the "
            "`batch_shape` argument to `Standardize`, but got "
            r"Y.shape\[:-2\]=torch.Size\(\[2\]\).",
        ):
            tf(torch.zeros(2, 3, 1, device=self.device), None)

        ms = (1, 2)
        batch_shapes = (torch.Size(), torch.Size([2]))
        dtypes = (torch.float, torch.double)
        ns = [1, 3]

        # test transform, untransform, untransform_posterior
        for m, batch_shape, dtype, n in itertools.product(ms, batch_shapes, dtypes, ns):
            # test init
            tf = Standardize(m=m, batch_shape=batch_shape)
            self.assertTrue(tf.training)
            self.assertEqual(tf._m, m)
            self.assertIsNone(tf._outputs)
            self.assertEqual(tf._batch_shape, batch_shape)
            self.assertEqual(tf._min_stdv, 1e-8)

            # no observation noise
            with torch.random.fork_rng():
                torch.manual_seed(0)
                Y = torch.rand(*batch_shape, n, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            self.assertTrue(torch.all(Y_tf.mean(dim=-2).abs() < 1e-4))
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]])
            self.assertTrue(torch.equal(Y_tf[..., [0]], Y_tf_subset))
            self.assertIsNone(Yvar_tf_subset)
            with self.assertRaises(RuntimeError):
                tf.subset_output(idcs=[0, 1, 2])

            # with observation noise
            tf = Standardize(m=m, batch_shape=batch_shape)
            with torch.random.fork_rng():
                torch.manual_seed(0)
                Y = torch.rand(*batch_shape, n, m, device=self.device, dtype=dtype)
                Yvar = 1e-8 + torch.rand(
                    *batch_shape, n, m, device=self.device, dtype=dtype
                )
            Y_tf, Yvar_tf = tf(Y, Yvar)
            self.assertTrue(tf.training)
            self.assertTrue(torch.all(Y_tf.mean(dim=-2).abs() < 1e-4))
            Yvar_tf_expected = (
                Yvar if n == 1 else Yvar / Y.std(dim=-2, keepdim=True) ** 2
            )
            self.assertAllClose(Yvar_tf, Yvar_tf_expected)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y)
            self.assertAllClose(Yvar_utf, Yvar)

            # untransform_posterior
            for interleaved, lazy in itertools.product((True, False), (True, False)):
                if m == 1 and interleaved:  # interleave has no meaning for m=1
                    continue
                shape = batch_shape + torch.Size([n, m])
                posterior = _get_test_posterior(
                    shape,
                    device=self.device,
                    dtype=dtype,
                    interleaved=interleaved,
                    lazy=lazy,
                )
                p_utf = tf.untransform_posterior(posterior)
                self.assertEqual(p_utf.device.type, self.device.type)
                self.assertTrue(p_utf.dtype == dtype)
                mean_expected = tf.means + tf.stdvs * posterior.mean
                variance_expected = tf.stdvs**2 * posterior.variance
                self.assertAllClose(p_utf.mean, mean_expected)
                self.assertAllClose(p_utf.variance, variance_expected)
                samples = p_utf.rsample()
                self.assertEqual(samples.shape, torch.Size([1]) + shape)
                samples = p_utf.rsample(sample_shape=torch.Size([4]))
                self.assertEqual(samples.shape, torch.Size([4]) + shape)
                samples2 = p_utf.rsample(sample_shape=torch.Size([4, 2]))
                self.assertEqual(samples2.shape, torch.Size([4, 2]) + shape)
                # TODO: Test expected covar (both interleaved and non-interleaved)

            # Untransform BlockDiagLinearOperator.
            if m > 1:
                base_lcv = DiagLinearOperator(
                    torch.rand(*batch_shape, m, n, device=self.device, dtype=dtype)
                )
                lcv = BlockDiagLinearOperator(base_lcv)
                mvn = MultitaskMultivariateNormal(
                    mean=torch.rand(
                        *batch_shape, n, m, device=self.device, dtype=dtype
                    ),
                    covariance_matrix=lcv,
                    interleaved=False,
                )
                posterior = GPyTorchPosterior(distribution=mvn)
                p_utf = tf.untransform_posterior(posterior)
                self.assertEqual(p_utf.device.type, self.device.type)
                self.assertTrue(p_utf.dtype == dtype)
                mean_expected = tf.means + tf.stdvs * posterior.mean
                variance_expected = tf.stdvs**2 * posterior.variance
                self.assertAllClose(p_utf.mean, mean_expected)
                self.assertAllClose(p_utf.variance, variance_expected)
                self.assertIsInstance(
                    p_utf.distribution.lazy_covariance_matrix, DiagLinearOperator
                )
                samples2 = p_utf.rsample(sample_shape=torch.Size([4, 2]))
                self.assertEqual(
                    samples2.shape,
                    torch.Size([4, 2]) + batch_shape + torch.Size([n, m]),
                )

            # untransform_posterior for non-GPyTorch posterior
            posterior2 = TransformedPosterior(
                posterior=posterior,
                sample_transform=lambda s: s,
                mean_transform=lambda m, v: m,
                variance_transform=lambda m, v: v,
            )
            p_utf2 = tf.untransform_posterior(posterior2)
            self.assertEqual(p_utf2.device.type, self.device.type)
            self.assertEqual(p_utf2.dtype, dtype)
            mean_expected = tf.means + tf.stdvs * posterior.mean
            variance_expected = tf.stdvs**2 * posterior.variance
            self.assertAllClose(p_utf2.mean, mean_expected)
            self.assertAllClose(p_utf2.variance, variance_expected)
            # TODO: Test expected covar (both interleaved and non-interleaved)
            samples = p_utf2.rsample()
            self.assertEqual(samples.shape, torch.Size([1]) + shape)
            samples = p_utf2.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4]) + shape)
            samples2 = p_utf2.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2]) + shape)

            # test error on incompatible output dimension
            # TODO: add a unit test for MTGP posterior once #840 goes in
            tf_big = Standardize(m=4)
            Y = torch.arange(4, device=self.device, dtype=dtype).reshape((1, 4))
            tf_big(Y)
            with self.assertRaisesRegex(
                RuntimeError,
                "Incompatible output dimensions encountered. Transform has output "
                f"dimension {tf_big._m} and posterior has "
                f"{posterior._extended_shape()[-1]}.",
            ):
                tf_big.untransform_posterior(posterior2)

        # test transforming a subset of outcomes
        for batch_shape, dtype in itertools.product(batch_shapes, dtypes):
            m = 2
            outputs = [-1]

            # test init
            tf = Standardize(m=m, outputs=outputs, batch_shape=batch_shape)
            self.assertTrue(tf.training)
            self.assertEqual(tf._m, m)
            self.assertEqual(tf._outputs, [1])
            self.assertEqual(tf._batch_shape, batch_shape)
            self.assertEqual(tf._min_stdv, 1e-8)

            # no observation noise
            with torch.random.fork_rng():
                torch.manual_seed(0)
                Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            Y_tf_mean = Y_tf.mean(dim=-2)
            self.assertTrue(torch.all(Y_tf_mean[..., 1].abs() < 1e-4))
            self.assertAllClose(Y_tf_mean[..., 0], Y.mean(dim=-2)[..., 0])
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]])
            self.assertTrue(torch.equal(Y_tf[..., [0]], Y_tf_subset))
            self.assertIsNone(Yvar_tf_subset)
            with self.assertRaises(RuntimeError):
                tf.subset_output(idcs=[0, 1, 2])

            # with observation noise
            tf = Standardize(m=m, outputs=outputs, batch_shape=batch_shape)
            with torch.random.fork_rng():
                torch.manual_seed(0)
                Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
                Yvar = 1e-8 + torch.rand(
                    *batch_shape, 3, m, device=self.device, dtype=dtype
                )
            Y_tf, Yvar_tf = tf(Y, Yvar)
            self.assertTrue(tf.training)
            Y_tf_mean = Y_tf.mean(dim=-2)
            self.assertTrue(torch.all(Y_tf_mean[..., 1].abs() < 1e-4))
            self.assertAllClose(Y_tf_mean[..., 0], Y.mean(dim=-2)[..., 0])
            Yvar_tf_expected = Yvar / Y.std(dim=-2, keepdim=True) ** 2
            self.assertAllClose(Yvar_tf[..., 1], Yvar_tf_expected[..., 1])
            self.assertAllClose(Yvar_tf[..., 0], Yvar[..., 0])
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y)
            self.assertAllClose(Yvar_utf, Yvar)

            # error on untransform_posterior
            with self.assertRaises(NotImplementedError):
                tf.untransform_posterior(None)

    def test_standardize_state_dict(self):
        for m in (1, 2):
            with self.subTest(m=2):
                transform = Standardize(m=m)
                self.assertFalse(transform._is_trained)
                self.assertTrue(transform.training)
                Y = torch.rand(2, m)
                transform(Y)
                state_dict = transform.state_dict()
                new_transform = Standardize(m=m)
                self.assertFalse(new_transform._is_trained)
                new_transform.load_state_dict(state_dict)
                self.assertTrue(new_transform._is_trained)

    def test_stratified_standardize(self):
        n = 5
        seed = randint(0, 100)
        torch.manual_seed(seed)
        for (
            dtype,
            batch_shape,
            observed_task_values,
            all_task_values,
        ) in itertools.product(
            (torch.float, torch.double),
            (torch.Size([]), torch.Size([3])),
            (
                torch.tensor([0, 1], dtype=torch.long, device=self.device),
                torch.tensor([0, 3], dtype=torch.long, device=self.device),
            ),
            (
                None,
                torch.tensor([0, 1, 3], dtype=torch.long, device=self.device),
            ),
        ):
            if all_task_values is None:
                all_task_values = observed_task_values
            torch.manual_seed(seed)
            tval = observed_task_values[1].item()
            X = torch.rand(*batch_shape, n, 2, dtype=dtype, device=self.device)
            X[..., -1] = torch.tensor(
                [0, tval, 0, tval, 0], dtype=dtype, device=self.device
            )
            Y = torch.randn(*batch_shape, n, 1, dtype=dtype, device=self.device)
            Yvar = torch.rand(*batch_shape, n, 1, dtype=dtype, device=self.device)
            strata_tf = StratifiedStandardize(
                observed_task_values=observed_task_values,
                all_task_values=all_task_values,
                stratification_idx=-1,
                batch_shape=batch_shape,
                default_task_value=0,
            )
            tf_Y, tf_Yvar = strata_tf(Y=Y, Yvar=Yvar, X=X)
            mask0 = X[..., -1] == 0
            mask1 = ~mask0
            Y0 = Y[mask0].view(*batch_shape, -1, 1)
            Yvar0 = Yvar[mask0].view(*batch_shape, -1, 1)
            X0 = X[mask0].view(*batch_shape, -1, 1)
            Y1 = Y[mask1].view(*batch_shape, -1, 1)
            Yvar1 = Yvar[mask1].view(*batch_shape, -1, 1)
            X1 = X[mask1].view(*batch_shape, -1, 1)
            tf0 = Standardize(m=1, batch_shape=batch_shape)
            tf_Y0, tf_Yvar0 = tf0(Y=Y0, Yvar=Yvar0, X=X0)
            tf1 = Standardize(m=1, batch_shape=batch_shape)
            tf_Y1, tf_Yvar1 = tf1(Y=Y1, Yvar=Yvar1, X=X1)
            # check that stratified means are expected
            self.assertAllClose(strata_tf.means[..., :1, :], tf0.means)
            # use remapped task values to index
            self.assertAllClose(strata_tf.means[..., 1:2, :], tf1.means)
            self.assertAllClose(strata_tf.stdvs[..., :1, :], tf0.stdvs)
            # use remapped task values to index
            self.assertAllClose(strata_tf.stdvs[..., 1:2, :], tf1.stdvs)
            # check the transformed values
            self.assertAllClose(tf_Y0, tf_Y[mask0].view(*batch_shape, -1, 1))
            self.assertAllClose(tf_Y1, tf_Y[mask1].view(*batch_shape, -1, 1))
            self.assertAllClose(tf_Yvar0, tf_Yvar[mask0].view(*batch_shape, -1, 1))
            self.assertAllClose(tf_Yvar1, tf_Yvar[mask1].view(*batch_shape, -1, 1))
            untf_Y, untf_Yvar = strata_tf.untransform(Y=tf_Y, Yvar=tf_Yvar, X=X)
            # test untransform
            if dtype == torch.float32:
                # defaults are 1e-5, 1e-8
                tols = {"rtol": 2e-5, "atol": 8e-8}
            else:
                tols = {}
            self.assertAllClose(Y, untf_Y, **tols)
            self.assertAllClose(Yvar, untf_Yvar)
            # check strata_mapping
            if not torch.equal(
                all_task_values,
                torch.tensor([0, 1], dtype=torch.long, device=self.device),
            ):
                expected_strata_mapping = torch.zeros(
                    4, dtype=torch.long, device=self.device
                )
                expected_strata_mapping[observed_task_values[1]] = 1
                self.assertTrue(
                    torch.equal(strata_tf.strata_mapping, expected_strata_mapping)
                )
            else:
                self.assertTrue(
                    torch.equal(strata_tf.strata_mapping, observed_task_values)
                )

            # test untransform_posterior
            for lazy in (True, False):
                shape = batch_shape + torch.Size([n, 1])
                posterior = _get_test_posterior(
                    shape,
                    device=self.device,
                    dtype=dtype,
                    interleaved=False,
                    lazy=lazy,
                )
                p_utf = strata_tf.untransform_posterior(posterior, X=X)
                self.assertEqual(p_utf.device.type, self.device.type)
                self.assertEqual(p_utf.dtype, dtype)
                strata_means, strata_stdvs, _ = strata_tf._get_per_input_means_stdvs(
                    X=X, include_stdvs_sq=False
                )
                mean_expected = strata_means + strata_stdvs * posterior.mean
                expected_raw_variance = (strata_stdvs**2 * posterior.variance).squeeze()
                self.assertAllClose(p_utf.mean, mean_expected)
                # The variance will be clamped to a minimum (typically 1e-6), so
                # check both the raw values and clamped values
                raw_variance = p_utf.mvn.lazy_covariance_matrix.diagonal(
                    dim1=-1, dim2=2
                )
                self.assertAllClose(raw_variance, expected_raw_variance)
                expected_clamped_variance = expected_raw_variance.clamp(
                    min=min_variance.value(dtype=raw_variance.dtype)
                ).unsqueeze(-1)
                self.assertAllClose(p_utf.variance, expected_clamped_variance)
                samples = p_utf.rsample()
                self.assertEqual(samples.shape, torch.Size([1]) + shape)
                samples = p_utf.rsample(sample_shape=torch.Size([4]))
                self.assertEqual(samples.shape, torch.Size([4]) + shape)
                samples2 = p_utf.rsample(sample_shape=torch.Size([4, 2]))
                self.assertEqual(samples2.shape, torch.Size([4, 2]) + shape)

        # test exception if X is None
        strata_tf = StratifiedStandardize(
            observed_task_values=observed_task_values,
            all_task_values=all_task_values,
            stratification_idx=-1,
            batch_shape=batch_shape,
            default_task_value=0,
        )
        with self.assertRaisesRegex(
            ValueError, "X is required for StratifiedStandardize."
        ):
            strata_tf(Y=Y, Yvar=Yvar)
        with self.assertRaisesRegex(
            ValueError, "X is required for StratifiedStandardize."
        ):
            strata_tf.untransform_posterior(posterior)
        with self.assertRaisesRegex(
            ValueError, "X is required for StratifiedStandardize."
        ):
            strata_tf.untransform(Y=tf_Y)
        with self.assertRaises(NotImplementedError):
            strata_tf.subset_output(idcs=[0])

    def test_log(self):
        ms = (1, 2)
        batch_shapes = (torch.Size(), torch.Size([2]))
        dtypes = (torch.float, torch.double)

        # test transform and untransform
        for m, batch_shape, dtype in itertools.product(ms, batch_shapes, dtypes):
            # test init
            tf = Log()
            self.assertTrue(tf.training)
            self.assertIsNone(tf._outputs)

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            self.assertAllClose(Y_tf, torch.log(Y))
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            torch.allclose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]])
            self.assertTrue(torch.equal(Y_tf[..., [0]], Y_tf_subset))
            self.assertIsNone(Yvar_tf_subset)

            # test error if observation noise present
            tf = Log()
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            with self.assertRaises(NotImplementedError):
                tf(Y, Yvar)
            tf.eval()
            with self.assertRaises(NotImplementedError):
                tf.untransform(Y, Yvar)

            # untransform_posterior
            tf = Log()
            Y_tf, Yvar_tf = tf(Y, None)
            tf.eval()
            shape = batch_shape + torch.Size([3, m])
            posterior = _get_test_posterior(shape, device=self.device, dtype=dtype)
            p_utf = tf.untransform_posterior(posterior)
            self.assertIsInstance(p_utf, TransformedPosterior)
            self.assertEqual(p_utf.device.type, self.device.type)
            self.assertTrue(p_utf.dtype == dtype)
            self.assertTrue(p_utf._sample_transform == torch.exp)
            mean_expected = norm_to_lognorm_mean(posterior.mean, posterior.variance)
            variance_expected = norm_to_lognorm_variance(
                posterior.mean, posterior.variance
            )
            self.assertAllClose(p_utf.mean, mean_expected)
            self.assertAllClose(p_utf.variance, variance_expected)
            samples = p_utf.rsample()
            self.assertEqual(samples.shape, torch.Size([1]) + shape)
            samples = p_utf.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4]) + shape)
            samples2 = p_utf.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2]) + shape)

        # test transforming a subset of outcomes
        for batch_shape, dtype in itertools.product(batch_shapes, dtypes):
            m = 2
            outputs = [-1]

            # test init
            tf = Log(outputs=outputs)
            self.assertTrue(tf.training)
            # cannot normalize indices b/c we don't know dimension yet
            self.assertEqual(tf._outputs, [-1])

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            self.assertAllClose(Y_tf[..., 1], torch.log(Y[..., 1]))
            self.assertAllClose(Y_tf[..., 0], Y[..., 0])
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            torch.allclose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            with self.assertRaises(NotImplementedError):
                tf_subset = tf.subset_output(idcs=[0])

            # with observation noise
            tf = Log(outputs=outputs)
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            with self.assertRaises(NotImplementedError):
                tf(Y, Yvar)

            # error on untransform_posterior
            with self.assertRaises(NotImplementedError):
                tf.untransform_posterior(None)

            # test subset_output with positive on subset of outcomes (pos. index)
            tf = Log(outputs=[0])
            Y_tf, Yvar_tf = tf(Y, None)
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]], None)
            self.assertTrue(torch.equal(Y_tf_subset, Y_tf[..., [0]]))
            self.assertIsNone(Yvar_tf_subset)

    def test_chained_outcome_transform(self):
        ms = (1, 2)
        batch_shapes = (torch.Size(), torch.Size([2]))
        dtypes = (torch.float, torch.double)

        # test transform and untransform
        for m, batch_shape, dtype in itertools.product(ms, batch_shapes, dtypes):
            # test init
            tf1 = Log()
            tf2 = Standardize(m=m, batch_shape=batch_shape)
            tf = ChainedOutcomeTransform(b=tf1, a=tf2)
            self.assertTrue(tf.training)
            self.assertEqual(list(tf.keys()), ["b", "a"])
            self.assertEqual(tf["b"], tf1)
            self.assertEqual(tf["a"], tf2)

            # make copies for validation below
            tf1_, tf2_ = deepcopy(tf1), deepcopy(tf2)

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            Y_tf_, Yvar_tf_ = tf2_(*tf1_(Y, None))
            self.assertTrue(tf.training)
            self.assertIsNone(Yvar_tf_)
            self.assertAllClose(Y_tf, Y_tf_)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            torch.allclose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]])
            self.assertTrue(torch.equal(Y_tf[..., [0]], Y_tf_subset))
            self.assertIsNone(Yvar_tf_subset)
            with self.assertRaises(RuntimeError):
                tf.subset_output(idcs=[0, 1, 2])

            # test error if observation noise present
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            with self.assertRaises(NotImplementedError):
                tf(Y, Yvar)

            # untransform_posterior
            tf1 = Log()
            tf2 = Standardize(m=m, batch_shape=batch_shape)
            tf = ChainedOutcomeTransform(log=tf1, standardize=tf2)
            Y_tf, Yvar_tf = tf(Y, None)
            tf.eval()
            shape = batch_shape + torch.Size([3, m])
            posterior = _get_test_posterior(shape, device=self.device, dtype=dtype)
            p_utf = tf.untransform_posterior(posterior)
            self.assertIsInstance(p_utf, TransformedPosterior)
            self.assertEqual(p_utf.device.type, self.device.type)
            self.assertTrue(p_utf.dtype == dtype)
            samples = p_utf.rsample()
            self.assertEqual(samples.shape, torch.Size([1]) + shape)
            samples = p_utf.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4]) + shape)
            samples2 = p_utf.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2]) + shape)

        # test transforming a subset of outcomes
        for batch_shape, dtype in itertools.product(batch_shapes, dtypes):
            m = 2
            outputs = [-1]

            # test init
            tf1 = Log(outputs=outputs)
            tf2 = Standardize(m=m, outputs=outputs, batch_shape=batch_shape)
            tf = ChainedOutcomeTransform(log=tf1, standardize=tf2)
            self.assertTrue(tf.training)
            self.assertEqual(sorted(tf.keys()), ["log", "standardize"])
            self.assertEqual(tf["log"], tf1)
            self.assertEqual(tf["standardize"], tf2)
            self.assertEqual(tf["log"]._outputs, [-1])  # don't know dimension yet
            self.assertEqual(tf["standardize"]._outputs, [1])

            # make copies for validation below
            tf1_, tf2_ = deepcopy(tf1), deepcopy(tf2)

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            Y_tf_, Yvar_tf_ = tf2_(*tf1_(Y, None))
            self.assertTrue(tf.training)
            self.assertIsNone(Yvar_tf_)
            self.assertAllClose(Y_tf, Y_tf_)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            torch.allclose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # with observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            with self.assertRaises(NotImplementedError):
                tf(Y, Yvar)

            # error on untransform_posterior
            with self.assertRaises(NotImplementedError):
                tf.untransform_posterior(None)

    def test_power(self, seed=0):
        torch.random.manual_seed(seed)

        ms = (1, 2)
        batch_shapes = (torch.Size(), torch.Size([2]))
        dtypes = (torch.float, torch.double)
        power = 1 / 3

        # test transform and untransform
        for m, batch_shape, dtype in itertools.product(ms, batch_shapes, dtypes):
            # test init
            tf = Power(power=power)
            self.assertTrue(tf.training)
            self.assertIsNone(tf._outputs)

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            self.assertAllClose(Y_tf, Y.pow(power))
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]])
            self.assertTrue(torch.equal(Y_tf[..., [0]], Y_tf_subset))
            self.assertIsNone(Yvar_tf_subset)

            # test error if observation noise present
            tf = Power(power=power)
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            with self.assertRaises(NotImplementedError):
                tf(Y, Yvar)
            tf.eval()
            with self.assertRaises(NotImplementedError):
                tf.untransform(Y, Yvar)

            # untransform_posterior
            tf = Power(power=power)
            Y_tf, Yvar_tf = tf(Y, None)
            tf.eval()
            shape = batch_shape + torch.Size([3, m])
            posterior = _get_test_posterior(shape, device=self.device, dtype=dtype)
            p_utf = tf.untransform_posterior(posterior)
            self.assertIsInstance(p_utf, TransformedPosterior)
            self.assertEqual(p_utf.device.type, self.device.type)
            self.assertTrue(p_utf.dtype == dtype)

            samples = p_utf.rsample()
            self.assertEqual(samples.shape, torch.Size([1]) + shape)
            samples = p_utf.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4]) + shape)
            samples2 = p_utf.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2]) + shape)

        # test transforming a subset of outcomes
        for batch_shape, dtype in itertools.product(batch_shapes, dtypes):
            m = 2
            outputs = [-1]

            # test init
            tf = Power(power=power, outputs=outputs)
            self.assertTrue(tf.training)
            # cannot normalize indices b/c we don't know dimension yet
            self.assertEqual(tf._outputs, [-1])

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            self.assertAllClose(Y_tf[..., 1], Y[..., 1].pow(power))
            self.assertAllClose(Y_tf[..., 0], Y[..., 0])
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            with self.assertRaises(NotImplementedError):
                tf_subset = tf.subset_output(idcs=[0])

            # with observation noise
            tf = Power(power=power, outputs=outputs)
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            with self.assertRaises(NotImplementedError):
                tf(Y, Yvar)

            # error on untransform_posterior
            with self.assertRaises(NotImplementedError):
                tf.untransform_posterior(None)

            # test subset_output with positive on subset of outcomes (pos. index)
            tf = Power(power=power, outputs=[0])
            Y_tf, Yvar_tf = tf(Y, None)
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]], None)
            self.assertTrue(torch.equal(Y_tf_subset, Y_tf[..., [0]]))
            self.assertIsNone(Yvar_tf_subset)

    def test_bilog(self, seed=0):
        torch.random.manual_seed(seed)

        ms = (1, 2)
        batch_shapes = (torch.Size(), torch.Size([2]))
        dtypes = (torch.float, torch.double)

        # test transform and untransform
        for m, batch_shape, dtype in itertools.product(ms, batch_shapes, dtypes):
            # test init
            tf = Bilog()
            self.assertTrue(tf.training)
            self.assertIsNone(tf._outputs)

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            self.assertAllClose(Y_tf, Y.sign() * (Y.abs() + 1).log())
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y, atol=1e-7)
            self.assertIsNone(Yvar_utf)

            # subset_output
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]])
            self.assertTrue(torch.equal(Y_tf[..., [0]], Y_tf_subset))
            self.assertIsNone(Yvar_tf_subset)

            # test error if observation noise present
            tf = Bilog()
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            with self.assertRaises(NotImplementedError):
                tf(Y, Yvar)
            tf.eval()
            with self.assertRaises(NotImplementedError):
                tf.untransform(Y, Yvar)

            # untransform_posterior
            tf = Bilog()
            Y_tf, Yvar_tf = tf(Y, None)
            tf.eval()
            shape = batch_shape + torch.Size([3, m])
            posterior = _get_test_posterior(shape, device=self.device, dtype=dtype)
            p_utf = tf.untransform_posterior(posterior)
            self.assertIsInstance(p_utf, TransformedPosterior)
            self.assertEqual(p_utf.device.type, self.device.type)
            self.assertTrue(p_utf.dtype == dtype)

            samples = p_utf.rsample()
            self.assertEqual(samples.shape, torch.Size([1]) + shape)
            samples = p_utf.rsample(sample_shape=torch.Size([4]))
            self.assertEqual(samples.shape, torch.Size([4]) + shape)
            samples2 = p_utf.rsample(sample_shape=torch.Size([4, 2]))
            self.assertEqual(samples2.shape, torch.Size([4, 2]) + shape)

        # test transforming a subset of outcomes
        for batch_shape, dtype in itertools.product(batch_shapes, dtypes):
            m = 2
            outputs = [-1]

            # test init
            tf = Bilog(outputs=outputs)
            self.assertTrue(tf.training)
            # cannot normalize indices b/c we don't know dimension yet
            self.assertEqual(tf._outputs, [-1])

            # no observation noise
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Y_tf, Yvar_tf = tf(Y, None)
            self.assertTrue(tf.training)
            self.assertTrue(
                torch.allclose(
                    Y_tf[..., 1], Y[..., 1].sign() * (Y[..., 1].abs() + 1).log()
                )
            )
            self.assertAllClose(Y_tf[..., 0], Y[..., 0])
            self.assertIsNone(Yvar_tf)
            tf.eval()
            self.assertFalse(tf.training)
            Y_utf, Yvar_utf = tf.untransform(Y_tf, Yvar_tf)
            self.assertAllClose(Y_utf, Y)
            self.assertIsNone(Yvar_utf)

            # subset_output
            with self.assertRaises(NotImplementedError):
                tf_subset = tf.subset_output(idcs=[0])

            # with observation noise
            tf = Bilog(outputs=outputs)
            Y = torch.rand(*batch_shape, 3, m, device=self.device, dtype=dtype)
            Yvar = 1e-8 + torch.rand(
                *batch_shape, 3, m, device=self.device, dtype=dtype
            )
            with self.assertRaises(NotImplementedError):
                tf(Y, Yvar)

            # error on untransform_posterior
            with self.assertRaises(NotImplementedError):
                tf.untransform_posterior(None)

            # test subset_output with positive on subset of outcomes (pos. index)
            tf = Bilog(outputs=[0])
            Y_tf, Yvar_tf = tf(Y, None)
            tf_subset = tf.subset_output(idcs=[0])
            Y_tf_subset, Yvar_tf_subset = tf_subset(Y[..., [0]], None)
            self.assertTrue(torch.equal(Y_tf_subset, Y_tf[..., [0]]))
            self.assertIsNone(Yvar_tf_subset)
