#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform

from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.models.deterministic import (
    AffineDeterministicModel,
    DeterministicModel,
    FixedSingleSampleModel,
    GenericDeterministicModel,
    MatheronPathModel,
    PosteriorMeanModel,
)
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.utils.testing import BotorchTestCase
from torch import Size


class DummyDeterministicModel(DeterministicModel):
    r"""A dummy deterministic model that uses transforms."""

    def __init__(self, outcome_transform, input_transform):
        r"""
        Args:
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass. Only input transforms are allowed which do not
                transform the categorical dimensions. This can be achieved
                by using the `indices` argument when constructing the transform.
        """
        super().__init__()
        self.input_transform = input_transform
        self.outcome_transform = outcome_transform

    def forward(self, X):
        # just a non-linear objective that is sure to break without transforms
        return (X - 1.0).pow(2).sum(dim=-1, keepdim=True) - 5.0


class TestDeterministicModels(BotorchTestCase):
    def test_abstract_base_model(self):
        with self.assertRaises(TypeError):
            DeterministicModel()

    def test_GenericDeterministicModel(self):
        def f(X):
            return X.mean(dim=-1, keepdim=True)

        model = GenericDeterministicModel(f)
        self.assertEqual(model.num_outputs, 1)
        d = 2
        X = torch.rand(3, d)
        # basic test
        p = model.posterior(X)
        self.assertIsInstance(p, EnsemblePosterior)
        self.assertEqual(p.ensemble_size, 1)
        self.assertTrue(torch.equal(p.mean, f(X)))
        # check that observation noise doesn't change things
        p_noisy = model.posterior(X, observation_noise=True)
        self.assertTrue(torch.equal(p_noisy.mean, f(X)))
        # test proper error on explicit observation noise
        with self.assertRaises(UnsupportedError):
            model.posterior(X, observation_noise=X[..., :-1])
        # check output indices
        model = GenericDeterministicModel(lambda X: X, num_outputs=2)
        self.assertEqual(model.num_outputs, 2)
        p = model.posterior(X, output_indices=[0])
        self.assertTrue(torch.equal(p.mean, X[..., [0]]))
        # test subset output
        subset_model = model.subset_output([0])
        self.assertIsInstance(subset_model, GenericDeterministicModel)
        p_sub = subset_model.posterior(X)
        self.assertTrue(torch.equal(p_sub.mean, X[..., [0]]))

        # testing batched model
        batch_shape = torch.Size([2, 4])
        batch_coefficients = torch.rand(*batch_shape, 1, d)

        def batched_f(X):
            return (X * batch_coefficients).sum(dim=-1, keepdim=True)

        model = GenericDeterministicModel(batched_f, batch_shape=batch_shape)
        Y = model(X)
        self.assertEqual(Y.shape, torch.Size([2, 4, 3, 1]))

        # testing with wrong batch shape
        model = GenericDeterministicModel(batched_f, batch_shape=torch.Size([2]))

        with self.assertRaisesRegex(
            ValueError, "GenericDeterministicModel was initialized with batch_shape="
        ):
            model(X)

    def test_AffineDeterministicModel(self):
        # test error on bad shape of a
        with self.assertRaises(ValueError):
            AffineDeterministicModel(torch.rand(2))
        # test error on bad shape of b
        with self.assertRaises(ValueError):
            AffineDeterministicModel(torch.rand(2, 1), torch.rand(2, 1))
        # test one-dim output
        a = torch.rand(3, 1)
        model = AffineDeterministicModel(a)
        self.assertEqual(model.num_outputs, 1)
        for shape in ((4, 3), (1, 4, 3)):
            X = torch.rand(*shape)
            p = model.posterior(X)
            mean_exp = model.b + (X.unsqueeze(-1) * a).sum(dim=-2)
            self.assertAllClose(p.mean, mean_exp)
        # # test two-dim output
        a = torch.rand(3, 2)
        model = AffineDeterministicModel(a)
        self.assertEqual(model.num_outputs, 2)
        for shape in ((4, 3), (1, 4, 3)):
            X = torch.rand(*shape)
            p = model.posterior(X)
            mean_exp = model.b + (X.unsqueeze(-1) * a).sum(dim=-2)
            self.assertAllClose(p.mean, mean_exp)
        # test subset output
        X = torch.rand(4, 3)
        subset_model = model.subset_output([0])
        self.assertIsInstance(subset_model, AffineDeterministicModel)
        p = model.posterior(X)
        p_sub = subset_model.posterior(X)
        self.assertTrue(torch.equal(p_sub.mean, p.mean[..., [0]]))

    def test_with_transforms(self):
        dim = 2
        bounds = torch.stack([torch.zeros(dim), torch.ones(dim) * 3])
        intf = Normalize(d=dim, bounds=bounds)
        octf = Standardize(m=1)
        # update octf state with dummy data
        octf(torch.rand(5, 1) * 7)
        octf.eval()
        model = DummyDeterministicModel(octf, intf)
        # check that the posterior output agrees with the manually transformed one
        test_X = torch.rand(3, dim)
        expected_Y, _ = octf.untransform(model.forward(intf(test_X)))
        with warnings.catch_warnings(record=True) as ws:
            posterior = model.posterior(test_X)
            msg = "does not have a `train_inputs` attribute"
            self.assertTrue(any(msg in str(w.message) for w in ws))
        self.assertAllClose(expected_Y, posterior.mean)
        # check that model.train/eval works and raises the warning
        model.train()
        with self.assertWarns(RuntimeWarning):
            model.eval()

    def test_posterior_transform(self):
        def f(X):
            return X

        model = GenericDeterministicModel(f)
        test_X = torch.rand(3, 2)
        post_tf = ScalarizedPosteriorTransform(weights=torch.rand(2))
        # expect error due to post_tf expecting an MVN
        with self.assertRaises(NotImplementedError):
            model.posterior(test_X, posterior_transform=post_tf)

    def test_PosteriorMeanModel(self):
        train_X = torch.rand(2, 3)
        train_Y = torch.rand(2, 2)
        model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
        mean_model = PosteriorMeanModel(model=model)
        self.assertTrue(mean_model.num_outputs == train_Y.shape[-1])
        self.assertTrue(mean_model.batch_shape == torch.Size([]))

        test_X = torch.rand(2, 3)
        post = model.posterior(test_X)
        mean_post = mean_model.posterior(test_X)
        self.assertTrue((mean_post.variance == 0).all())
        self.assertTrue(torch.equal(post.mean, mean_post.mean))

    def test_FixedSingleSampleModel(self):
        torch.manual_seed(123)
        train_X = torch.rand(2, 3)
        train_Y = torch.rand(2, 2)
        model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
        fss_model = FixedSingleSampleModel(model=model)

        # test without specifying w and dim
        test_X = torch.rand(2, 3)
        w = fss_model.w
        post = model.posterior(test_X)
        original_output = post.mean + post.variance.sqrt() * w
        fss_output = fss_model(test_X)
        self.assertAllClose(original_output, fss_output)

        self.assertTrue(hasattr(fss_model, "num_outputs"))

        # test specifying w
        w = torch.randn(4)
        fss_model = FixedSingleSampleModel(model=model, w=w)
        self.assertTrue(fss_model.w.shape == w.shape)
        # test dim
        dim = 5
        fss_model = FixedSingleSampleModel(model=model, w=w, dim=dim)
        # dim should be ignored
        self.assertTrue(fss_model.w.shape == w.shape)
        # test dim when no w is provided
        fss_model = FixedSingleSampleModel(model=model, dim=dim)
        # dim should be ignored
        self.assertTrue(fss_model.w.shape == torch.Size([dim]))

        # check w dtype conversion
        train_X_double = torch.rand(2, 3, dtype=torch.double)
        train_Y_double = torch.rand(2, 2, dtype=torch.double)
        model_double = SingleTaskGP(train_X=train_X_double, train_Y=train_Y_double)
        fss_model_double = FixedSingleSampleModel(model=model_double)
        test_X_float = torch.rand(2, 3, dtype=torch.float)

        # the following line should execute fine
        fss_model_double.posterior(test_X_float)


class TestMatheronPathModel(BotorchTestCase):
    def test_MatheronPathModel(self) -> None:
        """Test MatheronPathModel basic class attributes and properties."""
        tkwargs = {"device": self.device, "dtype": torch.double}

        # Setup test model
        train_X = torch.rand(5, 2, **tkwargs)
        train_Y = torch.rand(5, 1, **tkwargs)
        model = SingleTaskGP(train_X, train_Y)

        # Test basic class instantiation and attributes
        path_model = MatheronPathModel(model=model)
        self.assertIsInstance(path_model, DeterministicModel)
        self.assertEqual(path_model.num_outputs, model.num_outputs)
        self.assertEqual(path_model.batch_shape, model.batch_shape)
        self.assertFalse(path_model._is_ensemble)

        # Test with sample_shape
        sample_shape = Size([3])
        path_model = MatheronPathModel(model=model, sample_shape=sample_shape)
        self.assertTrue(path_model._is_ensemble)
        expected_batch_shape = sample_shape + model.batch_shape
        self.assertEqual(path_model.batch_shape, expected_batch_shape)

        # Test basic forward pass
        test_X = torch.rand(4, 2, **tkwargs)
        output = path_model(test_X)
        expected_shape = torch.Size([3, 4, 1])
        self.assertEqual(output.shape, expected_shape)

        # Test that output is deterministic (same inputs give same outputs)
        output2 = path_model(test_X)
        self.assertTrue(torch.equal(output, output2))

        # Test seed functionality, same seed should produce same outputs
        seed = 123
        path_model1 = MatheronPathModel(model=model, seed=seed)
        path_model2 = MatheronPathModel(model=model, seed=seed)
        test_X = torch.rand(4, 2, **tkwargs)
        self.assertTrue(torch.equal(path_model1(test_X), path_model2(test_X)))

        # Not setting seed should produce different outputs
        path_model1 = MatheronPathModel(model=model, seed=None)
        path_model2 = MatheronPathModel(model=model, seed=None)
        test_X = torch.rand(1, 2, **tkwargs)
        self.assertFalse(torch.equal(path_model1(test_X), path_model2(test_X)))

        # Different seeds should produce different outputs
        path_model3 = MatheronPathModel(model=model, seed=456)
        self.assertFalse(torch.equal(path_model1(test_X), path_model3(test_X)))
