#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.multi_objective.base import (
    MultiObjectiveAnalyticAcquisitionFunction,
    MultiObjectiveMCAcquisitionFunction,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.acquisition.objective import IdentityMCObjective
from botorch.exceptions.errors import UnsupportedError
from botorch.models.transforms.input import InputPerturbation
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.test_helpers import DummyNonScalarizingPosteriorTransform
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class DummyMultiObjectiveAnalyticAcquisitionFunction(
    MultiObjectiveAnalyticAcquisitionFunction
):
    def forward(self, X):
        pass


class DummyMultiObjectiveMCAcquisitionFunction(MultiObjectiveMCAcquisitionFunction):
    def forward(self, X):
        pass


class DummyMCMultiOutputObjective(MCMultiOutputObjective):
    def forward(self, samples, X=None):
        if X is not None:
            return samples[..., : X.shape[-2], :]
        else:
            return samples


class TestBaseMultiObjectiveAcquisitionFunctions(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            MultiObjectiveAnalyticAcquisitionFunction()
        with self.assertRaises(TypeError):
            MultiObjectiveMCAcquisitionFunction()

    def test_init_MultiObjectiveAnalyticAcquisitionFunction(self):
        mm = MockModel(MockPosterior(mean=torch.rand(2, 1)))
        # test default init
        acqf = DummyMultiObjectiveAnalyticAcquisitionFunction(model=mm)
        self.assertTrue(acqf.posterior_transform is None)  # is None by default
        # test custom init
        posterior_transform = DummyNonScalarizingPosteriorTransform()
        acqf = DummyMultiObjectiveAnalyticAcquisitionFunction(
            model=mm, posterior_transform=posterior_transform
        )
        self.assertEqual(acqf.posterior_transform, posterior_transform)
        # test unsupported objective
        with self.assertRaises(UnsupportedError):
            DummyMultiObjectiveAnalyticAcquisitionFunction(
                model=mm, posterior_transform=IdentityMCMultiOutputObjective()
            )
        acqf = DummyMultiObjectiveAnalyticAcquisitionFunction(model=mm)
        # test set_X_pending
        with self.assertRaises(UnsupportedError):
            acqf.set_X_pending()

    def test_init_MultiObjectiveMCAcquisitionFunction(self):
        mm = MockModel(MockPosterior(mean=torch.rand(2, 1), samples=torch.rand(2, 1)))
        # test default init
        acqf = DummyMultiObjectiveMCAcquisitionFunction(model=mm)
        self.assertIsInstance(acqf.objective, IdentityMCMultiOutputObjective)
        self.assertIsNone(acqf.sampler)
        # Initialize the sampler.
        acqf.get_posterior_samples(mm.posterior(torch.ones(1, 1)))
        self.assertEqual(acqf.sampler.sample_shape, torch.Size([128]))
        self.assertIsNone(acqf.X_pending)
        # test custom init
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]))
        objective = DummyMCMultiOutputObjective()
        X_pending = torch.rand(2, 1)
        acqf = DummyMultiObjectiveMCAcquisitionFunction(
            model=mm, sampler=sampler, objective=objective, X_pending=X_pending
        )
        self.assertEqual(acqf.objective, objective)
        self.assertEqual(acqf.sampler, sampler)
        self.assertTrue(torch.equal(acqf.X_pending, X_pending))
        # test unsupported objective
        with self.assertRaises(UnsupportedError):
            DummyMultiObjectiveMCAcquisitionFunction(
                model=mm, objective=IdentityMCObjective()
            )
        # test constraints with input perturbation.
        mm.input_transform = InputPerturbation(perturbation_set=torch.rand(2, 1))
        with self.assertRaises(UnsupportedError):
            DummyMultiObjectiveMCAcquisitionFunction(
                model=mm, constraints=[lambda Z: -100.0 * torch.ones_like(Z[..., -1])]
            )
