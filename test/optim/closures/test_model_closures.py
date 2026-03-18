#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import zip_longest
from math import pi

import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.optim.closures.model_closures import (
    get_loss_closure,
    get_loss_closure_with_grads,
)
from botorch.utils.testing import BotorchTestCase
from gpytorch import settings as gpytorch_settings
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.module import Module
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


# Mock wrapping the __call__ directly is leading to errors like
# TypeError: super(type, obj): obj must be an instance or subtype of type
# so, doing this manually here.
class WrapperLikelihood(GaussianLikelihood):
    def __init__(self, base_likelihood: GaussianLikelihood):
        """A wrapper around a GaussianLikelihood that stores the call args."""
        Module.__init__(self)
        self.base_likelihood = base_likelihood
        self.call_args = []

    def __call__(self, *args, **kwargs):
        # Store the train inputs arg for testing.
        self.call_args.append(args[1])
        return self.base_likelihood(*args, **kwargs)


def _get_mlls(
    device: torch.device, wrap_likelihood: bool = False
) -> tuple[Tensor, list[MarginalLogLikelihood]]:
    """Returns the train X, along two MLLs: one for a SingleTaskGP and
    one for a ModelListGP.

    Args:
        device: The device to use.
        wrap_likelihood: If True, wrap the likelihood in a WrapperLikelihood.
            This is useful for comparing call args later.
    """
    with torch.random.fork_rng():
        torch.manual_seed(0)
        # Inputs are not in the unit cube to ensure input transform is applied.
        train_X = torch.linspace(0, 5, 10).unsqueeze(-1)
        train_Y = torch.sin((2 * pi) * train_X)
        train_Y = train_Y + 0.1 * torch.randn_like(train_Y)
    mlls = []
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=1),
    )
    if wrap_likelihood:
        model.likelihood = WrapperLikelihood(model.likelihood)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mlls.append(mll.to(device=device, dtype=torch.double))

    model = ModelListGP(model, model)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    mlls.append(mll.to(device=device, dtype=torch.double))
    return train_X.to(device=device, dtype=torch.double), mlls


class TestComputeLossProtocol(BotorchTestCase):
    def test_compute_custom_loss_closure_dispatch_paths(self) -> None:
        # If mll has a compute_custom_loss method, get_loss_closure returns it
        # directly.
        with self.subTest("compute_custom_loss_on_mll"):
            _, mlls = _get_mlls(device=self.device)
            mll = mlls[0]  # ExactMarginalLogLikelihood

            sentinel = lambda **_kwargs: torch.tensor(42.0)  # noqa: E731
            mll.compute_custom_loss = sentinel
            result = get_loss_closure(mll)
            self.assertIs(result, sentinel)
            del mll.compute_custom_loss

        # If mll.model has a compute_custom_loss method, get_loss_closure wraps it.
        with self.subTest("compute_custom_loss_on_model"):
            _, mlls = _get_mlls(device=self.device)
            mll = mlls[0]

            def my_compute_custom_loss(
                *, mll: MarginalLogLikelihood, **_kwargs: object
            ) -> Tensor:
                return torch.tensor(-99.0)

            mll.model.compute_custom_loss = my_compute_custom_loss
            result = get_loss_closure(mll)
            # Should be a partial wrapping my_compute_custom_loss with mll as first arg
            self.assertEqual(result(), torch.tensor(-99.0))
            del mll.model.compute_custom_loss

        # Without compute_custom_loss, get_loss_closure uses isinstance checks.
        with self.subTest("isinstance_fallback"):
            _, mlls = _get_mlls(device=self.device)
            mll = mlls[0]
            self.assertFalse(hasattr(mll, "compute_custom_loss"))
            self.assertFalse(hasattr(mll.model, "compute_custom_loss"))
            closure = get_loss_closure(mll)
            loss = closure()
            self.assertIsInstance(loss, Tensor)

        # compute_custom_loss on mll takes priority over compute_custom_loss on model.
        with self.subTest("mll_takes_priority_over_model"):
            _, mlls = _get_mlls(device=self.device)
            mll = mlls[0]

            mll_sentinel = lambda **_kwargs: torch.tensor(1.0)  # noqa: E731
            model_sentinel = lambda *, mll, **_kwargs: torch.tensor(2.0)  # noqa: E731
            mll.compute_custom_loss = mll_sentinel
            mll.model.compute_custom_loss = model_sentinel
            result = get_loss_closure(mll)
            self.assertIs(result, mll_sentinel)
            del mll.compute_custom_loss
            del mll.model.compute_custom_loss


class TestLossClosures(BotorchTestCase):
    def test_main(self) -> None:
        for mll in _get_mlls(device=self.device)[1]:
            out = mll.model(*mll.model.train_inputs)
            loss = -mll(out, mll.model.train_targets).sum()
            loss.backward()
            params = {n: p for n, p in mll.named_parameters() if p.requires_grad}
            grads = [
                torch.zeros_like(p) if p.grad is None else p.grad
                for p in params.values()
            ]

            closure = get_loss_closure(mll)
            self.assertTrue(loss.equal(closure()))

            closure = get_loss_closure_with_grads(mll, params)
            _loss, _grads = closure()
            self.assertTrue(loss.equal(_loss))
            self.assertTrue(all(a.equal(b) for a, b in zip_longest(grads, _grads)))

    def test_data_loader(self) -> None:
        for mll in _get_mlls(device=self.device)[1]:
            if type(mll) is not ExactMarginalLogLikelihood:
                continue

            dataset = TensorDataset(*mll.model.train_inputs, mll.model.train_targets)
            loader = DataLoader(dataset, batch_size=len(mll.model.train_targets))
            params = {n: p for n, p in mll.named_parameters() if p.requires_grad}
            A = get_loss_closure_with_grads(mll, params)
            (a, das) = A()

            B = get_loss_closure_with_grads(mll, params, data_loader=loader)
            with gpytorch_settings.debug(False):  # disables GPyTorch's internal check
                (b, dbs) = B()

            self.assertAllClose(a, b)
            for da, db in zip_longest(das, dbs):
                self.assertAllClose(da, db)

        loader = DataLoader(mll.model.train_targets, len(mll.model.train_targets))
        closure = get_loss_closure_with_grads(mll, params, data_loader=loader)
        with self.assertRaisesRegex(TypeError, "Expected .* a batch of tensors"):
            closure()

    def test_with_input_transforms(self) -> None:
        # This test reproduces the bug reported in issue #2515.
        train_X, mlls = _get_mlls(device=self.device, wrap_likelihood=True)
        for mll in mlls:
            if isinstance(mll, SumMarginalLogLikelihood):
                # The likelihood is called twice here since it is the same
                # likelihood in both child models.
                likelihood = mll.model.models[0].likelihood
                expected_calls1 = 2  # In the closure call.
                expected_calls2 = 6  # Closure + posterior calls.
            else:
                likelihood = mll.model.likelihood
                expected_calls1 = 1  # In the closure call.
                expected_calls2 = 4  # Closure + posterior calls.
            likelihood.call_args = []  # reset since it is shared between the models.
            params = {n: p for n, p in mll.named_parameters() if p.requires_grad}
            # Evaluate the closure to mimic the model fitting process.
            mll.train()
            closure = get_loss_closure_with_grads(mll, params)
            closure()
            self.assertEqual(len(likelihood.call_args), expected_calls1)
            # Call the model posterior to reproduce post-fitting usage.
            mll.model.posterior(train_X, observation_noise=True)
            # Compare the call args to ensure they're all the same.
            # Likelihood is called twice on model(X) and once for adding the noise.
            self.assertEqual(len(likelihood.call_args), expected_calls2)
            arg0 = likelihood.call_args[0]
            for i in range(1, expected_calls2):
                argi = likelihood.call_args[i]
                # The arg may be a tensor or a single element list of the tensor.
                self.assertAllClose(
                    arg0 if isinstance(arg0, Tensor) else arg0[0],
                    argi if isinstance(argi, Tensor) else argi[0],
                )
