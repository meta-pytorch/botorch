#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Base specs and test mixins for acquisition function testing."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions import BotorchWarning
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.test_helpers import get_model
from botorch.utils.testing import MockModel, MockPosterior


def loop_filtered_specs(test_method: Callable) -> Callable:
    """Decorator that runs a test method for each acquisition spec.

    Automatically skips specs that have the test name in their bypass_tests list.
    The decorated method receives `spec` as its first argument after `self`.

    Usage:
        @loop_filtered_specs
        def test_something(self, spec: AcquisitionSpec) -> None:
            # Test code here - no need for manual iteration or bypass checks
    """

    @wraps(test_method)
    def wrapper(self) -> None:
        test_name = test_method.__name__
        for spec in self.acquisition_specs:
            if test_name in spec.bypass_tests:
                continue
            # subTest does not specify with acqf_class, fails, thus we propagate the
            # error trace here.
            with self.subTest(acqf_class=spec.acqf_class.__name__):
                try:
                    test_method(self, spec)
                except Exception as e:
                    raise type(e)(f"[{spec.acqf_class.__name__}] {e}").with_traceback(
                        e.__traceback__
                    ) from None

    return wrapper


@dataclass
class AcquisitionSpec:
    """Base spec for single-output acquisition functions requiring only model
    and standard keyword arguments (e.g., best_f, beta).

    For acquisition functions with special requirements like X_baseline or
    constraints, use specialized spec classes.

    Attributes:
        acqf_class: The acquisition function class to test.
        required_kwargs: Dict of required constructor arguments.
        requires_X_observed: If True, pass X_observed (model training inputs)
            to acquisition function constructor. Used by some analytic AFs
            for posterior mean centering.
        requires_fixed_noise: If True, the acquisition function requires a model
            with fixed/known observation noise (FixedNoiseGaussianLikelihood).
        convert_tensor_kwargs: If True, convert tensor kwargs to the test's
            dtype and device. Defaults to True.
        bypass_tests: List of test names to skip for this acquisition function.
            Defaults to empty list (run all tests).
        q: Number of points in q-batch for test inputs. Defaults to 1.
    """

    acqf_class: type[AcquisitionFunction]
    required_kwargs: dict[str, Any] = field(default_factory=dict)
    requires_X_observed: bool = False
    requires_fixed_noise: bool = False
    convert_tensor_kwargs: bool = True
    bypass_tests: list[str] = field(default_factory=list)
    q: int = 1

    def get_kwargs(self, dtype: torch.dtype, device: torch.device) -> dict[str, Any]:
        """Get required_kwargs with tensors converted to the specified dtype/device.

        Args:
            dtype: The target dtype for tensor conversion.
            device: The target device for tensor conversion.

        Returns:
            A copy of required_kwargs with all Tensor values converted to the
            specified dtype and device if convert_tensor_kwargs is True.
        """
        if not self.convert_tensor_kwargs:
            return dict(self.required_kwargs)
        kwargs = {}
        for key, value in self.required_kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs[key] = value.to(dtype=dtype, device=device)
            else:
                kwargs[key] = value
        return kwargs


class AcquisitionTestMixin:
    """Mixin providing standard tests for acquisition functions.

    Subclasses should override `acquisition_specs` to return a list of
    AcquisitionSpec instances defining which acquisition functions to test.
    """

    @property
    @abstractmethod
    def acquisition_specs(self) -> list[AcquisitionSpec]:
        """Return the list of AcquisitionSpec instances to test."""
        ...

    def _make_model(
        self,
        spec: AcquisitionSpec,
        dtype: torch.dtype,
        m: int = 1,
    ):
        """Create a model for testing.

        Args:
            spec: The acquisition spec defining the test configuration.
            dtype: The dtype for the model tensors.
            m: The number of outputs. Defaults to 1.

        Returns:
            A SingleTaskGP with random training data.
        """
        torch.manual_seed(0)
        train_X = torch.rand(5, 2, dtype=dtype, device=self.device)
        train_Y = torch.rand(5, m, dtype=dtype, device=self.device)
        train_Yvar = (
            torch.full_like(train_Y, 0.25) if spec.requires_fixed_noise else None
        )
        return get_model(train_X, train_Y, train_Yvar=train_Yvar)

    def _make_acquisition(
        self,
        spec: AcquisitionSpec,
        model,
        dtype: torch.dtype,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        """Create an acquisition function for testing.

        Args:
            spec: The acquisition spec defining the test configuration.
            model: The model to use for the acquisition function.
            dtype: The dtype for tensors.
            extra_kwargs: Additional kwargs to pass to the acquisition function,
                merged after spec kwargs and X_observed handling.

        Returns:
            An instance of the acquisition function specified by the spec.
        """
        kwargs = spec.get_kwargs(dtype=dtype, device=self.device)
        if spec.requires_X_observed:
            kwargs["X_observed"] = model.train_inputs[0]
        if extra_kwargs is not None:
            kwargs.update(extra_kwargs)
        return spec.acqf_class(model=model, **kwargs)

    @loop_filtered_specs
    def test_dtype(self, spec: AcquisitionSpec) -> None:
        """Test acquisition function with different dtypes."""
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                model = self._make_model(spec=spec, dtype=dtype)
                acqf = self._make_acquisition(spec=spec, model=model, dtype=dtype)
                torch.manual_seed(1)
                X = torch.rand(4, spec.q, 2, dtype=dtype, device=self.device)
                value = acqf(X)
                self.assertEqual(value.dtype, dtype)
                self.assertEqual(value.device.type, self.device.type)

    @loop_filtered_specs
    def test_output_shapes(self, spec: AcquisitionSpec) -> None:
        """Test acquisition function with different batch shapes."""
        model = self._make_model(spec=spec, dtype=torch.double)
        acqf = self._make_acquisition(spec=spec, model=model, dtype=torch.double)
        for batch_shape in [[5], [5, 3]]:
            with self.subTest(batch_shape=batch_shape):
                torch.manual_seed(1)
                X = torch.rand(*batch_shape, spec.q, 2, device=self.device)
                value = acqf(X)
                expected_shape = torch.Size(batch_shape)
                self.assertEqual(value.shape, expected_shape)

    @loop_filtered_specs
    def test_fixed_noise(self, spec: AcquisitionSpec) -> None:
        """Test acquisition function requiring X_observed with fixed noise model."""
        model = self._make_model(spec=spec, dtype=torch.double)
        acqf = self._make_acquisition(spec=spec, model=model, dtype=torch.double)
        torch.manual_seed(1)
        X = torch.rand(4, spec.q, 2, device=self.device)
        value = acqf(X)
        self.assertEqual(value.shape, torch.Size([4]))


class AnalyticAcquisitionTestMixin(AcquisitionTestMixin):
    """Mixin for analytic acquisition functions.

    Uses spec.q for q-batch size (defaults to 1 for most analytic AFs).
    """

    @loop_filtered_specs
    def test_maximize(self, spec: AcquisitionSpec) -> None:
        """Test that maximize flag correctly identifies best/worst points."""
        # Create controlled posterior: 5 points with different means, equal variance
        # Shape [5, 1, 1] for batch=5, q=1, m=1 to match X shape [5, 1, 2]
        means = torch.tensor(
            [[[0.1]], [[0.5]], [[0.9]], [[0.3]], [[0.2]]],
            device=self.device,
            dtype=torch.double,
        )
        variance = torch.full_like(means, 0.1)

        # Create mock model that returns our controlled posterior
        mock_posterior = MockPosterior(mean=means, variance=variance)
        mock_model = MockModel(posterior=mock_posterior)

        # X shape: (5, 1, 2) - 5 candidates, q=1, d=2
        X = torch.rand(5, 1, 2, device=self.device, dtype=torch.double)

        kwargs = spec.get_kwargs(dtype=torch.double, device=self.device)
        if spec.requires_X_observed:
            kwargs["X_observed"] = mock_model.train_inputs[0]

        acqf_max = spec.acqf_class(
            model=mock_model,
            **kwargs,
            maximize=True,
        )
        acqf_min = spec.acqf_class(
            model=mock_model,
            **kwargs,
            maximize=False,
        )

        value_max = acqf_max(X)
        value_min = acqf_min(X)

        # Point with max mean (index 2) should be most preferred under maximize=True
        # and least preferred under maximize=False
        max_mean_idx = means.argmax()
        self.assertEqual(value_max.argmax(), max_mean_idx)
        self.assertEqual(value_min.argmin(), max_mean_idx)


@dataclass
class MCAcquisitionSpec(AcquisitionSpec):
    """Spec for Monte Carlo acquisition functions.

    Attributes:
        requires_X_baseline: If True, pass X_baseline (model training inputs)
            to the acquisition function constructor (e.g. qNEI, qLogNEI).
        q: Number of candidates for MC tests. Defaults to 2.
    """

    requires_X_baseline: bool = False
    q: int = 2


class MCAcquisitionTestMixin(AcquisitionTestMixin):
    """Mixin for MC acquisition functions.

    Inherits test_dtype and test_output_shapes from AcquisitionTestMixin.
    Overrides _make_acquisition for MC-specific kwargs (X_baseline).
    Adds test_q_greater_than_one to verify q-reduction.
    """

    def _make_acquisition(
        self,
        spec: MCAcquisitionSpec,
        model,
        dtype: torch.dtype,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        kwargs = spec.get_kwargs(dtype=dtype, device=self.device)
        if spec.requires_X_baseline:
            kwargs["X_baseline"] = model.train_inputs[0]
        if extra_kwargs is not None:
            kwargs.update(extra_kwargs)
        return spec.acqf_class(model=model, **kwargs)

    @loop_filtered_specs
    def test_q_greater_than_one(self, spec: MCAcquisitionSpec) -> None:
        """Test that MC acqfs correctly reduce the q dimension."""
        model = self._make_model(spec=spec, dtype=torch.double)
        acqf = self._make_acquisition(spec=spec, model=model, dtype=torch.double)
        for batch_shape in [[4], [3, 2]]:
            with self.subTest(batch_shape=batch_shape):
                torch.manual_seed(1)
                q = max(spec.q, 2)  # Ensure q > 1 for this test
                X = torch.rand(*batch_shape, q, 2, device=self.device)
                value = acqf(X)
                self.assertEqual(value.shape, torch.Size(batch_shape))

    @loop_filtered_specs
    def test_X_pending(self, spec: MCAcquisitionSpec) -> None:
        """Test X_pending set/clear behavior for MC acquisition functions."""
        model = self._make_model(spec=spec, dtype=torch.double)
        acqf = self._make_acquisition(spec=spec, model=model, dtype=torch.double)

        acqf.set_X_pending()
        self.assertIsNone(acqf.X_pending)

        acqf.set_X_pending(None)
        self.assertIsNone(acqf.X_pending)

        torch.manual_seed(1)
        X_pending = torch.rand(spec.q, 2, device=self.device)
        acqf.set_X_pending(X_pending)
        self.assertEqual(acqf.X_pending.shape, X_pending.shape)

        torch.manual_seed(2)
        X = torch.rand(2, spec.q, 2, device=self.device)
        value = acqf(X)
        self.assertEqual(value.shape, torch.Size([2]))

        # Verify BotorchWarning is raised when X_pending has requires_grad=True
        torch.manual_seed(3)
        X_grad = torch.rand(1, 2, device=self.device, requires_grad=True)
        with warnings.catch_warnings(record=True) as ws:
            acqf.set_X_pending(X_grad)
        self.assertTrue(any(issubclass(w.category, BotorchWarning) for w in ws))

    @loop_filtered_specs
    def test_sampler_base_samples(self, spec: MCAcquisitionSpec) -> None:
        """Test that sampler base_samples are stable across forward calls."""
        model = self._make_model(spec=spec, dtype=torch.double)
        for sampler_cls in [IIDNormalSampler, SobolQMCNormalSampler]:
            with self.subTest(sampler=sampler_cls.__name__):
                sampler = sampler_cls(sample_shape=torch.Size([4]), seed=12345)
                acqf = self._make_acquisition(
                    spec=spec,
                    model=model,
                    dtype=torch.double,
                    extra_kwargs={"sampler": sampler},
                )
                torch.manual_seed(1)
                X = torch.rand(1, 2, device=self.device)

                acqf(X)
                bs = acqf.sampler.base_samples.clone()
                acqf(X)
                self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))
