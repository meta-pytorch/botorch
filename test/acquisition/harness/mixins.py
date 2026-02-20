#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Base specs and test mixins for acquisition function testing."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions import BotorchWarning
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

from .factories import make_trained_gp, make_X


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
            with self.subTest(cls=spec.cls.__name__):
                test_method(self, spec)

    return wrapper


@dataclass
class AcquisitionSpec:
    """Base spec for single-output acquisition functions requiring only model
    and standard keyword arguments (e.g., best_f, beta).

    For acquisition functions with special requirements like X_baseline or
    constraints, use specialized spec classes.

    Attributes:
        cls: The acquisition function class to test
        required_kwargs: Dict of required constructor arguments
        requires_X_observed: If True, pass X_observed (model training inputs)
            to acquisition function constructor.
        requires_fixed_noise: If True, the acquisition function requires a model
            with fixed/known observation noise (FixedNoiseGaussianLikelihood).
        convert_tensor_kwargs: If True, convert tensor kwargs to the test's
            dtype and device. Defaults to True.
        bypass_tests: List of test names to skip for this acquisition function.
            Defaults to empty list (run all tests).
    """

    cls: type[AcquisitionFunction]
    required_kwargs: dict[str, Any] = field(default_factory=dict)
    requires_X_observed: bool = False
    requires_fixed_noise: bool = False
    convert_tensor_kwargs: bool = True
    bypass_tests: list[str] = field(default_factory=list)
    q_dim: int = 1  # Number of points in q-batch for test inputs

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
    def acquisition_specs(self) -> list[AcquisitionSpec]:
        """Return the list of AcquisitionSpec instances to test."""
        return []

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
        return make_trained_gp(
            n_train=5,
            d=2,
            m=m,
            dtype=dtype,
            device=self.device,
            with_known_noise=spec.requires_fixed_noise,
        )

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
        return spec.cls(model=model, **kwargs)

    @loop_filtered_specs
    def test_dtype(self, spec: AcquisitionSpec) -> None:
        """Test acquisition function with different dtypes."""
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                model = self._make_model(spec=spec, dtype=dtype)
                acqf = self._make_acquisition(spec=spec, model=model, dtype=dtype)
                X = make_X(
                    batch_shape=[4], q=spec.q_dim, dtype=dtype, device=self.device
                )
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
                X = make_X(batch_shape=batch_shape, q=spec.q_dim, device=self.device)
                value = acqf(X)
                expected_shape = torch.Size(batch_shape)
                self.assertEqual(value.shape, expected_shape)

    @loop_filtered_specs
    def test_fixed_noise(self, spec: AcquisitionSpec) -> None:
        """Test acquisition function requiring X_observed with fixed noise model."""
        model = self._make_model(spec=spec, dtype=torch.double)
        acqf = self._make_acquisition(spec=spec, model=model, dtype=torch.double)
        X = make_X(batch_shape=[4], q=spec.q_dim, device=self.device)
        value = acqf(X)
        self.assertEqual(value.shape, torch.Size([4]))


class AnalyticAcquisitionTestMixin(AcquisitionTestMixin):
    """Mixin for analytic acquisition functions.

    Inherits dtype/device and batch shape tests from AcquisitionTestMixin.
    """

    pass


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
    ):
        kwargs = spec.get_kwargs(dtype=dtype, device=self.device)
        if spec.requires_X_baseline:
            kwargs["X_baseline"] = model.train_inputs[0]
        return spec.cls(model=model, **kwargs)

    @loop_filtered_specs
    def test_q_greater_than_one(self, spec: MCAcquisitionSpec) -> None:
        """Test that MC acqfs correctly reduce the q dimension."""
        model = self._make_model(spec=spec, dtype=torch.double)
        acqf = self._make_acquisition(spec=spec, model=model, dtype=torch.double)
        for batch_shape in [[4], [3, 2]]:
            with self.subTest(batch_shape=batch_shape):
                X = make_X(
                    batch_shape=batch_shape,
                    q=3,
                    device=self.device,
                )
                value = acqf(X)
                self.assertEqual(value.shape, torch.Size(batch_shape))

    @loop_filtered_specs
    def test_X_pending(self, spec: MCAcquisitionSpec) -> None:
        """Test X_pending set/clear behavior for MC acquisition functions."""
        model = self._make_model(spec=spec, dtype=torch.double)
        acqf = self._make_acquisition(spec=spec, model=model, dtype=torch.double)

        acqf.set_X_pending(None)
        self.assertIsNone(acqf.X_pending)

        X_pending = make_X(q=2, device=self.device)
        acqf.set_X_pending(X_pending)
        self.assertEqual(acqf.X_pending.shape, X_pending.shape)

        X = make_X(batch_shape=[2], q=1, device=self.device)
        value = acqf(X)
        self.assertEqual(value.shape, torch.Size([2]))

        X_grad = make_X(q=1, device=self.device).requires_grad_(True)
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
                kwargs = spec.get_kwargs(dtype=torch.double, device=self.device)
                if spec.requires_X_baseline:
                    kwargs["X_baseline"] = model.train_inputs[0]
                acqf = spec.cls(model=model, sampler=sampler, **kwargs)
                X = make_X(q=1, device=self.device)

                acqf(X)
                bs = acqf.sampler.base_samples.clone()
                acqf(X)
                self.assertTrue(torch.equal(acqf.sampler.base_samples, bs))
