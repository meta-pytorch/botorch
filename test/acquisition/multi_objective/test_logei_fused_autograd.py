#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import botorch.acquisition.multi_objective.logei as logei_module
import torch
from botorch.acquisition.multi_objective.logei import (
    _FusedLogAreas,
    _try_load_fused_kernel,
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models import SingleTaskGP
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior

_try_load_fused_kernel()


@unittest.skipIf(logei_module._C is None, "C++ extension not available")
class TestFusedLogAreasAutograd(BotorchTestCase):
    tau_relu = 0.15
    tau_max = 0.2

    def _inputs(self, batch_shape, m, batched_bounds, num_subsets=2, num_cells=2):
        generator = torch.Generator().manual_seed(721 + m)

        def tensor(shape, offset, scale):
            values = offset + scale * torch.rand(
                *shape, 2 * m, generator=generator, dtype=torch.double
            )
            return values[..., ::2].detach().requires_grad_(True)

        obj = tensor((*batch_shape, num_subsets, 2), 0.6, 1.5)
        cell_batch = batch_shape if batched_bounds else ()
        lower = tensor((*cell_batch, num_cells), -0.2, 0.6)
        upper = tensor((*cell_batch, num_cells), 1.5, 0.6)
        self.assertFalse(obj.is_contiguous())
        self.assertFalse(lower.is_contiguous())
        self.assertFalse(upper.is_contiguous())
        return obj, lower, upper

    def _reference_areas(self, obj, lower, upper):
        # Exercise the existing, unfused acquisition steps rather than another
        # implementation of the fused kernel's derivatives.
        m = obj.shape[-1]
        partitioning = NondominatedPartitioning(
            ref_point=torch.zeros(m, dtype=obj.dtype),
            Y=torch.empty(0, m, dtype=obj.dtype),
        )
        acq = qLogExpectedHypervolumeImprovement(
            model=MockModel(MockPosterior(samples=obj[..., 0, :, :])),
            ref_point=torch.zeros(m, dtype=obj.dtype),
            partitioning=partitioning,
            tau_relu=self.tau_relu,
            tau_max=self.tau_max,
        )
        acq.cell_lower_bounds = lower
        acq.cell_upper_bounds = upper
        view_shape = (
            *(1,) * (obj.ndim - lower.ndim - 1),
            *lower.shape[:-2],
            lower.shape[-2],
            1,
            m,
        )
        improvement = acq._log_improvement(obj, view_shape)
        improvement = acq._smooth_min(improvement, dim=-2)
        return acq._log_cell_lengths(improvement, view_shape).sum(dim=-1)

    def _fused_areas(self, obj, lower, upper):
        return _FusedLogAreas.apply(obj, lower, upper, self.tau_relu, self.tau_max)

    def test_first_derivatives_include_cell_bounds(self):
        for requires_grad in ((True, False, False), (False, True, True), (True,) * 3):
            with self.subTest(requires_grad=requires_grad):
                inputs = self._inputs((2, 3), 3, batched_bounds=True)
                for value, needed in zip(inputs, requires_grad):
                    value.requires_grad_(needed)
                needed = tuple(value for value in inputs if value.requires_grad)
                actual = self._fused_areas(*inputs)
                expected = self._reference_areas(*inputs)
                self.assertAllClose(actual, expected, atol=1e-11, rtol=1e-11)
                gradients = torch.autograd.grad(actual.square().sum(), needed)
                reference = torch.autograd.grad(expected.square().sum(), needed)
                for gradient, target in zip(gradients, reference):
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertAllClose(gradient, target, atol=1e-9, rtol=1e-9)

    def test_mixed_second_derivatives_and_noncontiguous_bounds(self):
        for batch_shape, m, batched_bounds in (
            ((), 2, False),
            ((2, 3), 3, False),
            ((2, 3), 3, True),
            ((2,), 8, True),
        ):
            with self.subTest(batch=batch_shape, m=m, batched_bounds=batched_bounds):
                inputs = self._inputs(batch_shape, m, batched_bounds)
                derivatives = []
                for evaluate in (self._fused_areas, self._reference_areas):
                    value = evaluate(*inputs)
                    gradients = torch.autograd.grad(
                        value.square().sum(), inputs, create_graph=True
                    )
                    probe = sum(
                        (
                            gradient
                            * torch.linspace(
                                -0.3,
                                0.7,
                                gradient.numel(),
                                dtype=gradient.dtype,
                            ).reshape(gradient.shape)
                        ).sum()
                        for gradient in gradients
                    )
                    hvp = torch.autograd.grad(probe, inputs)
                    derivatives.append((gradients, hvp))
                for actual, expected in zip(*derivatives):
                    for value, target in zip(actual, expected):
                        self.assertTrue(torch.isfinite(value).all())
                        self.assertAllClose(value, target, atol=1e-8, rtol=1e-8)

    def test_gradcheck_and_gradgradcheck(self):
        for batch_shape, batched_bounds in (((), False), ((2,), True)):
            with self.subTest(batch=batch_shape, batched_bounds=batched_bounds):
                inputs = self._inputs(
                    batch_shape, 2, batched_bounds, num_subsets=1, num_cells=1
                )
                self.assertTrue(torch.autograd.gradcheck(self._fused_areas, inputs))
                self.assertTrue(torch.autograd.gradgradcheck(self._fused_areas, inputs))

    def test_sample_gradgradcheck_with_fixed_bounds(self):
        obj, lower, upper = self._inputs((), 2, False, num_subsets=1, num_cells=1)
        lower = lower.detach()
        upper = upper.detach()

        def evaluate(samples):
            return self._fused_areas(samples, lower, upper)

        # Isolate double backward from differentiating the cell bounds. The
        # first-order-only C++ backward passes gradcheck but fails gradgradcheck.
        self.assertTrue(torch.autograd.gradcheck(evaluate, (obj,)))
        self.assertTrue(torch.autograd.gradgradcheck(evaluate, (obj,)))

    def test_dependent_arguments_follow_chain_rule(self):
        for create_graph in (False, True):
            with self.subTest(create_graph=create_graph):
                results = []
                for evaluate in (self._fused_areas, self._reference_areas):
                    obj = torch.tensor(
                        [[[0.8, 1.2], [1.1, 0.9]]],
                        dtype=torch.double,
                        requires_grad=True,
                    )
                    lower = 0.1 * obj[0, 0].unsqueeze(0)
                    upper = lower + 1.3
                    value = evaluate(obj, lower, upper)
                    gradient = torch.autograd.grad(
                        value.square().sum(), obj, create_graph=create_graph
                    )[0]
                    result = [value, gradient]
                    if create_graph:
                        result.append(torch.autograd.grad(gradient.sum(), obj)[0])
                    results.append(result)
                for actual, expected in zip(*results):
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertAllClose(actual, expected, atol=1e-9, rtol=1e-9)

    def test_public_acquisitions_bound_gradients_on_repeated_forwards(self):
        generator = torch.Generator().manual_seed(45)
        train_x = torch.rand(5, 2, generator=generator, dtype=torch.double)
        train_y = torch.cat((train_x[:, :1].sin(), train_x[:, 1:].cos()), dim=-1)
        model = SingleTaskGP(train_x, train_y).eval()
        ref_point = train_y.min(dim=0).values - 3.0
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_y)
        points = torch.rand(2, 2, 2, generator=generator, dtype=torch.double)
        for noisy in (False, True):
            with self.subTest(noisy=noisy):
                kwargs = {
                    "model": model,
                    "ref_point": ref_point,
                    "sampler": IIDNormalSampler(torch.Size([2]), seed=7),
                    "tau_relu": self.tau_relu,
                    "tau_max": self.tau_max,
                }
                if noisy:
                    acq = qLogNoisyExpectedHypervolumeImprovement(
                        **kwargs,
                        X_baseline=train_x[:1],
                        prune_baseline=False,
                        cache_root=False,
                    )
                else:
                    acq = qLogExpectedHypervolumeImprovement(
                        **kwargs, partitioning=partitioning
                    )
                acq.cell_lower_bounds = (
                    acq.cell_lower_bounds.detach().clone().requires_grad_(True)
                )
                acq.cell_upper_bounds = (
                    acq.cell_upper_bounds.detach().clone().requires_grad_(True)
                )
                # Fresh forwards must work without retaining any previous graph.
                for _ in range(2):
                    query = points.clone().requires_grad_(True)
                    inputs = (query, acq.cell_lower_bounds, acq.cell_upper_bounds)
                    with mock.patch.object(
                        _FusedLogAreas, "apply", wraps=_FusedLogAreas.apply
                    ) as fused:
                        actual = acq(query)
                    self.assertGreater(fused.call_count, 0)
                    gradients = torch.autograd.grad(actual.sum(), inputs)
                    with mock.patch.object(logei_module, "_C", None):
                        expected = acq(query)
                    reference = torch.autograd.grad(expected.sum(), inputs)
                    self.assertAllClose(actual, expected, atol=1e-9, rtol=1e-9)
                    for gradient, target in zip(gradients, reference):
                        self.assertTrue(torch.isfinite(gradient).all())
                        self.assertAllClose(gradient, target, atol=1e-8, rtol=1e-8)
