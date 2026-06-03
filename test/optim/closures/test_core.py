#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.optim.closures import get_loss_closure
from botorch.optim.closures.core import (
    BatchedNDarrayOptimizationClosure,
    FILL_VALUE,
    ForwardBackwardClosure,
    NdarrayOptimizationClosure,
)
from botorch.optim.utils import as_ndarray
from botorch.utils.testing import BotorchTestCase
from gpytorch.mlls import ExactMarginalLogLikelihood
from linear_operator.utils.errors import NanError, NotPSDError
from torch.nn import Module, Parameter


class ToyModule(Module):
    def __init__(self, w: Parameter, b: Parameter, x: Parameter, dummy: Parameter):
        r"""Toy module for unit testing."""
        super().__init__()
        self.w = w
        self.b = b
        self.x = x
        self.dummy = dummy

    def forward(self) -> torch.Tensor:
        return self.w.sum() * self.x + self.b

    @property
    def free_parameters(self) -> dict[str, torch.Tensor]:
        return {n: p for n, p in self.named_parameters() if p.requires_grad}


class TestForwardBackwardClosure(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        module = ToyModule(
            w=Parameter(torch.tensor(2.0)),
            b=Parameter(torch.tensor(3.0), requires_grad=False),
            x=Parameter(torch.tensor(4.0)),
            dummy=Parameter(torch.tensor(5.0)),
        ).to(self.device)
        self.modules = {}
        for dtype in ("float32", "float64"):
            self.modules[dtype] = module.to(dtype=getattr(torch, dtype))

    def test_main(self):
        for module in self.modules.values():
            closure = ForwardBackwardClosure(module, module.free_parameters)

            # Test __init__
            closure = ForwardBackwardClosure(module, module.free_parameters)
            self.assertEqual(module.free_parameters, closure.parameters)

            # Test return values
            value, (dw, dx, dd) = closure()
            self.assertTrue(value.equal(module()))
            self.assertTrue(dw.equal(module.x))
            self.assertTrue(dx.equal(module.w))
            self.assertEqual(dd, None)


class TestNdarrayOptimizationClosure(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings=suppress_input_warnings)
        self.module = ToyModule(
            w=Parameter(torch.tensor(2.0)),
            b=Parameter(torch.tensor(3.0), requires_grad=False),
            x=Parameter(torch.tensor(4.0)),
            dummy=Parameter(torch.tensor(5.0)),
        ).to(self.device)

        self.module_non_scalar = ToyModule(
            w=Parameter(torch.full((2,), 2.0)),
            b=Parameter(torch.tensor(3.0), requires_grad=False),
            x=Parameter(torch.tensor(4.0)),
            dummy=Parameter(torch.tensor(5.0)),
        ).to(self.device)

        self.wrappers = {}
        for dtype in ("float32", "float64"):
            for module, name in (
                (self.module, ""),
                (self.module_non_scalar, "non_scalar"),
            ):
                module = module.to(dtype=getattr(torch, dtype))
                closure = ForwardBackwardClosure(module, module.free_parameters)
                wrapper = NdarrayOptimizationClosure(closure, closure.parameters)
                self.wrappers[f"{name}_dtype"] = wrapper

    def test_main(self):
        for wrapper in self.wrappers.values():
            # Test setter/getter
            state = wrapper.state
            other = np.random.randn(*state.shape).astype(state.dtype)

            wrapper.state = other
            self.assertTrue(np.allclose(other, wrapper.state))
            n = 0
            for param in wrapper.parameters.values():
                k = param.numel()
                # Check that parameters are set correctly when setting state
                self.assertTrue(
                    np.allclose(other[n : n + k], param.view(-1).detach().cpu().numpy())
                )
                # Check getting state
                self.assertTrue(
                    np.allclose(
                        wrapper.state[n : n + k], param.view(-1).detach().cpu().numpy()
                    )
                )
                n += k

            index = 0
            for param in wrapper.closure.parameters.values():
                size = param.numel()
                self.assertTrue(
                    np.allclose(other[index : index + size], as_ndarray(param.view(-1)))
                )
                index += size

            # Test that getting and setting state work as expected
            wrapper.state = state
            self.assertTrue(np.allclose(state, wrapper.state))

            # Test __call__
            value, grads = wrapper(other)
            self.assertTrue(np.allclose(other, wrapper.state))
            self.assertIsInstance(value, np.ndarray)
            self.assertIsInstance(grads, np.ndarray)

            # Test return values
            value_tensor, grad_tensors = wrapper.closure()  # get raw Tensor equivalents
            self.assertTrue(np.allclose(value, as_ndarray(value_tensor)))
            index = 0
            for x, dx in zip(wrapper.parameters.values(), grad_tensors, strict=True):
                size = x.numel()
                grad = grads[index : index + size]
                if dx is None:
                    self.assertTrue((grad == FILL_VALUE).all())
                else:
                    self.assertTrue(np.allclose(grad, as_ndarray(dx)))
                index += size

            module = wrapper.closure.forward
            # The forward function is w.sum() * x + b, and there is no grad for b,
            # so the gradients are
            # x, w.sum(), FILL_VALUE
            n_w = module.w.numel()
            self.assertTrue(np.allclose(grads[:n_w], as_ndarray(module.x)))
            self.assertTrue(np.allclose(grads[n_w], as_ndarray(module.w.sum())))
            self.assertEqual(grads[n_w + 1], FILL_VALUE)

            # Test persistence
            self.assertIs(
                wrapper._get_gradient_ndarray(), wrapper._get_gradient_ndarray()
            )

    def test_exceptions(self):
        for wrapper in self.wrappers.values():
            mock_closure = MagicMock(return_value=wrapper.closure())
            mock_wrapper = NdarrayOptimizationClosure(
                mock_closure, wrapper.closure.parameters
            )
            with self.assertRaisesRegex(NotPSDError, "foo"):
                mock_wrapper.closure.side_effect = NotPSDError("foo")
                mock_wrapper()

            for exception in (
                NanError("foo"),
                RuntimeError("singular"),
                RuntimeError("input is not positive-definite"),
            ):
                mock_wrapper.closure.side_effect = exception
                value, grads = mock_wrapper()
                self.assertTrue(np.isnan(value).all())
                self.assertTrue(np.isnan(grads).all())

        with self.subTest("No-parameters exception"):
            wrapper = NdarrayOptimizationClosure(
                closure=lambda: torch.tensor(), parameters={}
            )
            with self.assertRaisesRegex(RuntimeError, "No parameters"):
                wrapper.state


class TestBatchedNDarrayOptimizationClosure(BotorchTestCase):
    def _make_closure(
        self, m: int = 2, d: int = 2, n: int = 5
    ) -> tuple[BatchedNDarrayOptimizationClosure, SingleTaskGP]:
        train_X = torch.rand(n, d, dtype=torch.double)
        train_Y = torch.rand(n, m, dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        parameters = {name: p for name, p in mll.named_parameters() if p.requires_grad}
        forward = get_loss_closure(mll)
        closure = BatchedNDarrayOptimizationClosure(
            forward=forward,
            parameters=parameters,
            batch_shape=model._aug_batch_shape,
        )
        return closure, model

    def test_state_roundtrip(self):
        closure, model = self._make_closure(m=2, d=2)

        state = closure.state
        self.assertEqual(state.ndim, 2)
        self.assertEqual(state.shape[0], 2)

        new_state = np.random.randn(*state.shape)
        closure.state = new_state
        np.testing.assert_array_almost_equal(closure.state, new_state)

    def test_call_shapes(self):
        closure, model = self._make_closure(m=2, d=2)

        values, grads = closure()
        self.assertEqual(values.shape, (2,))
        self.assertEqual(grads.shape, (2, closure._per_element_size))

    def test_call_with_batch_indices(self):
        closure, model = self._make_closure(m=3, d=2)

        all_values, all_grads = closure()
        self.assertEqual(all_values.shape, (3,))

        indices = np.array([0, 2])
        state = closure.state[indices]
        subset_values, subset_grads = closure(state=state, batch_indices=indices)
        self.assertEqual(subset_values.shape, (2,))
        self.assertEqual(subset_grads.shape, (2, closure._per_element_size))
        np.testing.assert_array_almost_equal(subset_values, all_values[indices])

    def test_gradients_are_correct(self):
        closure, model = self._make_closure(m=2, d=2)

        state = closure.state
        values, grads = closure(state=state)

        eps = 1e-6
        for b in range(closure.batch_size):
            for j in range(closure._per_element_size):
                state_plus = state.copy()
                state_plus[b, j] += eps
                vals_plus, _ = closure(state=state_plus)

                state_minus = state.copy()
                state_minus[b, j] -= eps
                vals_minus, _ = closure(state=state_minus)

                fd_grad = (vals_plus[b] - vals_minus[b]) / (2 * eps)
                self.assertAlmostEqual(
                    grads[b, j],
                    fd_grad,
                    places=4,
                    msg=f"Gradient mismatch at batch={b}, param={j}",
                )

    def test_runtime_error_handling(self):
        closure, model = self._make_closure(m=2, d=2)

        # Replace forward with one that raises RuntimeError
        def failing_forward(**kwargs):
            raise RuntimeError("singular")

        closure.forward = failing_forward

        values, grads = closure()
        # _handle_numerical_errors returns NaN for singular errors
        self.assertTrue(np.isnan(values).all())
        self.assertEqual(values.shape, (2,))
        # Grads should be zeros from the fallback path
        np.testing.assert_array_equal(grads, 0.0)
        self.assertEqual(grads.shape, (2, closure._per_element_size))

    def test_runtime_error_with_batch_indices(self):
        closure, model = self._make_closure(m=3, d=2)

        def failing_forward(**kwargs):
            raise RuntimeError("singular")

        closure.forward = failing_forward

        indices = np.array([0, 2])
        values, grads = closure(batch_indices=indices)
        # Should return filtered results for the requested batch indices
        self.assertTrue(np.isnan(values).all())
        self.assertEqual(values.shape, (2,))
        np.testing.assert_array_equal(grads, 0.0)
        self.assertEqual(grads.shape, (2, closure._per_element_size))
