#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest.mock import patch

import torch
from botorch.models import SingleTaskGP, SingleTaskVariationalGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.pathwise.utils import (
    get_input_transform,
    get_output_transform,
    get_train_inputs,
    get_train_targets,
)
from botorch.sampling.pathwise.utils.transforms import (
    InverseLengthscaleTransform,
    OutcomeUntransformer,
)
from botorch.utils.context_managers import delattr_ctx
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels import MaternKernel, ScaleKernel


class TestTransforms(BotorchTestCase):
    def test_inverse_lengthscale_transform(self):
        tkwargs = {"device": self.device, "dtype": torch.float64}
        kernel = MaternKernel(nu=2.5, ard_num_dims=3).to(**tkwargs)
        with self.assertRaisesRegex(RuntimeError, "does not implement `lengthscale`"):
            InverseLengthscaleTransform(ScaleKernel(kernel))

        x = torch.rand(3, 3, **tkwargs)
        transform = InverseLengthscaleTransform(kernel)
        self.assertTrue(transform(x).equal(kernel.lengthscale.reciprocal() * x))

    def test_outcome_untransformer(self):
        for untransformer in (
            OutcomeUntransformer(transform=Standardize(m=1), num_outputs=1),
            OutcomeUntransformer(transform=Standardize(m=2), num_outputs=2),
        ):
            with torch.random.fork_rng():
                torch.random.manual_seed(0)
                y = torch.rand(untransformer.num_outputs, 4, device=self.device)
            x = untransformer.transform(y.T)[0].T
            self.assertTrue(y.allclose(untransformer(x)))


class TestGetters(BotorchTestCase):
    def setUp(self):
        super().setUp()
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            train_X = torch.rand(5, 2)
            train_Y = torch.randn(5, 2)

        self.models = []
        for num_outputs in (1, 2):
            self.models.append(
                SingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y[:, :num_outputs],
                    input_transform=Normalize(d=2),
                )
            )

            self.models.append(
                SingleTaskVariationalGP(
                    train_X=train_X,
                    train_Y=train_Y[:, :num_outputs],
                    input_transform=Normalize(d=2),
                    outcome_transform=Standardize(m=num_outputs),
                )
            )

    def test_get_input_transform(self):
        for model in self.models:
            self.assertIs(get_input_transform(model), model.input_transform)

    def test_get_output_transform(self):
        for model in self.models:
            transform = get_output_transform(model)
            self.assertIsInstance(transform, OutcomeUntransformer)
            self.assertIs(transform.transform, model.outcome_transform)

    def test_get_train_inputs(self):
        for model in self.models:
            model.train()
            X = (
                model.model.train_inputs[0]
                if isinstance(model, SingleTaskVariationalGP)
                else model.train_inputs[0]
            )
            Z = model.input_transform(X)
            train_inputs = get_train_inputs(model, transformed=False)
            self.assertIsInstance(train_inputs, tuple)
            self.assertEqual(len(train_inputs), 1)

            self.assertTrue(X.equal(get_train_inputs(model, transformed=False)[0]))
            self.assertTrue(Z.equal(get_train_inputs(model, transformed=True)[0]))

            model.eval()
            self.assertTrue(X.equal(get_train_inputs(model, transformed=False)[0]))
            self.assertTrue(Z.equal(get_train_inputs(model, transformed=True)[0]))
            with delattr_ctx(model, "input_transform"), patch.object(
                model, "_original_train_inputs", new=None
            ):
                self.assertTrue(Z.equal(get_train_inputs(model, transformed=False)[0]))
                self.assertTrue(Z.equal(get_train_inputs(model, transformed=True)[0]))

        with self.subTest("test_model_list"):
            model_list = ModelListGP(*self.models)
            input_list = get_train_inputs(model_list)
            self.assertIsInstance(input_list, list)
            self.assertEqual(len(input_list), len(self.models))
            for model, train_inputs in zip(model_list.models, input_list):
                for a, b in zip(train_inputs, get_train_inputs(model)):
                    self.assertTrue(a.equal(b))

    def test_get_train_targets(self):
        for model in self.models:
            model.train()
            if isinstance(model, SingleTaskVariationalGP):
                F = model.model.train_targets
                Y = model.outcome_transform.untransform(F)[0].squeeze(dim=0)
            else:
                F = model.train_targets
                Y = OutcomeUntransformer(model.outcome_transform, model.num_outputs)(F)

            self.assertTrue(F.equal(get_train_targets(model, transformed=True)))
            self.assertTrue(Y.equal(get_train_targets(model, transformed=False)))

            model.eval()
            self.assertTrue(F.equal(get_train_targets(model, transformed=True)))
            self.assertTrue(Y.equal(get_train_targets(model, transformed=False)))
            with delattr_ctx(model, "outcome_transform"):
                self.assertTrue(F.equal(get_train_targets(model, transformed=True)))
                self.assertTrue(F.equal(get_train_targets(model, transformed=False)))

        with self.subTest("test_model_list"):
            model_list = ModelListGP(*self.models)
            target_list = get_train_targets(model_list)
            self.assertIsInstance(target_list, list)
            self.assertEqual(len(target_list), len(self.models))
            for model, Y in zip(self.models, target_list):
                self.assertTrue(Y.equal(get_train_targets(model)))


class TestUtilsHelpers(BotorchTestCase):
    def setUp(self):
        super().setUp()
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            train_X = torch.rand(5, 2)
            train_Y = torch.randn(5, 2)

        self.models = []
        for num_outputs in (1, 2):
            self.models.append(
                SingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y[:, :num_outputs],
                    input_transform=Normalize(d=2),
                    outcome_transform=Standardize(m=num_outputs),
                )
            )

            self.models.append(
                SingleTaskVariationalGP(
                    train_X=train_X,
                    train_Y=train_Y[:, :num_outputs],
                    input_transform=Normalize(d=2),
                    outcome_transform=Standardize(m=num_outputs),
                )
            )

    def test_sparse_block_diag_with_linear_operator(self):
        """Test sparse_block_diag with LinearOperator input"""
        from botorch.sampling.pathwise.utils.helpers import sparse_block_diag
        from linear_operator.operators import DiagLinearOperator

        # Create a LinearOperator block
        diag_values = torch.tensor([1.0, 2.0, 3.0])
        linear_op_block = DiagLinearOperator(diag_values)

        # Create a regular tensor block
        tensor_block = torch.tensor([[4.0, 5.0], [6.0, 7.0]])

        # Test with LinearOperator in blocks
        blocks = [linear_op_block, tensor_block]
        result = sparse_block_diag(blocks)

        # Verify the result
        self.assertTrue(result.is_sparse)
        dense_result = result.to_dense()

        # Check that the blocks are arranged diagonally
        expected_shape = (5, 5)  # 3x3 + 2x2
        self.assertEqual(dense_result.shape, expected_shape)

    def test_untransform_shape_with_input_transform(self):
        """Test untransform_shape with InputTransform."""
        from botorch.models.transforms.input import Normalize
        from botorch.sampling.pathwise.utils.helpers import untransform_shape

        # Create an InputTransform
        transform = Normalize(d=2)

        # Create a test shape
        shape = torch.Size([10, 2])

        # Test the untransform_shape function
        result_shape = untransform_shape(transform, shape)

        # Should return the same shape since InputTransform doesn't change
        # dimensionality
        self.assertEqual(result_shape, shape)

    def test_get_kernel_num_inputs_error_case(self):
        """Test get_kernel_num_inputs error case."""
        from botorch.sampling.pathwise.utils.helpers import get_kernel_num_inputs
        from gpytorch.kernels import RBFKernel

        # Create a kernel with no active_dims or ard_num_dims
        kernel = RBFKernel()

        # Test the error case
        with self.assertRaisesRegex(ValueError, "`num_ambient_inputs` must be passed"):
            get_kernel_num_inputs(kernel, num_ambient_inputs=None)

    def test_get_train_inputs_original_train_inputs(self):
        """Test _get_train_inputs_Model with _original_train_inputs."""
        from unittest.mock import patch

        from botorch.sampling.pathwise.utils.helpers import get_train_inputs

        # Use one of the models from setUp
        model = self.models[0]

        # Create a mock _original_train_inputs
        original_X = torch.rand(5, 2)

        # Test with _original_train_inputs set and transformed=False
        with patch.object(model, "_original_train_inputs", original_X):
            result = get_train_inputs(model, transformed=False)
            self.assertTrue(result[0].equal(original_X))

    def test_get_train_targets_multitask_variational(self):
        """Test _get_train_targets_SingleTaskVariationalGP with multitask."""
        from botorch.models import SingleTaskVariationalGP
        from botorch.sampling.pathwise.utils.helpers import get_train_targets

        # Create a variational model with multiple outputs
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            train_X = torch.rand(5, 2)
            train_Y = torch.randn(5, 2)  # 2 outputs

        variational_model = SingleTaskVariationalGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=Standardize(m=2),
        )

        # This should test the multitask branch (num_outputs > 1)
        result = get_train_targets(variational_model, transformed=False)
        self.assertIsInstance(result, torch.Tensor)
        # Check that the result has the correct shape
        self.assertEqual(result.shape, train_Y.shape)

    def test_append_transform_with_existing_transform(self):
        """Test append_transform when other transform exists"""
        from botorch.models.transforms.input import Normalize
        from botorch.sampling.pathwise.utils.helpers import append_transform
        from botorch.sampling.pathwise.utils.transforms import ChainedTransform

        # Create a mock module that has TransformedModuleMixin interface
        class MockModule:
            def __init__(self):
                self.existing_transform = Normalize(d=2)

        module = MockModule()
        new_transform = Normalize(d=3)

        # This should trigger line where ChainedTransform is created
        append_transform(module, "existing_transform", new_transform)

        # Verify ChainedTransform was created
        self.assertIsInstance(module.existing_transform, ChainedTransform)
        self.assertEqual(len(module.existing_transform.transforms), 2)

    def test_untransform_shape_with_none_transform(self):
        """Test untransform_shape with None transform"""
        from botorch.sampling.pathwise.utils.helpers import untransform_shape

        shape = torch.Size([10, 2])
        result_shape = untransform_shape(None, shape)

        # Should return the same shape when transform is None
        self.assertEqual(result_shape, shape)

    def test_untransform_shape_with_untrained_outcome_transform(self):
        """Test untransform_shape with untrained OutcomeTransform"""
        from botorch.models.transforms.outcome import OutcomeTransform
        from botorch.sampling.pathwise.utils.helpers import untransform_shape

        # Create a mock OutcomeTransform that is not trained
        class MockUntrainedOutcomeTransform(OutcomeTransform):
            def __init__(self):
                super().__init__()
                self._is_trained = False

            def forward(self, Y, Yvar=None):
                return Y, Yvar

            def untransform(self, Y, Yvar=None):
                return Y, Yvar

        transform = MockUntrainedOutcomeTransform()
        shape = torch.Size([10, 2])

        result_shape = untransform_shape(transform, shape)
        # Should return the same shape when transform is not trained
        self.assertEqual(result_shape, shape)

    def test_get_kernel_num_inputs_with_default(self):
        """Test get_kernel_num_inputs with default value"""
        from botorch.sampling.pathwise.utils.helpers import get_kernel_num_inputs
        from gpytorch.kernels import RBFKernel

        # Create a kernel with no active_dims or ard_num_dims
        kernel = RBFKernel()

        # Test with default value (should return default)
        result = get_kernel_num_inputs(kernel, num_ambient_inputs=None, default=5)
        self.assertEqual(result, 5)

        # Test with num_ambient_inputs (should return num_ambient_inputs)
        result = get_kernel_num_inputs(kernel, num_ambient_inputs=3, default=5)
        self.assertEqual(result, 3)

    def test_module_dict_mixin_update(self):
        """Test ModuleDictMixin update method"""
        from botorch.sampling.pathwise.utils.mixins import ModuleDictMixin
        from torch.nn import Linear, Module

        # Create a concrete class that uses ModuleDictMixin
        class TestModuleDictClass(Module, ModuleDictMixin):
            def __init__(self):
                Module.__init__(self)
                ModuleDictMixin.__init__(self, attr_name="modules")

        test_obj = TestModuleDictClass()

        new_modules = {"linear1": Linear(2, 3), "linear2": Linear(3, 1)}
        test_obj.update(new_modules)

        # Verify modules were added
        self.assertIn("linear1", test_obj)
        self.assertIn("linear2", test_obj)
        self.assertEqual(len(test_obj), 2)

    def test_untransform_shape_edge_case(self):
        """Test untransform_shape edge case"""
        from botorch.models.transforms.outcome import OutcomeTransform
        from botorch.sampling.pathwise.utils.helpers import untransform_shape

        # Create a mock OutcomeTransform that returns different shape
        class MockShapeChangingTransform(OutcomeTransform):
            def __init__(self):
                super().__init__()
                self._is_trained = True

            def forward(self, Y, Yvar=None):
                return Y, Yvar

            def untransform(self, Y, Yvar=None):
                # Return a tensor with different shape
                return Y.repeat(1, 2), Yvar  # Double the last dimension

        transform = MockShapeChangingTransform()
        shape = torch.Size([10, 2])

        result_shape = untransform_shape(transform, shape)
        # Should return the transformed shape (doubled last dimension)
        self.assertEqual(result_shape, torch.Size([10, 4]))
