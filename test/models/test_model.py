#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import InputDataError
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model, ModelDict, ModelList
from botorch.models.transforms.input import Normalize
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.posteriors.posterior_list import PosteriorList
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from torch import rand, Tensor


class NotSoAbstractBaseModel(Model):
    def posterior(self, X, output_indices, observation_noise, **kwargs):
        pass


class ModelWithInputTransformButNoTrainInputs(Model):
    """A model that has input_transform but no train_inputs attribute."""

    def __init__(self, input_transform):
        """Initialize the model with an input transform.

        Args:
            input_transform: The input transform to apply.
        """
        super().__init__()
        self.input_transform = input_transform

    def posterior(self, X, output_indices, observation_noise, **kwargs):
        pass


class GenericDeterministicModelWithBatchShape(GenericDeterministicModel):
    # mocking torch.nn.Module components is kind of funky, so let's do this instead
    @property
    def batch_shape(self):
        return self._batch_shape


class DummyPosteriorTransform(PosteriorTransform):
    def evaluate(self, Y: Tensor, X: Tensor | None = None) -> Tensor:
        return 2 * Y + 1

    def forward(
        self, posterior: PosteriorList, X: Tensor | None = None
    ) -> PosteriorList:
        return PosteriorList(
            *[
                EnsemblePosterior(2 * p.mean.unsqueeze(0) + 1)
                for p in posterior.posteriors
            ]
        )


class TestBaseModel(BotorchTestCase):
    def test_abstract_base_model(self):
        with self.assertRaises(TypeError):
            Model()

    def test_not_so_abstract_base_model(self):
        model = NotSoAbstractBaseModel()
        with self.assertRaises(NotImplementedError):
            model.condition_on_observations(None, None)
        with self.assertRaises(NotImplementedError):
            model.num_outputs
        with self.assertRaises(NotImplementedError):
            model.batch_shape
        with self.assertRaises(NotImplementedError):
            model.subset_output([0])

    def test_set_transformed_inputs_warning_without_train_inputs(self) -> None:
        # Test that a RuntimeWarning is raised when a model has an input_transform
        # but no train_inputs attribute.
        input_transform = Normalize(d=2)
        model = ModelWithInputTransformButNoTrainInputs(input_transform=input_transform)

        # Verify the model has input_transform but no train_inputs
        self.assertTrue(hasattr(model, "input_transform"))
        self.assertFalse(hasattr(model, "train_inputs"))

        # Test cases: (method_name, callable that triggers _set_transformed_inputs)
        test_cases = [
            ("_set_transformed_inputs", lambda: model._set_transformed_inputs()),
            ("eval", lambda: model.eval()),
            ("train(mode=False)", lambda: model.train(mode=False)),
        ]

        for method_name, trigger_fn in test_cases:
            with self.subTest(method=method_name):
                with self.assertWarnsRegex(
                    RuntimeWarning,
                    "Could not update `train_inputs` with transformed inputs since "
                    "ModelWithInputTransformButNoTrainInputs does not have a "
                    "`train_inputs` attribute",
                ):
                    trigger_fn()

    def test_construct_inputs(self) -> None:
        model = NotSoAbstractBaseModel()
        with (
            self.subTest("Wrong training data type"),
            self.assertRaisesRegex(
                TypeError,
                "Expected `training_data` to be a `SupervisedDataset`, but got ",
            ),
        ):
            model.construct_inputs(training_data=None)

        x = rand(3, 2)
        y = rand(3, 1)
        dataset = SupervisedDataset(
            X=x, Y=y, feature_names=["a", "b"], outcome_names=["y"]
        )
        model_inputs = model.construct_inputs(training_data=dataset)
        self.assertEqual(model_inputs, {"train_X": x, "train_Y": y})

        yvar = rand(3, 1)
        dataset = SupervisedDataset(
            X=x, Y=y, Yvar=yvar, feature_names=["a", "b"], outcome_names=["y"]
        )
        model_inputs = model.construct_inputs(training_data=dataset)
        self.assertEqual(model_inputs, {"train_X": x, "train_Y": y, "train_Yvar": yvar})

    def test_model_list(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        m1 = GenericDeterministicModel(lambda X: X[-1:], num_outputs=1)
        m2 = GenericDeterministicModel(lambda X: X[-2:], num_outputs=2)
        model = ModelList(m1, m2)
        self.assertEqual(model.num_outputs, 3)
        # test _get_group_subset_indices
        gsi = model._get_group_subset_indices(idcs=None)
        self.assertEqual(len(gsi), 2)
        self.assertIsNone(gsi[0])
        self.assertIsNone(gsi[1])
        gsi = model._get_group_subset_indices(idcs=[0, 2])
        self.assertEqual(len(gsi), 2)
        self.assertEqual(gsi[0], [0])
        self.assertEqual(gsi[1], [1])
        # test subset_model
        m_sub = model.subset_output(idcs=[0, 1])
        self.assertIsInstance(m_sub, ModelList)
        self.assertEqual(m_sub.num_outputs, 2)
        m_sub = model.subset_output(idcs=[1, 2])
        self.assertIsInstance(m_sub, GenericDeterministicModel)
        self.assertEqual(m_sub.num_outputs, 2)
        # test posterior
        X = torch.rand(2, 2, **tkwargs)
        p = model.posterior(X=X)
        self.assertIsInstance(p, PosteriorList)
        # test batch shape
        m1 = GenericDeterministicModelWithBatchShape(lambda X: X[-1:], num_outputs=1)
        m2 = GenericDeterministicModelWithBatchShape(lambda X: X[-2:], num_outputs=2)
        model = ModelList(m1, m2)
        m1._batch_shape = torch.Size([2])
        m2._batch_shape = torch.Size([2])
        self.assertEqual(model.batch_shape, torch.Size([2]))
        m2._batch_shape = torch.Size([3])
        with self.assertRaisesRegex(
            NotImplementedError,
            "is only supported if all constituent models have the same `batch_shape`",
        ):
            model.batch_shape

    def test_posterior_transform(self):
        tkwargs = {"device": self.device, "dtype": torch.double}
        m1 = GenericDeterministicModel(
            lambda X: X.sum(dim=-1, keepdims=True), num_outputs=1
        )
        m2 = GenericDeterministicModel(
            lambda X: X.prod(dim=-1, keepdims=True), num_outputs=1
        )
        model = ModelList(m1, m2)
        X = torch.rand(5, 3, **tkwargs)
        posterior_tf = model.posterior(X, posterior_transform=DummyPosteriorTransform())
        self.assertTrue(
            torch.allclose(
                posterior_tf.mean, torch.cat((2 * m1(X) + 1, 2 * m2(X) + 1), dim=-1)
            )
        )


class TestModelDict(BotorchTestCase):
    def test_model_dict(self):
        models = {"m1": MockModel(MockPosterior()), "m2": MockModel(MockPosterior())}
        model_dict = ModelDict(**models)
        self.assertIs(model_dict["m1"], models["m1"])
        self.assertIs(model_dict["m2"], models["m2"])
        with self.assertRaisesRegex(
            InputDataError, "Expected all models to be a BoTorch `Model`."
        ):
            ModelDict(m=MockPosterior())
