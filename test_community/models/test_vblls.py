#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
from botorch.utils.testing import BotorchTestCase
from botorch_community.models.blls import AbstractBLLModel
from botorch_community.models.vblls import VBLLModel
from botorch_community.posteriors.bll_posterior import BLLPosterior


def _reg_data_singletask(d, n=10):
    X = torch.randn(10, d, dtype=torch.float64)
    y = torch.randn(10, 1, dtype=torch.float64)
    return X, y


def _get_fast_training_settings():
    return {
        "num_epochs": 3,
        "lr": 0.01,
    }


class TestAbstractBLLClass(BotorchTestCase):
    def test_cannot_instantiate_abstract_class(self) -> None:
        # Instantiating AbstractBLLModel directly should raise TypeError
        with self.assertRaises(TypeError):
            _ = AbstractBLLModel()


class TestVBLLModel(BotorchTestCase):
    def test_initialization(self) -> None:
        d, num_hidden, num_outputs, num_layers = 2, 3, 1, 4
        model = VBLLModel(
            in_features=d,
            hidden_features=num_hidden,
            num_layers=num_layers,
            out_features=num_outputs,
            device=self.device,
        )
        self.assertEqual(model.num_inputs, d)
        self.assertEqual(model.num_outputs, num_outputs)
        self.assertEqual(model.device, self.device)

        hidden_layer_count = sum(
            isinstance(layer, torch.nn.Linear)
            for submodule in model.backbone[1:]  # note that the first layer is excluded
            for layer in (
                submodule if isinstance(submodule, torch.nn.Sequential) else [submodule]
            )
        )
        self.assertEqual(
            hidden_layer_count,
            num_layers,
            f"Expected {num_layers} hidden layers, but got {hidden_layer_count}.",
        )

        with self.assertRaises(ValueError):
            _ = VBLLModel(
                in_features=None,
            )

        # test printing
        self.assertEqual(
            str(model),
            str(model.model),
            "model is a wrapper hence str(model) should be equal to str(model.model).",
        )

        with self.assertRaises(ValueError):
            model = VBLLModel(
                in_features=d,
                hidden_features=num_hidden,
                num_layers=num_layers,
                out_features=num_outputs,
                device=self.device,
                parameterization="lowrank",  # lowrank requires cov_rank
            )

    def test_backbone_initialization(self) -> None:
        d, num_hidden = 4, 3
        test_backbone = torch.nn.Sequential(
            torch.nn.Linear(d, num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden),
        )
        model = VBLLModel(backbone=test_backbone, hidden_features=num_hidden)

        for key in test_backbone.state_dict():
            self.assertTrue(
                torch.allclose(
                    test_backbone.state_dict()[key],
                    model.backbone.state_dict()[key],
                    atol=1e-6,
                ),
                f"Mismatch of backbone state_dict for key: {key}",
            )

        # no Linear layer -> need explicit input size
        test_backbone = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(d, num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden),
        )
        model = VBLLModel(
            backbone=test_backbone, in_features=d, hidden_features=num_hidden
        )

        with self.assertRaises(ValueError):
            _ = VBLLModel(backbone=test_backbone, hidden_features=num_hidden)

    def test_training(self) -> None:
        d, num_hidden = 4, 4
        # test for all parameterizations of the VBLL head
        for covar_type in ("diagonal", "dense", "lowrank", "dense_precision"):
            # test for both, frozen and unfrozen backbone
            for freeze_backbone in (True, False):
                test_backbone = torch.nn.Sequential(
                    torch.nn.Linear(d, num_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(num_hidden, num_hidden),
                    torch.nn.ELU(),
                ).to(dtype=torch.float64)

                model = VBLLModel(
                    backbone=copy.deepcopy(test_backbone),
                    hidden_features=num_hidden,  # match the output of the backbone
                    parameterization=covar_type,
                    cov_rank=2 if covar_type == "lowrank" else None,
                )

                X, y = _reg_data_singletask(d)
                optim_settings = {
                    "num_epochs": 1,
                    "lr": 10.0,  # large lr to make sure that the weights change
                    "freeze_backbone": freeze_backbone,
                }
                model.fit(X, y, optimization_settings=optim_settings)

                if freeze_backbone:
                    # Ensure all parameters remain unchanged
                    model_bb_sdict = model.backbone.state_dict()
                    for key, val in test_backbone.state_dict().items():
                        self.assertTrue(
                            torch.allclose(val, model_bb_sdict[key], atol=1e-6),
                            f"Parameter {key} changed with freeze_backbone=True."
                            f"(Parameterization: {covar_type})",
                        )
                else:
                    # Ensure at least one parameter has changed
                    model_bb_sdict = model.backbone.state_dict()
                    changed = False
                    for key, val in test_backbone.state_dict().items():
                        if not torch.allclose(val, model_bb_sdict[key], atol=1e-6):
                            changed = True
                            break
                    self.assertTrue(
                        changed,
                        "Expected at least one parameter to change, but all remained "
                        f"the same with {freeze_backbone=}"
                        f"(Parameterization: {covar_type})",
                    )

    def test_early_stopping(self) -> None:
        d, num_hidden = 4, 4
        # test for all parameterizations of the VBLL head
        for covar_type in ("diagonal", "dense", "lowrank", "dense_precision"):
            # test for both, frozen and unfrozen backbone
            test_backbone = torch.nn.Sequential(
                torch.nn.Linear(d, num_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(num_hidden, num_hidden),
                torch.nn.ELU(),
            ).to(dtype=torch.float64)

            model = VBLLModel(
                backbone=copy.deepcopy(test_backbone),
                hidden_features=num_hidden,  # match the output of the backbone
                parameterization=covar_type,
                cov_rank=2 if covar_type == "lowrank" else None,
            )

            X, y = _reg_data_singletask(d)
            optim_settings = {
                "num_epochs": 3,
                "patience": 1,
                "lr": 0.0,  # no learning rate
            }
            with self.assertLogs(logger="botorch", level="INFO") as logs_cm:
                model.fit(X, y, optimization_settings=optim_settings)

            msg_contained = False
            for msg in logs_cm.output:
                if "Early stopping at epoch" in msg:
                    msg_contained = True
                    break

            self.assertTrue(
                msg_contained,
                "Expected early stopping log message not found.",
            )

    def test_initialization_of_model_parameters(self) -> None:
        d, num_hidden = 4, 4
        for covar_type in ("diagonal", "dense", "lowrank", "dense_precision"):
            # test for both, frozen and unfrozen backbone
            for freeze_backbone in (True, False):
                test_backbone = torch.nn.Sequential(
                    torch.nn.Linear(d, num_hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(num_hidden, num_hidden),
                    torch.nn.ELU(),
                ).to(dtype=torch.float64)

                model = VBLLModel(
                    backbone=copy.deepcopy(test_backbone),
                    hidden_features=num_hidden,  # match the output of the backbone
                    parameterization=covar_type,
                    cov_rank=2 if covar_type == "lowrank" else None,
                )

                X, y = _reg_data_singletask(d)
                optim_settings = {
                    "num_epochs": 1,
                    "lr": 10.0,  # large lr to make sure that the weights change
                    "freeze_backbone": freeze_backbone,
                }
                model.fit(X, y, optimization_settings=optim_settings)

                # store the initial state dict
                state_dict = model.model.state_dict()

                # use state_dict when fitting new model
                model2 = VBLLModel(
                    backbone=copy.deepcopy(test_backbone),
                    hidden_features=num_hidden,  # match the output of the backbone
                    parameterization=covar_type,
                    cov_rank=2 if covar_type == "lowrank" else None,
                )

                optim_settings = {
                    "num_epochs": 0,  # skip training
                    "lr": 10.0,
                    "freeze_backbone": freeze_backbone,
                }
                model2.fit(
                    X,
                    y,
                    optimization_settings=optim_settings,
                    initialization_params=state_dict,
                )

                # check that the parameters are the same
                model2_sdict = model2.model.state_dict()
                for key, val in state_dict.items():
                    self.assertTrue(
                        torch.allclose(val, model2_sdict[key], atol=1e-6),
                        f"Parameter {key} changed with freeze_backbone=True."
                        f"(Parameterization: {covar_type})",
                    )

    def test_update_of_reg_weight(self) -> None:
        kl_scale = 2.0
        d = 2
        model = VBLLModel(
            in_features=d,
            hidden_features=3,
            out_features=1,
            num_layers=1,
            kl_scale=kl_scale,
        )
        self.assertEqual(
            model.model.head.regularization_weight,
            1.0,
            "Regularization weight should be 1.0 after init.",
        )

        X, y = _reg_data_singletask(d)

        optim_settings = _get_fast_training_settings()
        model.fit(X, y, optimization_settings=optim_settings)

        self.assertEqual(
            model.model.head.regularization_weight,
            kl_scale / len(y),
            f"Regularization weight should be {kl_scale}/{len(y)}, but got"
            f"{model.model.head.regularization_weight}.",
        )

    def test_shape_of_sampling(self) -> None:
        d = 4

        for out_features in (1, 2):
            model = VBLLModel(
                in_features=d,
                hidden_features=4,
                out_features=out_features,
                num_layers=1,
            )
            n_test_points = 3
            X = torch.rand(n_test_points, d, dtype=torch.float64)
            for sample_shape in (torch.Size(), torch.Size([2])):
                sample_path = model.sample(sample_shape=sample_shape)
                y_hat = sample_path(X)
                expected_shape = sample_shape + torch.Size(
                    [n_test_points, out_features]
                )
                self.assertEqual(
                    y_hat.shape,
                    expected_shape,
                    f"Expected samples to have shape {expected_shape},"
                    f"but got {y_hat.shape}.",
                )

    def test_shape_of_forward(self) -> None:
        d = 4
        for out_features in (1, 2):
            model = VBLLModel(
                in_features=d,
                hidden_features=4,
                out_features=out_features,
                num_layers=1,
            )
            n_test_points = 3
            X = torch.rand(n_test_points, d, dtype=torch.float64)
            y_pred = model(X)

            y_hat = y_pred.mean
            y_var = y_pred.variance

            expected_shape = torch.Size([n_test_points, out_features])
            self.assertEqual(
                y_hat.shape,
                expected_shape,
                f"Expected forward pass to have shape {expected_shape},"
                f"but got {y_hat.shape}.",
            )
            self.assertEqual(
                y_var.shape,
                expected_shape,
                f"Expected forward pass to have shape {expected_shape},"
                f"but got {y_var.shape}.",
            )

    def test_shape_of_predictions(self) -> None:
        d = 4
        model = VBLLModel(
            in_features=d, hidden_features=4, out_features=1, num_layers=1
        )
        X_train, y_train = _reg_data_singletask(d)
        optim_settings = _get_fast_training_settings()
        model.fit(X_train, y_train, optimization_settings=optim_settings)

        valid_inputs = [
            torch.rand(3, d, dtype=torch.float64),  # (3, d)
            torch.rand(1, 3, d, dtype=torch.float64),  # (1, 3, d)
            torch.rand(2, 3, d, dtype=torch.float64),  # (2, 3, d)
            torch.rand(1, d, dtype=torch.float64),  # (1, d)
        ]

        for X_test in valid_inputs:
            if X_test.ndim == 1:
                expected_shape = torch.Size([1, 1])  # (1 sample, 1 output)
            elif X_test.ndim == 2:
                expected_shape = torch.Size([X_test.shape[0], 1])
            else:
                batch_shape = X_test.shape[:-2]
                expected_shape = batch_shape + torch.Size([X_test.shape[-2], 1])

            post = model.posterior(X_test)

            self.assertIsInstance(
                post,
                BLLPosterior,
                "Expected posterior to be an instance of BLLPosterior.",
            )

            # mean prediction
            self.assertEqual(
                post.mean.shape,
                expected_shape,
                f"Expected mean predictions to have shape {expected_shape},"
                f"but got {post.mean.shape}.",
            )

            self.assertEqual(
                post.variance.shape,
                expected_shape,
                f"Expected variance predictions to have shape {expected_shape},"
                f" but got {post.variance.shape}.",
            )

        # validate that posterior fails when x.dim > 3
        X = torch.rand(torch.Size([2, 5]) + torch.Size([3, d]), dtype=torch.float64)

        with self.assertRaises(ValueError):
            _ = model.posterior(X)

    def test_validation_loss(self) -> None:
        """Test that the model properly handles validation data during fitting."""
        d, num_hidden = 4, 4

        # Test for multiple parameterizations of the VBLL head
        for covar_type in ("diagonal", "dense", "lowrank", "dense_precision"):
            # Create model
            model = VBLLModel(
                in_features=d,
                hidden_features=num_hidden,
                out_features=1,
                num_layers=1,
                parameterization=covar_type,
                cov_rank=2 if covar_type == "lowrank" else None,
            )

            # Generate training and validation data
            X_train, y_train = _reg_data_singletask(d, n=10)
            X_val, y_val = _reg_data_singletask(d, n=5)

            # Fast training settings for testing
            optim_settings = {
                "num_epochs": 3,
                "patience": 1,
                "lr": 0.00,
            }

            # Test with validation data
            with self.assertLogs(logger="botorch", level="INFO") as logs_cm:
                model.fit(
                    X_train,
                    y_train,
                    val_X=X_val,
                    val_y=y_val,
                    optimization_settings=optim_settings,
                )

            # Check for early stopping message in logs
            msg_contained = False
            for msg in logs_cm.output:
                if "Early stopping at epoch" in msg:
                    msg_contained = True
                    break

            self.assertTrue(
                msg_contained,
                f"Expected early stopping log message not found"
                f"with parameterization {covar_type}.",
            )

            # providing only val data X or only y should raise error
            with self.assertRaises(ValueError):
                model.fit(
                    X_train,
                    y_train,
                    val_X=X_val,
                    val_y=None,
                    optimization_settings=optim_settings,
                )

            with self.assertRaises(ValueError):
                model.fit(
                    X_train,
                    y_train,
                    val_X=None,
                    val_y=y_val,
                    optimization_settings=optim_settings,
                )
