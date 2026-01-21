#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from logging import DEBUG, WARN
from unittest.mock import MagicMock, mock_open, patch

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.transforms.input import Normalize
from botorch.utils.testing import BotorchTestCase
from botorch_community.models.prior_fitted_network import (
    BoundedRiemannPosterior,
    DirectAcquisitionPFNModel,
    MultivariatePFNModel,
    PFNModel,
    PFNModelWithPendingPoints,
)
from botorch_community.models.utils.prior_fitted_network import (
    download_model,
    ModelPaths,
)
from botorch_community.posteriors.riemann import MultivariateRiemannPosterior
from pfns.model.transformer_config import CrossEntropyConfig, TransformerConfig
from pfns.train import MainConfig, OptimizerConfig
from torch import nn, Tensor


class DummyPFN(nn.Module):
    def __init__(self, n_buckets: int = 1000):
        """A dummy PFN model for testing purposes.

        This class implements a mocked PFN model that returns
        constant values for testing. It mimics the interface of actual PFN models
        but with simplified behavior.

        Args:
            n_buckets: Number of buckets for the output distribution. Default is 1000.
        """

        super().__init__()
        self.n_buckets = n_buckets
        self.criterion = MagicMock()
        self.criterion.borders = torch.linspace(0, 1, n_buckets + 1)
        self.style_encoder = None
        self.y_style_encoder = None

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        test_x: Tensor,
        style: Tensor | None = None,
        y_style: Tensor | None = None,
    ) -> Tensor:
        return torch.zeros(*test_x.shape[:-1], self.n_buckets, device=test_x.device)


class TestPriorFittedNetwork(BotorchTestCase):
    def test_raises(self):
        for dtype in (torch.float, torch.double):
            for model_type in (PFNModel, PFNModelWithPendingPoints):
                with self.subTest(model_type=model_type, dtype=dtype):
                    tkwargs = {"device": self.device, "dtype": dtype}
                    train_X = torch.rand(10, 3, **tkwargs)
                    train_Y = torch.rand(10, 1, **tkwargs)
                    train_Yvar = torch.rand(10, 1, **tkwargs)
                    test_X = torch.rand(5, 3, **tkwargs)

                    with self.assertLogs(logger="botorch", level=DEBUG) as log:
                        model_type(train_X, train_Y, DummyPFN(), train_Yvar=train_Yvar)
                        self.assertIn(
                            "train_Yvar provided but ignored for PFNModel.",
                            log.output[0],
                        )

                    train_Y_4d = torch.rand(10, 2, 2, 1, **tkwargs)
                    with self.assertRaisesRegex(
                        UnsupportedError, "train_Y must be 2-dimensional"
                    ):
                        model_type(train_X, train_Y_4d, DummyPFN())

                    train_Y_2d = torch.rand(10, 2, **tkwargs)
                    with self.assertRaisesRegex(
                        UnsupportedError, "Only 1 target allowed"
                    ):
                        model_type(train_X, train_Y_2d, DummyPFN())

                    with self.assertRaisesRegex(
                        UnsupportedError, "train_X must be 2-dimensional"
                    ):
                        model_type(
                            torch.rand(10, 3, 3, 2, **tkwargs), train_Y, DummyPFN()
                        )
                    with self.assertRaisesRegex(
                        UnsupportedError, "same number of rows"
                    ):
                        model_type(train_X, torch.rand(11, 1, **tkwargs), DummyPFN())

                    pfn = model_type(train_X, train_Y, DummyPFN())

                    with self.assertRaisesRegex(
                        UnsupportedError, "output_indices is not None"
                    ):
                        pfn.posterior(test_X, output_indices=[0, 1])
                    with self.assertLogs(logger="botorch", level=WARN) as log:
                        pfn.posterior(test_X, observation_noise=True)
                        self.assertIn(
                            "observation_noise is not supported for PFNModel",
                            log.output[0],
                        )
                    with self.assertRaisesRegex(
                        UnsupportedError, "posterior_transform is not supported"
                    ):
                        pfn.posterior(
                            test_X,
                            posterior_transform=ScalarizedPosteriorTransform(
                                weights=torch.ones(1)
                            ),
                        )

                    # (b', b, d) prediction works as expected
                    test_X = torch.rand(5, 4, 2, **tkwargs)
                    post = pfn.posterior(test_X)
                    self.assertEqual(post.mean.shape, torch.Size([5, 4, 1]))

                    # X dims should be 1 to 4
                    test_X = torch.rand(5, 4, 2, 1, 2, **tkwargs)
                    with self.assertRaisesRegex(
                        UnsupportedError, "X must be at most 3-d"
                    ):
                        pfn.posterior(test_X)

    def test_shapes(self):
        tkwargs = {"device": self.device, "dtype": torch.float}

        # no q dimension
        train_X = torch.rand(10, 3, **tkwargs)
        train_Y = torch.rand(10, 1, **tkwargs)
        test_X = torch.rand(5, 3, **tkwargs)

        pfn = PFNModel(train_X, train_Y, DummyPFN(n_buckets=100))

        for batch_first in [True, False]:
            with self.subTest(batch_first=batch_first):
                pfn.batch_first = batch_first
                posterior = pfn.posterior(test_X)

                self.assertEqual(posterior.probabilities.shape, torch.Size([5, 100]))
                self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))

        # q=1
        test_X = torch.rand(5, 1, 3, **tkwargs)
        posterior = pfn.posterior(test_X)

        self.assertEqual(posterior.probabilities.shape, torch.Size([5, 1, 100]))
        self.assertEqual(posterior.mean.shape, torch.Size([5, 1, 1]))

        # no shape basically
        test_X = torch.rand(3, **tkwargs)
        posterior = pfn.posterior(test_X)

        self.assertEqual(posterior.probabilities.shape, torch.Size([100]))
        self.assertEqual(posterior.mean.shape, torch.Size([1]))

        # prepare_data
        X = torch.rand(5, 3, **tkwargs)
        X, train_X, train_Y, orig_X_shape, styles = pfn._prepare_data(X)
        self.assertEqual(X.shape, torch.Size([1, 5, 3]))
        self.assertEqual(train_X.shape, torch.Size([1, 10, 3]))
        self.assertEqual(train_Y.shape, torch.Size([1, 10, 1]))
        self.assertEqual(orig_X_shape, torch.Size([5, 3]))
        self.assertEqual(styles, {})
        pfn.style = torch.rand(4, **tkwargs)
        X, train_X, train_Y, orig_X_shape, styles = pfn._prepare_data(X)
        self.assertEqual(styles["style"].shape, torch.Size([1, 1, 4]))

    def test_input_transform(self):
        model = PFNModel(
            train_X=torch.rand(10, 3),
            train_Y=torch.rand(10, 1),
            input_transform=Normalize(d=3),
            model=DummyPFN(),
        )
        self.assertIsInstance(model.input_transform, Normalize)
        self.assertEqual(model.input_transform.bounds.shape, torch.Size([2, 3]))

    def test_style_hyperparameters(self):
        """Test that style_hyperparameters are stored and passed through get_styles."""
        train_X, train_Y = torch.rand(10, 3), torch.rand(10, 1)
        style_hps = {"noise_std": 0.1}

        # Create PFN with mock style_encoder and y_style_encoder
        dummy_pfn = DummyPFN()
        mock_encoder = MagicMock()
        mock_encoder.hyperparameters = ["noise_std"]
        mock_encoder.hyperparameters_dict_to_tensor.return_value = torch.tensor([0.5])
        mock_y_encoder = MagicMock()
        mock_y_encoder.hyperparameters = ["noise_std"]
        mock_y_encoder.hyperparameters_dict_to_tensor.return_value = torch.tensor(
            [0.25]
        )
        dummy_pfn.style_encoder = [mock_encoder]
        dummy_pfn.y_style_encoder = [mock_y_encoder]

        pfn = PFNModel(train_X, train_Y, dummy_pfn, style_hyperparameters=style_hps)
        self.assertEqual(pfn.style_hyperparameters, style_hps)

        # Capture kwargs passed to forward
        captured = {}
        orig_forward = dummy_pfn.forward
        dummy_pfn.forward = lambda *a, **kw: (
            captured.update(kw),
            orig_forward(**kw),
        )[1]

        pfn.posterior(torch.rand(5, 3))

        self.assertIn("style", captured)
        self.assertIn("y_style", captured)
        self.assertEqual(captured["style"].item(), 0.5)
        self.assertEqual(captured["y_style"].item(), 0.25)
        mock_encoder.hyperparameters_dict_to_tensor.assert_called_once_with(style_hps)
        mock_y_encoder.hyperparameters_dict_to_tensor.assert_called_once_with(style_hps)

    def test_style_params_require_style_hyperparameters(self):
        """Test no style params if style_hyperparameters=None, some when {}."""
        train_X, train_Y = torch.rand(10, 3), torch.rand(10, 1)
        dummy_pfn = DummyPFN()
        mock_enc = MagicMock()
        mock_enc.hyperparameters = []
        mock_enc.hyperparameters_dict_to_tensor.return_value = torch.tensor([0.5])
        dummy_pfn.style_encoder = [mock_enc]

        captured = {}
        orig = dummy_pfn.forward
        dummy_pfn.forward = lambda *a, **kw: (captured.update(kw), orig(**kw))[1]

        PFNModel(train_X, train_Y, dummy_pfn).posterior(torch.rand(5, 3))
        self.assertNotIn("style", captured)

        pfn = PFNModel(train_X, train_Y, dummy_pfn, style_hyperparameters={})
        pfn.posterior(torch.rand(5, 3))
        self.assertIn("style", captured)

    def test_raw_style_tensor(self):
        """Test that raw style tensor is passed through when provided."""
        train_X, train_Y = torch.rand(10, 3), torch.rand(10, 1)
        style = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        captured = {}
        dummy_pfn = DummyPFN()
        orig = dummy_pfn.forward
        dummy_pfn.forward = lambda *a, **kw: (captured.update(kw), orig(**kw))[1]

        pfn = PFNModel(train_X, train_Y, dummy_pfn, style=style)
        pfn.posterior(torch.rand(5, 3))

        self.assertIn("style", captured)
        self.assertEqual(captured["style"].shape[1:], style.shape)
        self.assertTrue(torch.equal(captured["style"][0], style))

    def test_unpack_checkpoint(self):
        config = MainConfig(
            priors=[],
            optimizer=OptimizerConfig(
                optimizer="adam",
                lr=0.001,
            ),
            model=TransformerConfig(
                criterion=CrossEntropyConfig(num_classes=3),
            ),
            batch_shape_sampler=None,
        )

        model = config.model.create_model()

        state_dict = model.state_dict()
        checkpoint = {
            "config": config.to_dict(),
            "model_state_dict": state_dict,
        }

        loaded_model = PFNModel(
            train_X=torch.rand(10, 3),
            train_Y=torch.rand(10, 1),
            input_transform=Normalize(d=3),
            model=checkpoint,
            load_training_checkpoint=True,
        )

        loaded_state_dict = loaded_model.pfn.state_dict()
        self.assertEqual(
            sorted(loaded_state_dict.keys()),
            sorted(state_dict.keys()),
        )
        for k in loaded_state_dict.keys():
            self.assertTrue(torch.equal(loaded_state_dict[k], state_dict[k]))


class TestPriorFittedNetworkUtils(BotorchTestCase):
    @patch("botorch_community.models.utils.prior_fitted_network.requests.get")
    @patch("botorch_community.models.utils.prior_fitted_network.gzip.GzipFile")
    @patch("botorch_community.models.utils.prior_fitted_network.torch.load")
    @patch("botorch_community.models.utils.prior_fitted_network.torch.save")
    @patch("botorch_community.models.utils.prior_fitted_network.os.path.exists")
    @patch("botorch_community.models.utils.prior_fitted_network.os.makedirs")
    def test_download_model_cache_miss(
        self,
        _mock_makedirs,
        mock_exists,
        mock_torch_save,
        mock_torch_load,
        mock_gzip,
        mock_requests_get,
    ):
        # Simulate cache miss
        mock_exists.return_value = False

        # Mock the requests.get to simulate a network call
        mock_requests_get.return_value = MagicMock(
            status_code=200, content=b"fake content"
        )

        # Mock the gzip.GzipFile to simulate decompression
        mock_gzip.return_value.__enter__.return_value = mock_open(
            read_data=b"fake model data"
        ).return_value

        # Mock torch.load to simulate loading a model
        fake_model = MagicMock(spec=torch.nn.Module)
        mock_torch_load.return_value = fake_model

        # Call the function
        model = download_model(
            ModelPaths.pfns4bo_hebo,
            cache_dir=os.environ.get("RUNNER_TEMP", "/tmp") + "/test_cache",
            # $RUNNER_TEMP is set by GitHub Actions as tmp, /tmp does not work there
        )

        # Assertions for cache miss
        mock_requests_get.assert_called_once()
        mock_gzip.assert_called_once()
        mock_torch_load.assert_called_once()
        mock_torch_save.assert_called_once()
        self.assertEqual(model, fake_model)

        # Test loading in model init
        model = PFNModel(
            train_X=torch.rand(10, 3),
            train_Y=torch.rand(10, 1),
        )
        self.assertEqual(model.pfn, fake_model.to("cpu"))

    @patch("botorch_community.models.utils.prior_fitted_network.torch.load")
    @patch("botorch_community.models.utils.prior_fitted_network.os.path.exists")
    def test_download_model_cache_hit(self, mock_exists, mock_torch_load):
        # Simulate cache hit
        mock_exists.return_value = True

        # Mock torch.load to simulate loading a model
        fake_model = MagicMock(spec=torch.nn.Module)
        mock_torch_load.return_value = fake_model

        # Call the function
        model = download_model(
            ModelPaths.pfns4bo_hebo,
            cache_dir=os.environ.get("RUNNER_TEMP", "/tmp") + "/test_cache",
            # $RUNNER_TEMP is set by GitHub Actions as tmp, /tmp does not work there
        )

        # Assertions for cache hit
        # mock_exists is called once here and once through os.makedirs
        # which checks if directory exists
        self.assertEqual(mock_exists.call_count, 2)
        mock_torch_load.assert_called_once()
        self.assertEqual(model, fake_model)


class TestMultivariatePFN(BotorchTestCase):
    def setUp(self):
        train_X = torch.rand(10, 3)
        train_Y = torch.rand(10, 1)
        self.pfn = MultivariatePFNModel(train_X, train_Y, DummyPFN())

    def test_posterior(self):
        X = torch.rand(1, 3)
        post = self.pfn.posterior(X)
        self.assertNotIsInstance(post, MultivariateRiemannPosterior)
        X = torch.rand(4, 3)
        R = torch.rand(1, 4, 4)
        with patch(
            "botorch_community.models.prior_fitted_network.MultivariatePFNModel"
            ".estimate_correlations",
            return_value=R,
        ):
            post = self.pfn.posterior(X)
        self.assertIsInstance(post, MultivariateRiemannPosterior)
        self.assertTrue(torch.equal(post.correlation_matrix, R.squeeze(0)))

    def test_estimate_covariances(self):
        b = 3
        q = 4
        cond_val = torch.rand(b, q)
        cond_mean = torch.rand(b, q, q)
        var = torch.ones(b, q)
        mean = torch.rand(b, q)
        # Fill in particular values for the [1, 1, 2] entries
        mean[1, 1] = 2.0
        mean[1, 2] = 3.0
        cond_mean[1, 2, 1] = 3.0
        cond_mean[1, 1, 2] = 4.0
        cond_mean[1, 1, 1] = 2.1
        cond_mean[1, 2, 2] = 3.1
        cond_val[1, 1] = 3.0
        cond_val[1, 2] = 4.0
        with patch(
            "botorch_community.models.prior_fitted_network.MultivariatePFNModel."
            "_map_psd"
        ) as mock_map_psd:
            self.pfn._estimate_covariances(
                cond_mean=cond_mean, cond_val=cond_val, mean=mean, var=var
            )
        cov = mock_map_psd.call_args[0][0]
        # Compare to analytical value of 10
        self.assertEqual(torch.round(cov[1, 1, 2], decimals=2).item(), 10.0)
        self.assertEqual(torch.round(cov[1, 2, 1], decimals=2).item(), 10.0)

    def test_compute_conditional_means(self):
        probabilities = torch.zeros(3, 2, 1000)
        probabilities[0, 0, 9] = 1.0
        probabilities[0, 1, 19] = 1.0
        probabilities[1, 0, 29] = 1.0
        probabilities[1, 1, 39] = 1.0
        probabilities[2, 0, 49] = 1.0
        probabilities[2, 1, 59] = 1.0
        marginals = BoundedRiemannPosterior(
            borders=self.pfn.borders,
            probabilities=probabilities,
        )
        return_value = torch.zeros(3 * 2, 2, 1000)
        return_value[..., 100] = 1.0
        X = torch.ones(3, 2, 5)
        X[:, 1, :] = 2.0
        with patch(
            "botorch_community.models.prior_fitted_network.MultivariatePFNModel."
            "pfn_predict",
            return_value=return_value,
        ) as mock_pfn_predict:
            self.pfn._compute_conditional_means(
                X=X,
                train_X=torch.zeros(3, 4, 5),
                train_Y=torch.zeros(3, 4, 1),
                styles={"style": torch.zeros(3, 7)},
                marginals=marginals,
            )
        res = mock_pfn_predict.call_args[1]
        self.assertTrue(torch.equal(res["X"], torch.cat([X, X])))
        X1 = torch.zeros(1, 5, 5)
        X1[:, -1, :] = 1.0
        X2 = torch.zeros(1, 5, 5)
        X2[:, -1, :] = 2.0
        self.assertTrue(
            torch.equal(res["train_X"], torch.cat([X1, X2, X1, X2, X1, X2], dim=0))
        )
        a = []
        for i in range(6):
            Y = torch.zeros(1, 5, 1)
            Y[0, -1, 0] = (i + 1) * 0.01
            a.append(Y)
        self.assertTrue(
            torch.equal(torch.round(res["train_Y"], decimals=2), torch.cat(a, dim=0))
        )
        # Verify style and y_style are passed correctly
        self.assertTrue(torch.equal(res["style"], torch.zeros(6, 7)))
        self.assertNotIn("y_style", res)

    def test_estimate_correlations(self):
        probabilities = torch.ones(2, 3, 1000)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        marginals = BoundedRiemannPosterior(
            borders=self.pfn.borders,
            probabilities=probabilities,
        )
        cond_mean = 0.5 * (1 + torch.rand(2, 3, 3))
        with patch(
            "botorch_community.models.prior_fitted_network.MultivariatePFNModel."
            "_compute_conditional_means",
            return_value=(cond_mean, 0.9 * torch.ones(2, 3)),
        ):
            R = self.pfn.estimate_correlations(
                X=torch.ones(2, 3, 5),
                train_X=torch.zeros(2, 4, 5),
                train_Y=torch.zeros(2, 4, 1),
                styles={"style": torch.zeros(2, 4), "y_style": torch.ones(2, 4)},
                marginals=marginals,
            )
        self.assertAllClose(torch.diagonal(R, dim1=-2, dim2=-1), torch.ones(2, 3))
        # Test with no batch dimension
        marginals = BoundedRiemannPosterior(
            borders=self.pfn.borders,
            probabilities=probabilities[0, ...],
        )
        cond_mean = cond_mean[:1, ...]
        with patch(
            "botorch_community.models.prior_fitted_network.MultivariatePFNModel."
            "_compute_conditional_means",
            return_value=(cond_mean, 0.9 * torch.ones(1, 3)),
        ):
            R = self.pfn.estimate_correlations(
                X=torch.ones(1, 3, 5),
                train_X=torch.zeros(1, 4, 5),
                train_Y=torch.zeros(1, 4, 1),
                styles={"style": torch.zeros(1, 4), "y_style": torch.ones(1, 4)},
                marginals=marginals,
            )
        self.assertEqual(R.shape, torch.Size([1, 3, 3]))
        self.assertAllClose(torch.diagonal(R, dim1=-2, dim2=-1), torch.ones(1, 3))


class TestPFNModelWithPendingPoints(BotorchTestCase):
    def setUp(self):
        self.train_X = torch.rand(10, 3)
        self.train_Y = torch.rand(10, 1)
        self.pfn = PFNModelWithPendingPoints(
            self.train_X, self.train_Y, DummyPFN(n_buckets=100)
        )

    def test_posterior_without_pending_X(self):
        """Test that posterior works the same as PFNModel when pending_X is None."""
        test_X = torch.rand(5, 3)
        posterior = self.pfn.posterior(test_X)
        self.assertIsInstance(posterior, BoundedRiemannPosterior)
        self.assertEqual(posterior.probabilities.shape, torch.Size([5, 100]))
        self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))

    def test_posterior_with_pending_X(self):
        """Test that posterior correctly handles pending_X."""
        test_X = torch.rand(5, 3)
        pending_X = torch.rand(3, 3)  # 3 pending points

        posterior = self.pfn.posterior(test_X, pending_X=pending_X)
        self.assertIsInstance(posterior, BoundedRiemannPosterior)
        self.assertEqual(posterior.probabilities.shape, torch.Size([5, 100]))
        self.assertEqual(posterior.mean.shape, torch.Size([5, 1]))

    def test_posterior_with_pending_X_batched(self):
        """Test that posterior correctly handles pending_X with batched input."""
        test_X = torch.rand(2, 5, 3)  # batched input (b=2, q=5)
        pending_X = torch.rand(3, 3)  # 3 pending points

        posterior = self.pfn.posterior(test_X, pending_X=pending_X)
        self.assertIsInstance(posterior, BoundedRiemannPosterior)
        self.assertEqual(posterior.probabilities.shape, torch.Size([2, 5, 100]))
        self.assertEqual(posterior.mean.shape, torch.Size([2, 5, 1]))

    def test_pending_X_concatenation(self):
        """Test that pending_X is correctly concatenated to train_X."""
        test_X = torch.rand(5, 3)
        pending_X = torch.rand(3, 3)  # 3 pending points

        # Capture the inputs to pfn_predict
        captured = {}
        orig_pfn_predict = self.pfn.pfn_predict

        def capture_pfn_predict(X, train_X, train_Y, **kwargs):
            captured["train_X"] = train_X
            captured["train_Y"] = train_Y
            return orig_pfn_predict(X, train_X, train_Y, **kwargs)

        self.pfn.pfn_predict = capture_pfn_predict
        self.pfn.posterior(test_X, pending_X=pending_X)

        # train_X should have shape (1, n + n', d) where n=10, n'=3
        self.assertEqual(captured["train_X"].shape, torch.Size([1, 13, 3]))
        # train_Y should have shape (1, n + n', 1) with NaN for pending points
        self.assertEqual(captured["train_Y"].shape, torch.Size([1, 13, 1]))
        # Last 3 entries of train_Y should be NaN
        self.assertTrue(torch.isnan(captured["train_Y"][:, -3:, :]).all())
        # First 10 entries should not be NaN
        self.assertFalse(torch.isnan(captured["train_Y"][:, :10, :]).any())

    def test_pending_X_must_be_2d(self):
        """Test that pending_X must be 2-dimensional."""
        test_X = torch.rand(5, 3)
        pending_X_3d = torch.rand(2, 3, 3)  # 3D tensor

        with self.assertRaises(AssertionError):
            self.pfn.posterior(test_X, pending_X=pending_X_3d)

    def test_pending_X_with_negate_train_ys(self):
        """Test that pending_X works with negate_train_ys=True."""
        # Create zero-centered train_Y for negate_train_ys
        train_Y_centered = self.train_Y - self.train_Y.mean()
        pfn = PFNModelWithPendingPoints(
            self.train_X, train_Y_centered, DummyPFN(n_buckets=100)
        )

        test_X = torch.rand(5, 3)
        pending_X = torch.rand(3, 3)

        posterior = pfn.posterior(test_X, pending_X=pending_X, negate_train_ys=True)
        self.assertIsInstance(posterior, BoundedRiemannPosterior)
        self.assertEqual(posterior.probabilities.shape, torch.Size([5, 100]))


class DummyDirectAcquisitionPFN(nn.Module):
    """A dummy PFN model that outputs a single logit (direct acquisition value)."""

    def __init__(self):
        super().__init__()
        self.style_encoder = None
        self.y_style_encoder = None

    def forward(self, train_X: Tensor, train_Y: Tensor, test_X: Tensor) -> Tensor:
        # Returns a single logit per test point (shape: batch x q)
        return torch.randn(*test_X.shape[:-1], device=test_X.device)


class TestDirectAcquisitionPFNModel(BotorchTestCase):
    def test_acquisition_values(self):
        """Test that get_acquisition_values returns correctly shaped tensors."""
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X = torch.rand(10, 3, **tkwargs)
            train_Y = torch.rand(10, 1, **tkwargs)
            train_Y = train_Y - train_Y.mean()  # Zero-center for negate_train_ys

            model = DirectAcquisitionPFNModel(
                train_X, train_Y, DummyDirectAcquisitionPFN()
            )

            # Test with 2D input (n, d)
            test_X = torch.rand(5, 3, **tkwargs)
            acq_values = model.get_acquisition_values(test_X)
            self.assertEqual(acq_values.shape, torch.Size([5]))

            # Test with 3D input (b, q, d)
            test_X = torch.rand(2, 5, 3, **tkwargs)
            acq_values = model.get_acquisition_values(test_X)
            self.assertEqual(acq_values.shape, torch.Size([2, 5]))

            # Test with 1D input (d,)
            test_X = torch.rand(3, **tkwargs)
            acq_values = model.get_acquisition_values(test_X)
            self.assertEqual(acq_values.shape, torch.Size([]))

            # Test with negate_train_ys=True
            test_X = torch.rand(5, 3, **tkwargs)
            acq_values = model.get_acquisition_values(test_X, negate_train_ys=True)
            self.assertEqual(acq_values.shape, torch.Size([5]))

    def test_raises(self):
        """Test error handling for invalid inputs."""
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            train_X = torch.rand(10, 3, **tkwargs)
            train_Y = torch.rand(10, 1, **tkwargs)

            # Test 4D train_Y raises
            train_Y_4d = torch.rand(10, 2, 2, 1, **tkwargs)
            with self.assertRaisesRegex(
                UnsupportedError, "train_Y must be 2-dimensional"
            ):
                DirectAcquisitionPFNModel(
                    train_X, train_Y_4d, DummyDirectAcquisitionPFN()
                )

            # Test multi-output raises
            train_Y_2d = torch.rand(10, 2, **tkwargs)
            with self.assertRaisesRegex(UnsupportedError, "Only 1 target allowed"):
                DirectAcquisitionPFNModel(
                    train_X, train_Y_2d, DummyDirectAcquisitionPFN()
                )

            # Test train_X must be 2D
            with self.assertRaisesRegex(
                UnsupportedError, "train_X must be 2-dimensional"
            ):
                DirectAcquisitionPFNModel(
                    torch.rand(10, 3, 3, 2, **tkwargs),
                    train_Y,
                    DummyDirectAcquisitionPFN(),
                )

            # Test mismatched row counts
            with self.assertRaisesRegex(UnsupportedError, "same number of rows"):
                DirectAcquisitionPFNModel(
                    train_X, torch.rand(11, 1, **tkwargs), DummyDirectAcquisitionPFN()
                )

            # Test X dims should be 1 to 3
            model = DirectAcquisitionPFNModel(
                train_X, train_Y, DummyDirectAcquisitionPFN()
            )
            test_X = torch.rand(5, 4, 2, 1, 2, **tkwargs)
            with self.assertRaisesRegex(UnsupportedError, "X must be at most 3-d"):
                model.get_acquisition_values(test_X)

    def test_input_transform(self):
        """Test that input transforms work correctly."""
        train_X = torch.rand(10, 3)
        train_Y = torch.rand(10, 1)

        model = DirectAcquisitionPFNModel(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=Normalize(d=3),
            model=DummyDirectAcquisitionPFN(),
        )
        self.assertIsInstance(model.input_transform, Normalize)
        self.assertEqual(model.input_transform.bounds.shape, torch.Size([2, 3]))

        # Test that acquisition values still work with input transform
        test_X = torch.rand(5, 3)
        acq_values = model.get_acquisition_values(test_X)
        self.assertEqual(acq_values.shape, torch.Size([5]))
