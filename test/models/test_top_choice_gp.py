#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
import warnings
from copy import deepcopy
from unittest import mock

import torch
from botorch.acquisition.input_constructors import get_acqf_input_constructor
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.acquisition.preference import (
    AnalyticExpectedUtilityOfBestOption,
    qExpectedUtilityOfBestOption,
)
from botorch.exceptions import OptimizationWarning, UnsupportedError
from botorch.exceptions.warnings import _get_single_precision_warning, InputDataWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import (
    PairwiseGP,
    SingleTaskGP,
    TopChoiceVariationalELBO,
    VariationalTopChoiceGP,
)
from botorch.models.likelihoods import (
    PairwiseLogitLikelihood,
    TopChoiceSoftmaxLikelihood,
)
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize, Warp
from botorch.optim.utils import get_X_baseline
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.containers import SliceContainer
from botorch.utils.datasets import RankingDataset, SupervisedDataset
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ZeroMean
from gpytorch.priors import GammaPrior, LogNormalPrior
from linear_operator.utils.errors import NotPSDError
from torch import Tensor


def _random_block_covar(m: int, k: int, **tkwargs) -> Tensor:
    """Return an ``m x k x k`` tensor of random positive definite blocks."""
    A = torch.randn(m, k, k, **tkwargs)
    return A @ A.mT / k + 0.1 * torch.eye(k, **tkwargs)


class TestTopChoiceSoftmaxLikelihood(BotorchTestCase):
    def test_forward(self) -> None:
        for dtype, k, batch_shape in itertools.product(
            (torch.float, torch.double), (2, 4), (torch.Size(), torch.Size([3]))
        ):
            with self.subTest(dtype=dtype, k=k, batch_shape=batch_shape):
                likelihood = TopChoiceSoftmaxLikelihood(num_alternatives=k)
                utilities = torch.randn(
                    *batch_shape, 5 * k, device=self.device, dtype=dtype
                )
                dist = likelihood(utilities)
                self.assertEqual(dist.batch_shape, batch_shape + torch.Size([5]))
                self.assertAllClose(
                    dist.probs, utilities.view(*batch_shape, 5, k).softmax(dim=-1)
                )

    def test_expected_log_prob(self) -> None:
        for dtype in (torch.float, torch.double):
            tkwargs = {"device": self.device, "dtype": dtype}
            torch.manual_seed(0)
            likelihood = TopChoiceSoftmaxLikelihood(num_alternatives=2, seed=0)
            m = 6
            mean = 2 * torch.randn(m, 2, **tkwargs)
            block_covar = _random_block_covar(m, 2, **tkwargs)
            dist = MultivariateNormal(
                mean.flatten(), torch.block_diag(*block_covar.unbind(0))
            )
            observations = torch.randint(2, (m,), device=self.device)

            with self.subTest("deterministic with fixed seed", dtype=dtype):
                self.assertTrue(
                    torch.equal(
                        likelihood.expected_log_prob(observations, dist),
                        likelihood.expected_log_prob(observations, dist),
                    )
                )

            with self.subTest("independent of cross-comparison covariance"):
                # Only the covariances within each comparison affect the result.
                C = torch.randn(2 * m, 2 * m, **tkwargs)
                full_covar = C @ C.mT / (2 * m) + 0.1 * torch.eye(2 * m, **tkwargs)
                idx = torch.arange(2 * m, device=self.device).view(m, 2)
                block_diag_covar = torch.block_diag(
                    *full_covar[idx.unsqueeze(-1), idx.unsqueeze(-2)].unbind(0)
                )
                self.assertAllClose(
                    likelihood.expected_log_prob(
                        observations, MultivariateNormal(mean.flatten(), full_covar)
                    ),
                    likelihood.expected_log_prob(
                        observations,
                        MultivariateNormal(mean.flatten(), block_diag_covar),
                    ),
                )

            for k in (2, 3):
                with self.subTest("accuracy for batched inputs", dtype=dtype, k=k):
                    # The inputs and the reference are computed on the CPU with a
                    # fixed generator, so that they do not depend on the device.
                    generator = torch.Generator().manual_seed(0)
                    mean = torch.randn(2, 4, k, generator=generator, dtype=torch.double)
                    A = torch.randn(4, k, k, generator=generator, dtype=torch.double)
                    block_covar = A @ A.mT / k + 0.1 * torch.eye(k, dtype=torch.double)
                    observations = torch.tensor([0, 1, k - 1, 0])
                    # Monte Carlo reference with many i.i.d. samples.
                    z = torch.randn(
                        2**16, 2, 4, k, generator=generator, dtype=torch.double
                    )
                    samples = (
                        mean
                        + (torch.linalg.cholesky(block_covar) @ z[..., None])[..., 0]
                    )
                    expected = (
                        samples.log_softmax(dim=-1)
                        .gather(
                            dim=-1, index=observations.expand(2**16, 2, 4)[..., None]
                        )
                        .squeeze(-1)
                        .mean(dim=0)
                    )
                    dist = MultivariateNormal(
                        mean.flatten(-2).to(**tkwargs),
                        torch.block_diag(*block_covar.unbind(0))
                        .expand(2, 4 * k, 4 * k)
                        .to(**tkwargs),
                    )
                    log_prob = TopChoiceSoftmaxLikelihood(
                        num_alternatives=k, seed=0
                    ).expected_log_prob(observations.to(self.device), dist)
                    self.assertEqual(log_prob.shape, torch.Size([2, 4]))
                    self.assertAllClose(
                        log_prob, expected.to(**tkwargs), atol=2e-2, rtol=0
                    )

            with self.subTest("exact for degenerate distribution", dtype=dtype):
                mean = torch.tensor([[-50.0, 0.0], [3.0, 1.0]], **tkwargs)
                mean.requires_grad_(True)
                dist = MultivariateNormal(
                    mean.flatten(), 1e-8 * torch.eye(4, **tkwargs)
                )
                observations = torch.tensor([0, 1], device=self.device)
                log_prob = TopChoiceSoftmaxLikelihood(
                    num_alternatives=2
                ).expected_log_prob(observations, dist)
                expected = mean.detach().log_softmax(dim=-1)[[0, 1], [0, 1]]
                self.assertAllClose(log_prob, expected, atol=1e-3, rtol=1e-4)
                # The gradient does not vanish for very unlikely choices.
                log_prob[0].backward()
                self.assertAllClose(
                    mean.grad[0], torch.tensor([1.0, -1.0], **tkwargs), atol=1e-4
                )


class TestVariationalTopChoiceGP(BotorchTestCase):
    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings)
        self.tkwargs = {"device": self.device, "dtype": torch.double}

    def _make_data(
        self, n: int = 12, d: int = 2, m: int = 20, k: int = 2, **tkwargs
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return datapoints, comparisons with the winner by a known utility first,
        and the utility of the datapoints. The data are generated on the CPU, so that
        they do not depend on the device."""
        tkwargs = tkwargs or self.tkwargs
        datapoints = torch.rand(n, d, dtype=tkwargs["dtype"])
        utility = -(datapoints - 0.3).pow(2).sum(dim=-1)
        comparisons = torch.stack([torch.randperm(n)[:k] for _ in range(m)])
        order = utility[comparisons].argsort(dim=-1, descending=True)
        return (
            datapoints.to(**tkwargs),
            comparisons.gather(dim=-1, index=order).to(self.device),
            utility.to(**tkwargs),
        )

    def _fit(self, model: VariationalTopChoiceGP) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OptimizationWarning)
            fit_gpytorch_mll(TopChoiceVariationalELBO(model.likelihood, model))

    def _perturb(self, model: VariationalTopChoiceGP) -> None:
        """Set non-default hyperparameters and variational parameters."""
        model.covar_module.outputscale = 2.0
        model.covar_module.base_kernel.lengthscale = 0.3
        model.train()
        model(*model.train_inputs)  # initializes the variational distribution
        with torch.no_grad():
            for p in model.variational_strategy.parameters():
                p.add_(0.3 * torch.randn_like(p))

    @staticmethod
    def _trainable(model: VariationalTopChoiceGP) -> set[str]:
        return {name for name, p in model.named_parameters() if p.requires_grad}

    def test_init(self) -> None:
        torch.manual_seed(0)
        dtype = self.tkwargs["dtype"]
        for k in (2, 3):
            with self.subTest("defaults", k=k):
                datapoints, comparisons, _ = self._make_data(k=k)
                model = VariationalTopChoiceGP(datapoints, comparisons.to(dtype))
                self.assertEqual(model.num_outputs, 1)
                self.assertEqual(model.batch_shape, torch.Size())
                self.assertTrue(torch.equal(model.datapoints, datapoints))
                self.assertTrue(torch.equal(model.comparisons, comparisons))
                self.assertEqual(model.comparisons.dtype, torch.long)
                self.assertEqual(model.likelihood.num_alternatives, k)
                self.assertEqual(model.dim, 2)
                self.assertIsInstance(model.covar_module, ScaleKernel)
                self.assertIsInstance(model.covar_module.outputscale_prior, GammaPrior)
                self.assertIsInstance(model.mean_module, ZeroMean)
                self.assertTrue(
                    torch.equal(
                        model.train_inputs[0], datapoints[comparisons].flatten(0, 1)
                    )
                )
                self.assertTrue(
                    torch.equal(
                        model.train_targets,
                        torch.zeros(20, dtype=torch.long, device=self.device),
                    )
                )
                self.assertEqual(
                    model.covar_module.base_kernel.lengthscale.dtype, dtype
                )
                # The training data follow the model to another dtype.
                model.to(torch.half)
                self.assertEqual(model.train_inputs[0].dtype, torch.half)
                self.assertEqual(model.train_targets.dtype, torch.long)

        with self.subTest("inducing points"):
            # Datapoint 4 is never compared and datapoint 3 duplicates datapoint 0.
            datapoints = torch.rand(5, 2, **self.tkwargs)
            datapoints[3] = datapoints[0]
            comparisons = torch.tensor([[0, 1], [2, 3], [1, 2]], device=self.device)
            model = VariationalTopChoiceGP(datapoints, comparisons)
            inducing_points = model.variational_strategy.inducing_points
            self.assertEqual(inducing_points.shape, torch.Size([3, 2]))
            # The inducing points are the distinct compared datapoints.
            matches = (inducing_points.unsqueeze(-2) == datapoints[:3]).all(-1)
            self.assertTrue((matches.sum(dim=-1) == 1).all())
            self.assertTrue((matches.sum(dim=0) == 1).all())

        datapoints, comparisons, _ = self._make_data()
        with self.subTest("custom modules"):
            likelihood = TopChoiceSoftmaxLikelihood(num_alternatives=2, num_samples=8)
            covar_module = ScaleKernel(MaternKernel())
            model = VariationalTopChoiceGP(
                datapoints,
                comparisons,
                likelihood=likelihood,
                covar_module=covar_module,
            )
            self.assertIs(model.likelihood, likelihood)
            self.assertIs(model.covar_module, covar_module)

        with self.subTest("invalid inputs"):
            with self.assertRaisesRegex(UnsupportedError, "without data"):
                VariationalTopChoiceGP(None, None)
            with self.assertRaisesRegex(UnsupportedError, "batched datapoints"):
                VariationalTopChoiceGP(datapoints.unsqueeze(0), comparisons)
            for bad_comparisons in (
                comparisons[:, 0],
                comparisons[:, :1],
                comparisons[:0],
            ):
                with self.assertRaisesRegex(ValueError, "`m >= 1` and `k >= 2`"):
                    VariationalTopChoiceGP(datapoints, bad_comparisons)
            for bad_index in (-1, 12):
                bad_comparisons = comparisons.clone()
                bad_comparisons[0, 0] = bad_index
                with self.assertRaisesRegex(ValueError, "indices in"):
                    VariationalTopChoiceGP(datapoints, bad_comparisons)
            with self.assertRaisesRegex(ValueError, "num_alternatives"):
                VariationalTopChoiceGP(
                    datapoints,
                    comparisons,
                    likelihood=TopChoiceSoftmaxLikelihood(num_alternatives=3),
                )
            with self.assertRaisesRegex(UnsupportedError, "TopChoiceSoftmax"):
                VariationalTopChoiceGP(
                    datapoints, comparisons, likelihood=PairwiseLogitLikelihood()
                )
            with self.assertRaisesRegex(UnsupportedError, "ScaleKernel"):
                VariationalTopChoiceGP(
                    datapoints, comparisons, covar_module=MaternKernel()
                )

    def test_construct_inputs(self) -> None:
        datapoints = torch.rand(6, 2, **self.tkwargs)
        indices = torch.tensor([[0, 1, 2], [3, 4, 5], [1, 3, 5]], device=self.device)
        # Winners: item 1 (losers tied), item 5 (full ranking), and item 1.
        ranks = torch.tensor([[1, 0, 1], [2, 1, 0], [0, 2, 1]], device=self.device)
        dataset = RankingDataset(
            X=SliceContainer(
                values=datapoints, indices=indices, event_shape=torch.Size([6])
            ),
            Y=ranks,
            feature_names=["x1", "x2"],
            outcome_names=["choice"],
        )
        model_inputs = VariationalTopChoiceGP.construct_inputs(dataset)
        self.assertTrue(torch.equal(model_inputs["datapoints"], datapoints))
        self.assertEqual(model_inputs["comparisons"][:, 0].tolist(), [1, 5, 1])
        self.assertTrue(
            torch.equal(
                model_inputs["comparisons"].sort(dim=-1).values,
                indices.sort(dim=-1).values,
            )
        )
        # The inputs construct a model, as in Ax.
        self.assertIsInstance(
            VariationalTopChoiceGP(**model_inputs), VariationalTopChoiceGP
        )

        with self.subTest("pairwise rankings match PairwiseGP"):
            pairwise_dataset = RankingDataset(
                X=SliceContainer(
                    values=datapoints,
                    indices=indices[:, :2].contiguous(),
                    event_shape=torch.Size([4]),
                ),
                Y=torch.tensor([[1, 0], [0, 1], [1, 0]], device=self.device),
                feature_names=["x1", "x2"],
                outcome_names=["choice"],
            )
            self.assertTrue(
                torch.equal(
                    VariationalTopChoiceGP.construct_inputs(pairwise_dataset)[
                        "comparisons"
                    ],
                    PairwiseGP.construct_inputs(pairwise_dataset)["comparisons"],
                )
            )

        with self.subTest("invalid inputs"):
            tied_dataset = RankingDataset(
                X=SliceContainer(
                    values=datapoints, indices=indices, event_shape=torch.Size([6])
                ),
                Y=torch.tensor([[0, 0, 2], [2, 1, 0], [0, 2, 1]], device=self.device),
                feature_names=["x1", "x2"],
                outcome_names=["choice"],
            )
            with self.assertRaisesRegex(ValueError, "unique top-ranked"):
                VariationalTopChoiceGP.construct_inputs(tied_dataset)
            supervised_dataset = SupervisedDataset(
                X=datapoints,
                Y=datapoints[:, :1],
                feature_names=["x1", "x2"],
                outcome_names=["y"],
            )
            with self.assertRaisesRegex(UnsupportedError, "RankingDataset"):
                VariationalTopChoiceGP.construct_inputs(supervised_dataset)

    def test_posterior(self) -> None:
        torch.manual_seed(0)
        datapoints, comparisons, _ = self._make_data()
        model = VariationalTopChoiceGP(datapoints, comparisons)
        self._perturb(model)
        # Parameters that are not trainable stay so.
        model.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
        trainable = self._trainable(model)
        for batch_shape in (torch.Size(), torch.Size([3])):
            X = torch.rand(*batch_shape, 4, 2, **self.tkwargs)
            posterior = model.posterior(X)
            self.assertIsInstance(posterior, GPyTorchPosterior)
            self.assertEqual(posterior.mean.shape, batch_shape + torch.Size([4, 1]))
            self.assertTrue((posterior.variance > 0).all())
            # As for `PairwiseGP`, `observation_noise` is ignored.
            self.assertAllClose(
                model.posterior(X, observation_noise=True).variance,
                posterior.variance,
            )
        transformed = model.posterior(
            X,
            posterior_transform=ScalarizedPosteriorTransform(
                weights=torch.tensor([2.0], **self.tkwargs)
            ),
        )
        self.assertAllClose(transformed.mean, 2 * posterior.mean)
        with self.assertRaisesRegex(UnsupportedError, "output_indices"):
            model.posterior(X, output_indices=[0])

        self.assertEqual(self._trainable(model), trainable)

        with self.subTest("gradients and copies"):
            # As for `PairwiseGP`, gradients with respect to `X` can be computed
            # repeatedly, and the model can be deep-copied afterwards, also if the
            # caches of the model are first computed with gradients.
            model.train()
            for _ in range(2):
                X_grad = X.clone().requires_grad_(True)
                model.posterior(X_grad).variance.sum().backward()
                self.assertTrue((X_grad.grad != 0).any())
            self.assertAllClose(deepcopy(model).posterior(X).mean, posterior.mean)

        with self.subTest("utility"):
            model.train()
            utility = model.utility
            self.assertTrue(model.training)
            self.assertAllClose(utility, model.posterior(datapoints).mean[..., 0])

    def test_input_transform(self) -> None:
        # A model with a `Normalize` transform is equivalent to a model without a
        # transform on the normalized datapoints.
        bounds = torch.tensor([[-1.0, 0.0], [3.0, 10.0]], **self.tkwargs)
        torch.manual_seed(0)
        unit_datapoints, comparisons, _ = self._make_data()
        datapoints = bounds[0] + (bounds[1] - bounds[0]) * unit_datapoints
        model = VariationalTopChoiceGP(
            datapoints, comparisons, input_transform=Normalize(d=2, bounds=bounds)
        )
        self._perturb(model)
        unit_model = VariationalTopChoiceGP(unit_datapoints, comparisons)
        self.assertAllClose(model.train_inputs[0], unit_model.train_inputs[0])
        self.assertAllClose(
            model.variational_strategy.inducing_points,
            unit_model.variational_strategy.inducing_points,
        )
        unit_model.covar_module.load_state_dict(model.covar_module.state_dict())
        unit_model.variational_strategy.load_state_dict(
            model.variational_strategy.state_dict()
        )
        X_unit = torch.rand(4, 2, **self.tkwargs)
        X = bounds[0] + (bounds[1] - bounds[0]) * X_unit
        posterior = model.posterior(X)
        unit_posterior = unit_model.posterior(X_unit)
        self.assertAllClose(posterior.mean, unit_posterior.mean, atol=1e-5)
        self.assertAllClose(posterior.variance, unit_posterior.variance, atol=1e-5)

        with self.subTest("the posterior does not update the transform"):
            model = VariationalTopChoiceGP(
                datapoints, comparisons, input_transform=Normalize(d=2)
            )
            # The bounds are set from the datapoints at initialization.
            self.assertAllClose(
                model.input_transform.bounds,
                torch.stack([datapoints.amin(dim=0), datapoints.amax(dim=0)]),
            )
            coefficient = model.input_transform.coefficient.clone()
            model.posterior(2 * X)
            self.assertAllClose(model.input_transform.coefficient, coefficient)

        with self.subTest("learnable transform"):
            # As for `PairwiseGP`, the parameters of the transform are fit with the
            # other hyperparameters: the ELBO depends on them through the training
            # inputs, which follow the transform.
            warp = Warp(
                d=2,
                indices=[0, 1],
                concentration0_prior=LogNormalPrior(0.0, 0.75**0.5),
                concentration1_prior=LogNormalPrior(0.0, 0.75**0.5),
            )
            model = VariationalTopChoiceGP(
                unit_datapoints, comparisons, input_transform=warp
            )
            model.train()
            output = model(*model.train_inputs)
            expected_log_prob = model.likelihood.expected_log_prob(
                model.train_targets, output
            ).sum()
            for grad in torch.autograd.grad(
                expected_log_prob, [warp.concentration0, warp.concentration1]
            ):
                self.assertTrue((grad != 0).all())
            with torch.no_grad():
                warp.concentration0.fill_(0.5)
            model.eval()
            self.assertAllClose(
                model.train_inputs[0],
                warp(unit_datapoints[comparisons].flatten(0, 1)),
            )
            self.assertAllClose(
                model.utility, model.posterior(unit_datapoints).mean[..., 0]
            )
            # Conditioning on datapoints outside of the range of the datapoints does
            # not update the transform, including the `Normalize` of `Warp`.
            new_model = model.condition_on_observations(
                X=torch.tensor([[-1.0, 2.0], [2.0, -1.0]], **self.tkwargs),
                Y=torch.tensor([[0, 1]], device=self.device),
            )
            self.assertAllClose(
                new_model.train_inputs[0][: comparisons.numel()],
                model.train_inputs[0],
            )

    def test_full_covariance_in_train_mode(self) -> None:
        # GPyTorch only computes the predictive variances in train mode if there are
        # fewer inducing points than inputs, unless `diag=False`.
        torch.manual_seed(0)
        datapoints, comparisons, _ = self._make_data(n=5, m=10)
        model = VariationalTopChoiceGP(datapoints, comparisons)
        X = model.train_inputs[0]
        self.assertLess(model.variational_strategy.inducing_points.shape[-2], len(X))
        model.train()
        model(X)  # initializes the variational distribution
        variational_distribution = model.variational_strategy._variational_distribution
        with torch.no_grad():
            variational_distribution.variational_mean.normal_()
            variational_distribution.chol_variational_covar.mul_(0.5)
        train_covar = model(X).covariance_matrix
        model.eval()
        self.assertAllClose(train_covar, model(X).covariance_matrix)

    def test_fit(self) -> None:
        torch.manual_seed(0)
        datapoints, comparisons, utility = self._make_data(n=15, m=30, k=3)
        model = VariationalTopChoiceGP(datapoints, comparisons)
        # Keyword arguments as passed by Ax.
        mll = TopChoiceVariationalELBO(
            likelihood=model.likelihood, model=model, beta=0.5
        )
        self.assertEqual(mll.beta, 0.5)
        self.assertEqual(mll.num_data, comparisons.numel())
        mll = TopChoiceVariationalELBO(model.likelihood, model)
        model.train()
        output = model(*model.train_inputs)
        initial_elbo = mll(output, model.train_targets)
        # The objective is the ELBO plus the log prior, divided by `m * k`.
        log_prior = sum(
            prior.log_prob(closure(module)).sum()
            for _, module, prior, closure, _ in mll.named_priors()
        )
        expected_log_prob = model.likelihood.expected_log_prob(
            model.train_targets, output
        ).sum()
        kl = model.variational_strategy.kl_divergence()
        self.assertAllClose(
            initial_elbo, (expected_log_prob - kl + log_prior) / comparisons.numel()
        )
        # Other approximate GPs are fit with Adam above `full_batch_limit`.
        with (
            mock.patch("botorch.fit.fit_gpytorch_mll_torch") as fit_torch,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", category=OptimizationWarning)
            fit_gpytorch_mll(mll, full_batch_limit=1)
        fit_torch.assert_not_called()
        model.train()
        fitted_elbo = mll(model(*model.train_inputs), model.train_targets)
        self.assertGreater(fitted_elbo.item(), initial_elbo.item())
        fitted_utility = model.utility
        self.assertGreater(
            torch.corrcoef(torch.stack([fitted_utility, utility]))[0, 1], 0.7
        )
        winner_utility = fitted_utility[comparisons[:, 0]]
        self.assertGreater(
            (winner_utility.unsqueeze(-1) > fitted_utility[comparisons[:, 1:]])
            .all(dim=-1)
            .double()
            .mean(),
            0.8,
        )
        # The Laplace objective of `PairwiseGP` does not apply.
        with self.assertRaisesRegex(UnsupportedError, "variational objective"):
            fit_gpytorch_mll(
                PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
            )

        with self.subTest("numerical errors in the warm start are retried"):
            with mock.patch.object(
                model,
                "_fit_variational_distribution",
                side_effect=itertools.chain(
                    [NotPSDError("not p.d.")], itertools.repeat(None)
                ),
            ) as warm_start:
                self._fit(model)
            self.assertGreaterEqual(warm_start.call_count, 2)
            self.assertFalse(model.training)

        with self.subTest("frozen variational distribution"):
            # Only the hyperparameters are fit.
            for p in model.variational_strategy.parameters():
                p.requires_grad_(False)
            variational_state = deepcopy(model.variational_strategy.state_dict())
            model.covar_module.base_kernel.lengthscale = 1.0
            self._fit(model)
            self.assertFalse(
                torch.allclose(
                    model.covar_module.base_kernel.lengthscale,
                    torch.ones(1, 2, **self.tkwargs),
                )
            )
            for name, value in model.variational_strategy.state_dict().items():
                self.assertTrue(torch.equal(value, variational_state[name]))
            self.assertFalse(
                any(p.requires_grad for p in model.variational_strategy.parameters())
            )

        with self.subTest("short lengthscale"):
            # A joint fit that starts from the prior ends at an over-smoothed optimum
            # with a nearly constant utility here, unlike the warm-started fit. The
            # model is in double precision since in single precision, L-BFGS-B can
            # stop at the optimum with an abnormal termination, after which the
            # retries of `fit_gpytorch_mll` can end at the over-smoothed optimum.
            generator = torch.Generator().manual_seed(9)
            datapoints = torch.rand(30, 2, generator=generator, dtype=torch.double)
            utility = 2 * torch.sin(2 * math.pi * datapoints.sum(dim=-1))
            comparisons = torch.rand(40, 30, generator=generator).argsort(dim=-1)
            comparisons = comparisons[:, :3]
            chosen = torch.multinomial(
                utility[comparisons].softmax(dim=-1), 1, generator=generator
            )
            # Move the chosen datapoint of each comparison to the front.
            order = (torch.arange(3) != chosen).long().argsort(dim=-1, stable=True)
            model = VariationalTopChoiceGP(
                datapoints.to(self.device),
                comparisons.gather(dim=-1, index=order).to(self.device),
            )
            self._fit(model)
            fitted_utility = model.utility.cpu()
            self.assertGreater(
                torch.corrcoef(torch.stack([fitted_utility, utility]))[0, 1], 0.8
            )

    def test_pairwise_compatibility(self) -> None:
        # With the same kernel hyperparameters, the variational posterior is close to
        # the Laplace posterior of `PairwiseGP` with the same (logit) likelihood.
        torch.manual_seed(0)
        datapoints, comparisons, _ = self._make_data(n=20, m=40)
        pairwise_model = PairwiseGP(
            datapoints, comparisons, likelihood=PairwiseLogitLikelihood()
        )
        covar_module = deepcopy(pairwise_model.covar_module)
        pairwise_model.covar_module.outputscale = 1.0
        pairwise_model.covar_module.base_kernel.lengthscale = 0.3
        pairwise_model.set_train_data(datapoints, comparisons, update_model=True)
        model = VariationalTopChoiceGP(
            datapoints, comparisons, covar_module=covar_module
        )
        # Loading the hyperparameters of the `PairwiseGP` fits the variational
        # distribution with these hyperparameters.
        model.load_state_dict(pairwise_model.state_dict())
        self.assertAllClose(
            model.covar_module.outputscale, torch.tensor(1.0, **self.tkwargs)
        )
        pairwise_posterior = pairwise_model.posterior(datapoints)
        posterior = model.posterior(datapoints)
        self.assertAllClose(
            posterior.mean, pairwise_posterior.mean, rtol=0.1, atol=0.05
        )
        self.assertAllClose(
            posterior.variance.mean(), pairwise_posterior.variance.mean(), rtol=0.1
        )

    def test_preference_acquisition(self) -> None:
        # As `PairwiseGP`, the model can be the preference model of BOPE, whose
        # datapoints are outcomes of an outcome model.
        torch.manual_seed(0)
        bounds = torch.tensor([[-1.0, 0.0], [3.0, 10.0]], **self.tkwargs)
        unit_datapoints, comparisons, _ = self._make_data()
        datapoints = bounds[0] + (bounds[1] - bounds[0]) * unit_datapoints
        model = VariationalTopChoiceGP(
            datapoints, comparisons, input_transform=Normalize(d=2, bounds=bounds)
        )
        outcome_model = SingleTaskGP(
            torch.rand(8, 3, **self.tkwargs), torch.rand(8, 2, **self.tkwargs)
        )
        X = torch.rand(5, 2, 3, **self.tkwargs)
        for acqf_class in (
            AnalyticExpectedUtilityOfBestOption,
            qExpectedUtilityOfBestOption,
        ):
            with self.subTest(acqf_class=acqf_class.__name__):
                acqf = acqf_class(
                    **get_acqf_input_constructor(acqf_class)(
                        model=outcome_model, pref_model=model
                    )
                )
                self.assertEqual(acqf(X).shape, torch.Size([5]))
        # The baseline points for initializing the optimization of preference
        # acquisition functions are in the original space.
        self.assertAllClose(
            get_X_baseline(AnalyticExpectedUtilityOfBestOption(pref_model=model)),
            datapoints[comparisons].flatten(0, 1),
        )

    def test_condition_on_observations(self) -> None:
        torch.manual_seed(0)
        datapoints, comparisons, _ = self._make_data(m=10, k=3)
        model = VariationalTopChoiceGP(datapoints, comparisons)
        self._fit(model)
        X = torch.tensor([[0.05, 0.95], [0.95, 0.05]], **self.tkwargs)
        # Evaluate the posterior with gradients, which must not prevent copying the
        # model.
        model.posterior(X.requires_grad_(True)).mean.sum().backward()
        self.assertIsNotNone(X.grad)
        Y = torch.tensor([[1, 0, 3]], device=self.device, dtype=torch.int32)
        X_new = torch.cat([X.detach(), datapoints[:2]])
        new_model = model.condition_on_observations(X_new, Y)
        self.assertTrue(torch.equal(new_model.datapoints[12:], X_new))
        self.assertEqual(new_model.comparisons[-1].tolist(), [13, 12, 15])
        self.assertTrue(torch.equal(new_model.comparisons[:10], comparisons))
        self.assertEqual(model.comparisons.shape, torch.Size([10, 3]))
        self.assertFalse(new_model.training)
        for name, param in model.covar_module.named_parameters():
            self.assertAllClose(param, new_model.covar_module.get_parameter(name))
        # The hyperparameters can be fit again.
        self.assertEqual(self._trainable(new_model), self._trainable(model))
        utility = new_model.utility
        old_utility = model.posterior(X_new).mean.squeeze(-1)
        self.assertGreater(utility[13] - utility[12], old_utility[1] - old_utility[0])
        for bad_Y in (
            Y.double(),
            Y[:, :2],
            Y.unsqueeze(0),
        ):
            with self.assertRaisesRegex(UnsupportedError, "can only be conditioned"):
                model.condition_on_observations(X_new, bad_Y)
        with self.assertRaisesRegex(UnsupportedError, "can only be conditioned"):
            model.condition_on_observations(X_new.unsqueeze(0), Y)
        for bad_index in (-1, 4):
            with self.assertRaisesRegex(ValueError, "indices in"):
                model.condition_on_observations(X_new, Y.clone().fill_(bad_index))

        with self.subTest("unsuccessful refit"):
            # The result of an unsuccessful refit is kept with a warning, rather than
            # discarded with an error as by `fit_gpytorch_mll`.
            with (
                mock.patch(
                    "botorch.models.top_choice_gp.fit_gpytorch_mll_scipy",
                    side_effect=lambda mll: warnings.warn(
                        "failed", OptimizationWarning, stacklevel=2
                    ),
                ),
                self.assertWarns(OptimizationWarning),
            ):
                model.condition_on_observations(X_new, Y)

        with self.subTest("the model settings are kept"):
            likelihood = TopChoiceSoftmaxLikelihood(
                num_alternatives=3, num_samples=64, seed=1
            )
            model = VariationalTopChoiceGP(
                datapoints,
                comparisons,
                likelihood=likelihood,
                input_transform=Normalize(d=2),
            )
            # In train mode, in which the input transform is updated at
            # initialization, it is not updated with the new datapoints.
            new_model = model.train().condition_on_observations(1 + X_new, Y)
            self.assertAllClose(
                new_model.input_transform.coefficient,
                model.input_transform.coefficient,
            )
            self.assertEqual(new_model.likelihood.num_samples, 64)
            self.assertEqual(new_model.likelihood.seed, 1)
            # The inducing points are the compared datapoints of the augmented data.
            self.assertAllClose(
                new_model.variational_strategy.inducing_points,
                new_model.train_inputs[0].unique(dim=0),
            )

    def test_load_state_dict(self) -> None:
        torch.manual_seed(0)
        datapoints, comparisons, _ = self._make_data(m=20)
        model = VariationalTopChoiceGP(
            datapoints, comparisons, input_transform=Normalize(d=2)
        )
        self._fit(model)
        state_dict = model.state_dict()

        with self.subTest("same data"):
            new_model = VariationalTopChoiceGP(
                datapoints, comparisons, input_transform=Normalize(d=2)
            )
            new_model.load_state_dict(state_dict, strict=True)
            self.assertAllClose(new_model.utility, model.utility)

        with self.subTest("different data"):
            # E.g., cross-validation with the hyperparameters of the full model.
            fold_datapoints = datapoints[:10]
            fold_comparisons = comparisons[(comparisons < 10).all(dim=-1)]
            new_model = VariationalTopChoiceGP(
                fold_datapoints, fold_comparisons, input_transform=Normalize(d=2)
            )
            with self.assertRaisesRegex(UnsupportedError, "strict=True"):
                new_model.load_state_dict(state_dict, strict=True)
            new_model.load_state_dict(state_dict)
            self.assertAllClose(
                new_model.covar_module.outputscale, model.covar_module.outputscale
            )
            # The training inputs and the inducing points use the loaded transform.
            self.assertAllClose(
                new_model.train_inputs[0],
                model.input_transform(fold_datapoints[fold_comparisons].flatten(0, 1)),
            )
            self.assertAllClose(
                new_model.variational_strategy.inducing_points,
                new_model.train_inputs[0].unique(dim=0),
            )
            self.assertGreater(
                torch.corrcoef(torch.stack([new_model.utility, model.utility[:10]]))[
                    0, 1
                ],
                0.8,
            )
            # The hyperparameters can be fit again, also if the refit fails.
            self.assertEqual(self._trainable(new_model), self._trainable(model))
            with (
                mock.patch(
                    "botorch.models.top_choice_gp.fit_gpytorch_mll_scipy",
                    side_effect=RuntimeError,
                ),
                self.assertRaises(RuntimeError),
            ):
                new_model.load_state_dict(state_dict)
            self.assertEqual(self._trainable(new_model), self._trainable(model))

        with self.subTest("different comparisons"):
            # The same datapoints with the opposite choices.
            new_model = VariationalTopChoiceGP(
                datapoints, comparisons.flip(-1), input_transform=Normalize(d=2)
            )
            with self.assertRaisesRegex(UnsupportedError, "strict=True"):
                new_model.load_state_dict(state_dict, strict=True)
            new_model.load_state_dict(state_dict)
            self.assertAllClose(
                new_model.covar_module.outputscale, model.covar_module.outputscale
            )
            self.assertLess(
                torch.corrcoef(torch.stack([new_model.utility, model.utility]))[0, 1],
                -0.9,
            )


class TestVariationalTopChoiceGP_float32(TestVariationalTopChoiceGP):
    """Runs the tests of ``TestVariationalTopChoiceGP`` in single precision."""

    def setUp(self, suppress_input_warnings: bool = True) -> None:
        super().setUp(suppress_input_warnings)
        self.tkwargs["dtype"] = torch.float

    def test_init_warns_on_single_precision(self) -> None:
        datapoints, comparisons, _ = self._make_data()
        with self.assertWarnsRegex(
            InputDataWarning,
            expected_regex=_get_single_precision_warning(str(torch.float32)),
        ):
            VariationalTopChoiceGP(datapoints, comparisons)
