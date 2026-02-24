#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from unittest import mock
from unittest.mock import patch

import torch
from botorch.acquisition.objective import (
    ExpectationPosteriorTransform,
    GenericMCObjective,
    LearnedObjective,
    LinearMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch.acquisition.utils import (
    compute_best_feasible_objective,
    expand_trace_observations,
    get_acquisition_function,
    get_infeasible_cost,
    get_optimal_samples,
    project_to_sample_points,
    project_to_target_fidelity,
    prune_inferior_points,
    repeat_to_match_aug_dim,
)
from botorch.exceptions.errors import (
    BotorchTensorDimensionError,
    DeprecationError,
    UnsupportedError,
)
from botorch.exceptions.warnings import BotorchWarning
from botorch.models import SingleTaskGP
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from gpytorch.distributions import MultivariateNormal


class TestGetAcquisitionFunctionDeprecation(BotorchTestCase):
    def test_get_acquisition_function_deprecation(self):
        msg = (
            "`get_acquisition_function` has been moved to"
            " `botorch.acquisition.factory`."
        )
        with self.assertRaisesRegex(DeprecationError, msg):
            get_acquisition_function()


class TestConstraintUtils(BotorchTestCase):
    def test_compute_best_feasible_objective(self):
        for dtype in (torch.float, torch.double):
            with self.subTest(dtype=dtype):
                tkwargs = {"dtype": dtype, "device": self.device}
                n = 5
                X = torch.arange(n, **tkwargs).view(-1, 1)
                for batch_shape, sample_shape in itertools.product(
                    (torch.Size([]), torch.Size([2])),
                    (torch.Size([1]), torch.Size([3])),
                ):
                    means = torch.arange(n, **tkwargs).view(-1, 1)
                    if len(batch_shape) > 0:
                        view_means = means.view(1, *means.shape)
                        means = view_means.expand(batch_shape + means.shape)
                    if sample_shape[0] == 1:
                        samples = means.unsqueeze(0)
                    else:
                        samples = torch.stack([means, means + 1, means + 4], dim=0)
                    variances = torch.tensor(
                        [0.09, 0.25, 0.36, 0.25, 0.09], **tkwargs
                    ).view(-1, 1)
                    mm = MockModel(MockPosterior(mean=means, variance=variances))

                    # testing all feasible points
                    obj = samples.squeeze(-1)
                    constraints = [lambda samples: -torch.ones_like(samples[..., 0])]
                    best_f = compute_best_feasible_objective(
                        samples=samples, obj=obj, constraints=constraints
                    )
                    self.assertAllClose(best_f, obj.amax(dim=-1, keepdim=False))

                    # testing with some infeasible points
                    con_cutoff = 3.0
                    best_f = compute_best_feasible_objective(
                        samples=samples,
                        obj=obj,
                        constraints=[
                            lambda samples, cc=con_cutoff: samples[..., 0]
                            - (cc + 1 / 2)
                        ],
                        model=mm,
                        X_baseline=X,
                    )

                    if sample_shape[0] == 3:
                        # under some samples, all baseline points are infeasible, so
                        # the best_f is set to the negative infeasible cost for
                        # for samples where no point is feasible
                        expected_best_f = torch.tensor(
                            [
                                3.0,
                                3.0,
                                -get_infeasible_cost(
                                    X=X,
                                    model=mm,
                                ).item(),
                            ],
                            **tkwargs,
                        )
                        # When ``batch_shape = (b,)``, this expands ``expected_best_f``
                        # from shape (3, 1) to (3, 1, 1), then to
                        # (1, 1, ..., 1, 3, b, 1), where there are
                        # ``len(sample_shape) - 1`` leading ones.
                        if len(batch_shape) > 0:
                            expected_best_f = expected_best_f.unsqueeze(1).repeat(
                                *[1] * len(sample_shape), *batch_shape
                            )
                    else:
                        expected_best_f = torch.full(
                            sample_shape + batch_shape,
                            con_cutoff,
                            **tkwargs,
                        )
                    self.assertAllClose(best_f, expected_best_f)
                    # test some feasible points with infeasible obi
                    if sample_shape[0] == 3:
                        best_f = compute_best_feasible_objective(
                            samples=samples,
                            obj=obj,
                            constraints=[
                                lambda samples, cc=con_cutoff: samples[..., 0]
                                - (cc + 1 / 2)
                            ],
                            infeasible_obj=torch.ones(1, **tkwargs),
                        )
                        expected_best_f[-1] = 1
                        self.assertAllClose(best_f, expected_best_f)

                    # testing with no feasible points and infeasible obj
                    infeasible_obj = torch.tensor(torch.pi, **tkwargs)
                    expected_best_f = torch.full(
                        sample_shape + batch_shape,
                        torch.pi,
                        **tkwargs,
                    )

                    best_f = compute_best_feasible_objective(
                        samples=samples,
                        obj=obj,
                        constraints=[lambda X: torch.ones_like(X[..., 0])],
                        infeasible_obj=infeasible_obj,
                    )
                    self.assertAllClose(best_f, expected_best_f)

                    # testing with no feasible points and not infeasible obj
                    def objective(Y, X):
                        return Y.squeeze(-1) - 5.0

                    with self.assertWarnsRegex(
                        BotorchWarning, "ProbabilityOfFeasibility"
                    ):
                        best_f = compute_best_feasible_objective(
                            samples=samples,
                            obj=obj,
                            constraints=[lambda X: torch.ones_like(X[..., 0])],
                            model=mm,
                            X_baseline=X,
                            objective=objective,
                        )
                    expected_best_f = torch.full(
                        sample_shape + batch_shape,
                        -get_infeasible_cost(X=X, model=mm, objective=objective).item(),
                        **tkwargs,
                    )
                    self.assertAllClose(best_f, expected_best_f)

                    with self.assertRaisesRegex(ValueError, "Must specify `model`"):
                        best_f = compute_best_feasible_objective(
                            samples=means,
                            obj=obj,
                            constraints=[lambda X: torch.ones_like(X[..., 0])],
                            X_baseline=X,
                        )
                    with self.assertRaisesRegex(
                        ValueError, "Must specify `X_baseline`"
                    ):
                        best_f = compute_best_feasible_objective(
                            samples=means,
                            obj=obj,
                            constraints=[lambda X: torch.ones_like(X[..., 0])],
                            model=mm,
                        )

    def test_get_infeasible_cost(self):
        for dtype in (torch.float, torch.double):
            tkwargs = {"dtype": dtype, "device": self.device}
            X = torch.ones(5, 1, **tkwargs)
            means = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], **tkwargs).view(-1, 1)
            variances = torch.tensor([0.09, 0.25, 0.36, 0.25, 0.09], **tkwargs).view(
                -1, 1
            )
            mm = MockModel(MockPosterior(mean=means, variance=variances))
            # means - 6 * std = [-0.8, -1, -0.6, 1, 3.2]. After applying the
            # objective, the minimum becomes -6.0, so 6.0 should be returned.
            M = get_infeasible_cost(
                X=X, model=mm, objective=lambda Y, X: Y.squeeze(-1) - 5.0
            )
            self.assertAllClose(M, torch.tensor([6.0], **tkwargs))
            M = get_infeasible_cost(
                X=X, model=mm, objective=lambda Y, X: Y.squeeze(-1) - 5.0 - X[0, 0]
            )
            self.assertAllClose(M, torch.tensor([7.0], **tkwargs))
            # test it with using also X in the objective
            # Test default objective (squeeze last dim).
            M2 = get_infeasible_cost(X=X, model=mm)
            self.assertAllClose(M2, torch.tensor([1.0], **tkwargs))
            # Test multi-output.
            m_ = means.repeat(1, 2)
            m_[:, 1] -= 10
            mm = MockModel(MockPosterior(mean=m_, variance=variances.expand(-1, 2)))
            M3 = get_infeasible_cost(X=X, model=mm)
            self.assertAllClose(M3, torch.tensor([1.0, 11.0], **tkwargs))
            # With a batched model.
            means = means.expand(2, 4, -1, -1)
            variances = variances.expand(2, 4, -1, -1)
            mm = MockModel(MockPosterior(mean=means, variance=variances))
            M4 = get_infeasible_cost(X=X, model=mm)
            self.assertAllClose(M4, torch.tensor([1.0], **tkwargs))


class TestPruneInferiorPoints(BotorchTestCase):
    def test_prune_inferior_points(self):
        for dtype in (torch.float, torch.double):
            X = torch.rand(3, 2, device=self.device, dtype=dtype)
            # the event shape is ``q x t`` = 3 x 1
            samples = torch.tensor(
                [[-1.0], [0.0], [1.0]], device=self.device, dtype=dtype
            )
            mm = MockModel(MockPosterior(samples=samples))
            # test that a batched X raises errors
            with self.assertRaises(UnsupportedError):
                prune_inferior_points(model=mm, X=X.expand(2, 3, 2))
            # test marginalize_dim
            mm2 = MockModel(MockPosterior(samples=samples.expand(2, 3, 1)))
            X_pruned = prune_inferior_points(model=mm2, X=X, marginalize_dim=-3)
            with self.assertRaises(UnsupportedError):
                # test error raised when marginalize_dim is not specified with
                # a batch model
                prune_inferior_points(model=mm2, X=X)
            self.assertTrue(torch.equal(X_pruned, X[[-1]]))
            # test that a batched model raises errors when there are multiple batch dims
            mm2 = MockModel(MockPosterior(samples=samples.expand(1, 2, 3, 1)))
            with self.assertRaises(UnsupportedError):
                prune_inferior_points(model=mm2, X=X)
            # test that invalid max_frac is checked properly
            with self.assertRaises(ValueError):
                prune_inferior_points(model=mm, X=X, max_frac=1.1)
            # test that invalid X is checked properly
            with self.assertRaises(ValueError):
                prune_inferior_points(model=mm, X=torch.empty(0, 0))
            # test basic behaviour
            X_pruned = prune_inferior_points(model=mm, X=X)
            self.assertTrue(torch.equal(X_pruned, X[[-1]]))
            # test custom objective
            neg_id_obj = GenericMCObjective(lambda Y, X: -(Y.squeeze(-1)))
            X_pruned = prune_inferior_points(model=mm, X=X, objective=neg_id_obj)
            self.assertTrue(torch.equal(X_pruned, X[[0]]))
            # test non-repeated samples (requires mocking out MockPosterior's rsample)
            samples = torch.tensor(
                [[[3.0], [0.0], [0.0]], [[0.0], [2.0], [0.0]], [[0.0], [0.0], [1.0]]],
                device=self.device,
                dtype=dtype,
            )
            with mock.patch.object(MockPosterior, "rsample", return_value=samples):
                mm = MockModel(MockPosterior(samples=samples))
                X_pruned = prune_inferior_points(model=mm, X=X)
            self.assertTrue(torch.equal(X_pruned, X))
            # test max_frac limiting
            with mock.patch.object(MockPosterior, "rsample", return_value=samples):
                mm = MockModel(MockPosterior(samples=samples))
                X_pruned = prune_inferior_points(model=mm, X=X, max_frac=2 / 3)
            self.assertTrue(
                torch.equal(
                    torch.sort(X_pruned, stable=True).values,
                    torch.sort(X[:2], stable=True).values,
                )
            )
            # test that zero-probability is in fact pruned
            samples[2, 0, 0] = 10
            with mock.patch.object(MockPosterior, "rsample", return_value=samples):
                mm = MockModel(MockPosterior(samples=samples))
                X_pruned = prune_inferior_points(model=mm, X=X)
            self.assertTrue(torch.equal(X_pruned, X[:2]))
            # test constraints
            constraints = [lambda Y: Y[..., 1] + 0.1]
            # only the last sample if feasible and it has the worst objective value
            samples = torch.tensor(
                [[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]],
                device=self.device,
                dtype=dtype,
            )
            mm = MockModel(MockPosterior(samples=samples))
            X_pruned = prune_inferior_points(
                model=mm,
                X=X,
                objective=GenericMCObjective(objective=lambda Y, X: Y[..., 0]),
                constraints=constraints,
            )
            self.assertTrue(torch.equal(X_pruned, X[[-1]]))


class TestFidelityUtils(BotorchTestCase):
    def test_project_to_target_fidelity(self):
        for batch_shape, dtype in itertools.product(
            ([], [2]), (torch.float, torch.double)
        ):
            X = torch.rand(*batch_shape, 3, 4, device=self.device, dtype=dtype)
            ones = torch.ones(*X.shape[:-1], 1, device=self.device, dtype=dtype)
            # test single target fidelity
            target_fids = {3: 1.0}
            X_proj = project_to_target_fidelity(X, target_fidelities=target_fids, d=4)
            self.assertTrue(torch.equal(X_proj[..., :, [-1]], ones))
            self.assertTrue(torch.equal(X_proj[..., :-1], X[..., :-1]))
            # test custom target fidelity
            target_fids = {2: 0.5}
            X_proj = project_to_target_fidelity(X, target_fidelities=target_fids, d=4)
            self.assertTrue(torch.equal(X_proj[..., :, [2]], 0.5 * ones))
            # test multiple target fidelities
            target_fids = {2: 0.5, 0: 0.1}
            X_proj = project_to_target_fidelity(X, target_fidelities=target_fids, d=4)
            self.assertTrue(torch.equal(X_proj[..., :, [0]], 0.1 * ones))
            self.assertTrue(torch.equal(X_proj[..., :, [2]], 0.5 * ones))
            # test gradients
            X.requires_grad_(True)
            X_proj = project_to_target_fidelity(X, target_fidelities=target_fids, d=4)
            out = (X_proj**2).sum()
            out.backward()
            self.assertTrue(torch.all(X.grad[..., [0, 2]] == 0))
            self.assertTrue(torch.equal(X.grad[..., [1, 3]], 2 * X[..., [1, 3]]))
            # test X without fidelity dims
            X_proj = project_to_target_fidelity(
                X[..., :2], target_fidelities=target_fids, d=4
            )
            self.assertTrue(torch.equal(X_proj[..., :, [0]], 0.1 * ones))
            self.assertTrue(torch.equal(X_proj[..., :, [2]], 0.5 * ones))
            # test unexpected shape
            msg = "X must have a last dimension with size `d` or `d-d_f`, but got 3."
            with self.assertRaisesRegex(BotorchTensorDimensionError, msg):
                project_to_target_fidelity(
                    X[..., :3], target_fidelities=target_fids, d=4
                )
            # test positive index out of bounds
            with self.assertRaisesRegex(
                BotorchTensorDimensionError,
                "Target fidelity indices.*out of bounds",
            ):
                project_to_target_fidelity(X, target_fidelities={10: 1.0}, d=4)
            # test negative index works
            X_proj = project_to_target_fidelity(X, target_fidelities={-1: 0.5}, d=4)
            self.assertTrue(torch.equal(X_proj[..., :, [-1]], 0.5 * ones))
            # test negative index resolving out of bounds
            with self.assertRaisesRegex(
                BotorchTensorDimensionError,
                "Target fidelity indices.*out of bounds",
            ):
                project_to_target_fidelity(X, target_fidelities={-10: 1.0}, d=4)
            # test X without fidelity dims, inserting fidelity columns
            X_no_fidelity = torch.rand(
                *batch_shape, 3, 3, device=self.device, dtype=dtype
            )
            X_proj = project_to_target_fidelity(
                X_no_fidelity, target_fidelities={3: 1.0}, d=4
            )
            self.assertEqual(X_proj.shape[-1], 4)
            self.assertTrue(torch.equal(X_proj[..., :, [3]], ones))

    def test_expand_trace_observations(self):
        for batch_shape, dtype in itertools.product(
            ([], [2]), (torch.float, torch.double)
        ):
            q, d = 3, 4
            X = torch.rand(*batch_shape, q, d, device=self.device, dtype=dtype)
            # test nullop behavior
            self.assertTrue(torch.equal(expand_trace_observations(X), X))
            self.assertTrue(
                torch.equal(expand_trace_observations(X, fidelity_dims=[1]), X)
            )
            # test default behavior
            num_tr = 2
            X_expanded = expand_trace_observations(X, num_trace_obs=num_tr)
            self.assertEqual(
                X_expanded.shape, torch.Size(batch_shape + [q * (1 + num_tr), d])
            )
            for i in range(num_tr):
                X_sub = X_expanded[..., q * i : q * (i + 1), :]
                self.assertTrue(torch.equal(X_sub[..., :-1], X[..., :-1]))
                X_sub_expected = (1 - i / (num_tr + 1)) * X[..., :q, -1]
                self.assertTrue(torch.equal(X_sub[..., -1], X_sub_expected))
            # test custom fidelity dims
            fdims = [0, 2]
            num_tr = 3
            X_expanded = expand_trace_observations(
                X, fidelity_dims=fdims, num_trace_obs=num_tr
            )
            self.assertEqual(
                X_expanded.shape, torch.Size(batch_shape + [q * (1 + num_tr), d])
            )
            for j, i in itertools.product([1, 3], range(num_tr)):
                X_sub = X_expanded[..., q * i : q * (i + 1), j]
                self.assertTrue(torch.equal(X_sub, X[..., j]))
            for j, i in itertools.product(fdims, range(num_tr)):
                X_sub = X_expanded[..., q * i : q * (i + 1), j]
                X_sub_expected = (1 - i / (1 + num_tr)) * X[..., :q, j]
                self.assertTrue(torch.equal(X_sub, X_sub_expected))
            # test gradients
            num_tr = 2
            fdims = [1]
            X.requires_grad_(True)
            X_expanded = expand_trace_observations(
                X, fidelity_dims=fdims, num_trace_obs=num_tr
            )
            out = X_expanded.sum()
            out.backward()
            grad_exp = torch.full_like(X, 1 + num_tr)
            grad_exp[..., fdims] = 1 + sum(
                (i + 1) / (num_tr + 1) for i in range(num_tr)
            )
            self.assertAllClose(X.grad, grad_exp)

    def test_project_to_sample_points(self):
        for batch_shape, dtype in itertools.product(
            ([], [2]), (torch.float, torch.double)
        ):
            q, d, p, d_prime = 1, 12, 7, 4
            X = torch.rand(*batch_shape, q, d, device=self.device, dtype=dtype)
            sample_points = torch.rand(p, d_prime, device=self.device, dtype=dtype)
            X_augmented = project_to_sample_points(X=X, sample_points=sample_points)
            self.assertEqual(X_augmented.shape, torch.Size(batch_shape + [p, d]))
            if batch_shape == [2]:
                self.assertAllClose(X_augmented[0, :, -d_prime:], sample_points)
            else:
                self.assertAllClose(X_augmented[:, -d_prime:], sample_points)


class TestGetOptimalSamples(BotorchTestCase):
    def test_get_optimal_samples(self):
        dims = 3
        dtype = torch.float64
        for_testing_speed_kwargs = {"raw_samples": 20, "num_restarts": 2}
        num_optima = 7
        batch_shape = (3,)

        bounds = torch.tensor([[0, 1]] * dims, dtype=dtype).T
        X = torch.rand(*batch_shape, 4, dims, dtype=dtype)
        Y = torch.sin(2 * 3.1415 * X).sum(dim=-1, keepdim=True).to(dtype)
        model = SingleTaskGP(train_X=X, train_Y=Y)
        posterior_transform = ScalarizedPosteriorTransform(
            weights=torch.ones(1, dtype=dtype)
        )
        posterior_transform_neg = ScalarizedPosteriorTransform(
            weights=-torch.ones(1, dtype=dtype)
        )
        with torch.random.fork_rng():
            torch.manual_seed(0)
            X_opt_def, f_opt_def = get_optimal_samples(
                model=model,
                bounds=bounds,
                num_optima=num_optima,
                **for_testing_speed_kwargs,
            )
        correct_X_shape = (num_optima,) + batch_shape + (dims,)
        correct_f_shape = (num_optima,) + batch_shape + (1,)
        self.assertEqual(X_opt_def.shape, correct_X_shape)
        self.assertEqual(f_opt_def.shape, correct_f_shape)
        with torch.random.fork_rng():
            torch.manual_seed(0)
            X_opt_ps, f_opt_ps = get_optimal_samples(
                model=model,
                bounds=bounds,
                num_optima=num_optima,
                posterior_transform=posterior_transform,
                **for_testing_speed_kwargs,
            )
        self.assertAllClose(X_opt_def, X_opt_ps)

        with torch.random.fork_rng():
            torch.manual_seed(0)
            X_opt_ps_neg, f_opt_ps_neg = get_optimal_samples(
                model=model,
                bounds=bounds,
                num_optima=num_optima,
                posterior_transform=posterior_transform_neg,
                **for_testing_speed_kwargs,
            )
        # maxima larger than minima when the seed is fixed
        self.assertTrue(torch.all(f_opt_ps_neg < f_opt_ps))

        obj = LinearMCObjective(weights=-torch.ones(1, dtype=dtype))
        with torch.random.fork_rng():
            torch.manual_seed(0)
            X_opt_obj_neg, f_opt_obj_neg = get_optimal_samples(
                model=model,
                bounds=bounds,
                num_optima=num_optima,
                objective=obj,
                **for_testing_speed_kwargs,
            )
            # check that the minimum is the same for negative objective and
            # negative posterior transform
            self.assertAllClose(X_opt_ps_neg, X_opt_obj_neg)

        obj = LinearMCObjective(weights=-torch.ones(1, dtype=dtype))
        with torch.random.fork_rng():
            torch.manual_seed(0)
            _, f_opt_obj_pos = get_optimal_samples(
                model=model,
                bounds=bounds,
                num_optima=num_optima,
                objective=obj,
                return_transformed=True,
                **for_testing_speed_kwargs,
            )
            # check that the transformed return value is the negation of the
            # non-transformed return value
            self.assertAllClose(f_opt_obj_pos, -f_opt_obj_neg)

        with self.assertRaisesRegex(
            ValueError,
            "Only the ScalarizedPosteriorTransform is supported for "
            "get_optimal_samples.",
        ):
            get_optimal_samples(
                model=model,
                bounds=bounds,
                num_optima=num_optima,
                posterior_transform=ExpectationPosteriorTransform(n_w=5),
                **for_testing_speed_kwargs,
            )
        with self.assertRaisesRegex(
            ValueError,
            "Only one of `posterior_transform` and `objective` can be specified.",
        ):
            get_optimal_samples(
                model=model,
                bounds=bounds,
                num_optima=num_optima,
                posterior_transform=posterior_transform,
                objective=obj,
                **for_testing_speed_kwargs,
            )


class TestPreferenceUtils(BotorchTestCase):
    def test_repeat_to_match_aug_dim(self):
        """test repeat_to_match_aug_dim to ensure it repeat the elements
        in the correct order
        """
        # simple working case
        target_tensor = torch.arange(3).repeat(2, 1).T
        repeated_tensor = repeat_to_match_aug_dim(target_tensor, torch.zeros(6))
        self.assertEqual(repeated_tensor.shape, torch.Size([6, 2]))

        # simple invalid cases
        target_tensor = torch.rand(6, 2, 3)
        reference_tensor = torch.rand(5, 2, 3)
        with self.assertRaisesRegex(
            ValueError,
            "The first dimension of reference_tensor must be a multiple of",
        ):
            repeat_to_match_aug_dim(
                target_tensor=target_tensor, reference_tensor=reference_tensor
            )

        # similarting real use cases
        num_outcome_samples, n, q, d = 3, 2, 4, 5
        model = SingleTaskGP(train_X=torch.rand(n, d), train_Y=torch.rand(n, 1))
        obj = LearnedObjective(pref_model=model)
        samples = torch.rand(num_outcome_samples, q, d)

        # Save a reference to the original posterior method
        original_posterior_method = SingleTaskGP.posterior

        def nearly_zero_covar_posterior(self, *args, **kwargs):
            original_posterior = original_posterior_method(self, *args, **kwargs)

            # Modify the distribution
            original_posterior.distribution = MultivariateNormal(
                mean=original_posterior.distribution.mean,
                covariance_matrix=torch.diag_embed(
                    torch.full_like(
                        original_posterior.distribution.mean, fill_value=1e-15
                    )
                ),
            )

            # Return the modified posterior
            return original_posterior

        # Patch the posterior call such that sampling from the model's output will give
        # basically the same samples. This way, we are able to tell which preference
        # sample comes from which outcome sample.
        # When ``samples`` of shape ``num_samples x ...`` being passed through obj,
        # the returned augmented sample is of shape
        # ``(num_pref_sample * num_samples) x ...``.
        # If num_samples = 3 and num_pref_sample = 2,
        # along the first dimension of objective, objective values should correspond to
        # index [0, 1, 2, 0, 1, 2] of ``samples``.
        with patch.object(SingleTaskGP, "posterior", new=nearly_zero_covar_posterior):
            objective = obj(samples)

        repeated_samples = repeat_to_match_aug_dim(
            target_tensor=samples, reference_tensor=objective
        )

        self.assertAllClose(
            objective,
            torch.roll(objective, shifts=num_outcome_samples, dims=0),
            rtol=1e-3,
        )
        self.assertAllClose(
            repeated_samples,
            torch.roll(repeated_samples, shifts=num_outcome_samples, dims=0),
            rtol=1e-3,
        )
