import itertools
from typing import Any

import torch
from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
from botorch.acquisition.probabilistic_reparameterization import (
    AnalyticProbabilisticReparameterization,
    MCProbabilisticReparameterization,
)
from botorch.generation.gen import gen_candidates_scipy, gen_candidates_torch
from botorch.models import MixedSingleTaskGP
from botorch.models.transforms.factory import get_rounding_input_transform
from botorch.models.transforms.input import (
    AnalyticProbabilisticReparameterizationInputTransform,
    MCProbabilisticReparameterizationInputTransform,
    OneHotToNumeric,
)
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.test_functions.synthetic import Ackley, AckleyMixed
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.test_helpers import get_model
from botorch.utils.testing import BotorchTestCase


def get_categorical_features_dict(feature_to_num_categories: dict[int, int]):
    r"""Get the mapping of starting index in one-hot space to cardinality.

    This mapping is used to construct the OneHotToNumeric transform. This
    requires that all of the categorical parameters are the rightmost elements.

    Args:
        feature_to_num_categories: Mapping of feature index to cardinality in the
            untransformed space.

    """
    start = None
    categorical_features = {}
    for idx, cardinality in sorted(
        feature_to_num_categories.items(), key=lambda kv: kv[0]
    ):
        if start is None:
            start = idx
        categorical_features[start] = cardinality
        # add cardinality to start
        start += cardinality
    return categorical_features


class TestProbabilisticReparameterizationInputTransform(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.tkwargs: dict[str, Any] = {"device": self.device, "dtype": torch.double}
        self.one_hot_bounds = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 4.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            **self.tkwargs,
        )

        self.analytic_params = dict(
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=True,
            tau=0.1,
        )

        self.mc_params = dict(
            **self.analytic_params,
            mc_samples=128,
            resample=False,
        )

    def test_probabilistic_reparameterization_transform_construction(self):
        for use_analytic in (True, False):
            with self.subTest(use_analytic=use_analytic):
                self._test_probabilistic_reparameterization_transform_construction(
                    use_analytic=use_analytic
                )

    def _test_probabilistic_reparameterization_transform_construction(
        self, use_analytic: bool
    ) -> None:
        bounds = self.one_hot_bounds
        integer_indices = [2, 3]
        categorical_features = {4: 2, 6: 3}
        transform_params = self.analytic_params if use_analytic else self.mc_params
        pr_transform_cls = (
            AnalyticProbabilisticReparameterizationInputTransform
            if use_analytic
            else MCProbabilisticReparameterizationInputTransform
        )

        # must provide either categorical or discrete features
        with self.assertRaises(ValueError):
            pr_transform_cls(
                one_hot_bounds=bounds,
                **transform_params,
            )

        # categorical features must be in the rightmost columns
        with self.assertRaisesRegex(ValueError, "rightmost"):
            pr_transform_cls(
                one_hot_bounds=bounds,
                integer_indices=integer_indices,
                categorical_features={4: 2, 6: 1},
                **transform_params,
            )

        # integer features must be between continuous and categorical
        with self.assertRaisesRegex(ValueError, "integer"):
            pr_transform_cls(
                one_hot_bounds=bounds,
                integer_indices=[1, 2],
                categorical_features=categorical_features,
                **transform_params,
            )

        # correct construction passes without raising errors
        pr_transform_cls(
            one_hot_bounds=bounds,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
            **transform_params,
        )

    def test_analytic_probabilistic_reparameterization_transform_enumeration(self):
        # analytic generates all discrete options correctly
        # use subset of features so that we can manually generate all options
        bounds = self.one_hot_bounds
        sub_bounds = bounds[:, [0, 2, 6, 7, 8]]
        sub_integer_indices = [1]
        sub_categorical_features = {2: 3}
        tf_analytic = AnalyticProbabilisticReparameterizationInputTransform(
            one_hot_bounds=sub_bounds,
            integer_indices=sub_integer_indices,
            categorical_features=sub_categorical_features,
            **self.analytic_params,
        )

        num_discrete_options = 5 * 3
        expected_all_discrete_options = torch.zeros(
            (num_discrete_options, sub_bounds.shape[-1])
        )
        expected_all_discrete_options[:, 1] = torch.repeat_interleave(
            torch.arange(5), 3
        )
        expected_all_discrete_options[:, 2:] = torch.eye(3).repeat([5, 1])

        self.assertAllClose(
            expected_all_discrete_options, tf_analytic.all_discrete_options
        )

    def test_probabilistic_reparameterization_transform_invalid_forward(self):
        for use_analytic in (True, False):
            with self.subTest(use_analytic=use_analytic):
                self._test_probabilistic_reparameterization_transform_invalid_forward(
                    use_analytic=use_analytic
                )

    def _test_probabilistic_reparameterization_transform_invalid_forward(
        self, use_analytic: bool
    ) -> None:
        bounds = self.one_hot_bounds
        integer_indices = [2, 3]
        categorical_features = {4: 2, 6: 3}
        transform_params = self.analytic_params if use_analytic else self.mc_params
        pr_transform_cls = (
            AnalyticProbabilisticReparameterizationInputTransform
            if use_analytic
            else MCProbabilisticReparameterizationInputTransform
        )

        tf = pr_transform_cls(
            one_hot_bounds=bounds,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
            **transform_params,
        )

        X = torch.rand(4, 1, 1, bounds.shape[1])

        with self.assertRaisesRegex(ValueError, "`n`"):
            tf.transform(X.expand(-1, -1, 2, -1))

        with self.assertRaisesRegex(ValueError, "dimension of size 1 at index -3"):
            tf.transform(X.expand(-1, 2, -1, -1))

    def test_probabilistic_reparameterization_transform_forward(self):
        bounds = self.one_hot_bounds
        integer_indices = [2, 3]
        categorical_features = {4: 2, 6: 3}

        tf_analytic = AnalyticProbabilisticReparameterizationInputTransform(
            one_hot_bounds=bounds,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
            **self.analytic_params,
        )

        X = torch.tensor(
            [[[0.2, 0.8, 3.2, 1.5, 0.9, 0.05, 0.05, 0.05, 0.95]]], **self.tkwargs
        )
        X_transformed_analytic = tf_analytic.transform(X)

        expected_shape = [5 * 6 * 2 * 3, 1, bounds.shape[-1]]
        self.assertEqual(X_transformed_analytic.shape, torch.Size(expected_shape))

        tf_mc = MCProbabilisticReparameterizationInputTransform(
            one_hot_bounds=bounds,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
            **self.mc_params,
        )

        X_transformed_mc = tf_mc.transform(X)

        expected_shape = [tf_mc.mc_samples, 1, bounds.shape[-1]]
        self.assertEqual(X_transformed_mc.shape, torch.Size(expected_shape))

        continuous_indices = [0, 1]
        discrete_indices = [
            d for d in range(bounds.shape[-1]) if d not in continuous_indices
        ]
        for X_transformed in [X_transformed_analytic, X_transformed_mc]:
            self.assertAllClose(
                X[..., continuous_indices].repeat([X_transformed.shape[0], 1, 1]),
                X_transformed[..., continuous_indices],
            )

            # all discrete indices have been rounded
            self.assertAllClose(
                X_transformed[..., discrete_indices] % 1,
                torch.zeros_like(X_transformed[..., discrete_indices]),
            )

        # for MC, all integer indices should be within [floor(X), ceil(X)]
        # categoricals should be approximately proportional to their probability
        self.assertTrue(
            ((X.floor() <= X_transformed_mc) & (X_transformed_mc <= X.ceil()))[
                ..., integer_indices
            ].all()
        )
        self.assertAllClose(X_transformed_mc[..., -1].mean().item(), 0.95, atol=0.10)


class TestProbabilisticReparameterization(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.tkwargs: dict[str, Any] = {"device": self.device, "dtype": torch.double}

        self.acqf_params = dict(
            batch_limit=32,
        )

        self.optimize_acqf_params = dict(
            num_restarts=10,
            raw_samples=512,
            options={
                "batch_limit": 5,
                "maxiter": 200,
                "rel_tol": float("-inf"),
            },
        )

    def test_probabilistic_reparameterization_binary(
        self,
        base_acq_func_cls=LogExpectedImprovement,
    ):
        torch.manual_seed(0)
        f = AckleyMixed(dim=6, randomize_optimum=False)
        f.discrete_inds = [3, 4, 5]
        train_X = torch.rand((10, f.dim), **self.tkwargs)
        train_X[:, f.discrete_inds] = train_X[:, f.discrete_inds].round()
        train_Y = f(train_X).unsqueeze(-1)
        model = get_model(train_X, train_Y)
        base_acq_func = base_acq_func_cls(model, best_f=train_Y.max())

        pr_acq_func_params = dict(
            acq_function=base_acq_func,
            one_hot_bounds=f.bounds,
            integer_indices=f.discrete_inds,
            **self.acqf_params,
        )

        pr_analytic_acq_func = AnalyticProbabilisticReparameterization(
            **pr_acq_func_params
        )

        pr_mc_acq_func = MCProbabilisticReparameterization(**pr_acq_func_params)

        X = torch.tensor([[[0.3, 0.7, 0.8, 0.0, 0.5, 1.0]]], **self.tkwargs)
        X_lb, X_ub = X.clone(), X.clone()
        X_lb[..., 4] = 0.0
        X_ub[..., 4] = 1.0

        acq_value_base_mean = (base_acq_func(X_lb) + base_acq_func(X_ub)) / 2
        acq_value_analytic = pr_analytic_acq_func(X)
        acq_value_mc = pr_mc_acq_func(X)

        # this is not exact due to sigmoid transform in discrete probabilities
        self.assertAllClose(acq_value_analytic, acq_value_base_mean, rtol=0.1)
        self.assertAllClose(acq_value_mc, acq_value_base_mean, rtol=0.1)

        candidate_analytic, acq_values_analytic = optimize_acqf(
            acq_function=pr_analytic_acq_func,
            bounds=f.bounds,
            q=1,
            gen_candidates=gen_candidates_scipy,
            **self.optimize_acqf_params,
        )

        candidate_mc, acq_values_mc = optimize_acqf(
            acq_function=pr_mc_acq_func,
            bounds=f.bounds,
            q=1,
            gen_candidates=gen_candidates_torch,
            **self.optimize_acqf_params,
        )

        fixed_features_list = [
            {feat_dim + 3: val for feat_dim, val in enumerate(vals)}
            for vals in itertools.product([0, 1], repeat=len(f.discrete_inds))
        ]
        candidate_exhaustive, acq_values_exhaustive = optimize_acqf_mixed(
            acq_function=base_acq_func,
            fixed_features_list=fixed_features_list,
            bounds=f.bounds,
            q=1,
            **self.optimize_acqf_params,
        )

        self.assertTrue(candidate_analytic.shape == (1, f.dim))
        self.assertTrue(candidate_mc.shape == (1, f.dim))

        self.assertAllClose(candidate_analytic, candidate_exhaustive, rtol=0.1)
        self.assertAllClose(acq_values_analytic, acq_values_exhaustive, rtol=0.1)
        self.assertAllClose(candidate_mc, candidate_exhaustive, rtol=0.1)
        self.assertAllClose(acq_values_mc, acq_values_exhaustive, rtol=0.1)

    def test_probabilistic_reparameterization_binary_qLogEI(self):
        self.test_probabilistic_reparameterization_binary(
            base_acq_func_cls=qLogExpectedImprovement,
        )

    def test_probabilistic_reparameterization_categorical(
        self,
        base_acq_func_cls=LogExpectedImprovement,
    ):
        torch.manual_seed(0)
        # we use Ackley here to ensure the categorical features are the
        # rightmost elements
        dim = 5
        bounds = [(0.0, 1.0)] * 5
        f = Ackley(dim=dim, bounds=bounds)
        # convert the continuous features into categorical features
        feature_to_num_categories = {3: 3, 4: 5}
        for feature_idx, num_categories in feature_to_num_categories.items():
            f.bounds[1, feature_idx] = num_categories - 1

        categorical_features = get_categorical_features_dict(feature_to_num_categories)
        one_hot_bounds = torch.zeros(
            2, 3 + sum(categorical_features.values()), **self.tkwargs
        )
        one_hot_bounds[1, :] = 1.0
        init_exact_rounding_func = get_rounding_input_transform(
            one_hot_bounds=one_hot_bounds,
            categorical_features=categorical_features,
            initialization=True,
        )
        one_hot_to_numeric = OneHotToNumeric(
            dim=one_hot_bounds.shape[1],
            categorical_features=categorical_features,
            transform_on_train=False,
        ).to(**self.tkwargs)

        raw_X = (
            draw_sobol_samples(one_hot_bounds, n=10, q=1).squeeze(-2).to(**self.tkwargs)
        )
        train_X = init_exact_rounding_func(raw_X)
        train_Y = f(one_hot_to_numeric.transform(train_X)).unsqueeze(-1)
        model = MixedSingleTaskGP(
            train_X=one_hot_to_numeric.transform(train_X),
            train_Y=train_Y,
            cat_dims=list(feature_to_num_categories.keys()),
            input_transform=one_hot_to_numeric,
        )
        base_acq_func = base_acq_func_cls(model, best_f=train_Y.max())

        pr_acq_func_params = dict(
            acq_function=base_acq_func,
            one_hot_bounds=one_hot_bounds,
            categorical_features=categorical_features,
            **self.acqf_params,
        )

        pr_analytic_acq_func = AnalyticProbabilisticReparameterization(
            **pr_acq_func_params
        )

        pr_mc_acq_func = MCProbabilisticReparameterization(**pr_acq_func_params)

        X = one_hot_bounds[:1, :].clone().unsqueeze(0)
        X[..., -1] = 1.0
        X_lb, X_ub = X.clone(), X.clone()
        X[..., 3:5] = 0.5
        X_lb[..., 3] = 1.0
        X_ub[..., 4] = 1.0

        acq_value_base_mean = (base_acq_func(X_lb) + base_acq_func(X_ub)) / 2
        acq_value_analytic = pr_analytic_acq_func(X)
        acq_value_mc = pr_mc_acq_func(X)

        # this is not exact due to sigmoid transform in discrete probabilities
        self.assertAllClose(acq_value_analytic, acq_value_base_mean, rtol=0.1)
        self.assertAllClose(acq_value_mc, acq_value_base_mean, rtol=0.1)

        candidate_analytic, acq_values_analytic = optimize_acqf(
            acq_function=pr_analytic_acq_func,
            bounds=one_hot_bounds,
            q=1,
            gen_candidates=gen_candidates_scipy,
            **self.optimize_acqf_params,
        )

        candidate_mc, acq_values_mc = optimize_acqf(
            acq_function=pr_mc_acq_func,
            bounds=one_hot_bounds,
            q=1,
            gen_candidates=gen_candidates_torch,
            **self.optimize_acqf_params,
        )

        fixed_features_list = [
            {
                start_dim + i: float(val == i)
                for (start_dim, num_cat), val in zip(categorical_features.items(), vals)
                for i in range(num_cat)
            }
            for vals in itertools.product(*map(range, categorical_features.values()))
        ]
        candidate_exhaustive, acq_values_exhaustive = optimize_acqf_mixed(
            acq_function=base_acq_func,
            fixed_features_list=fixed_features_list,
            bounds=one_hot_bounds,
            q=1,
            **self.optimize_acqf_params,
        )

        self.assertTrue(candidate_analytic.shape == (1, one_hot_bounds.shape[-1]))
        self.assertTrue(candidate_mc.shape == (1, one_hot_bounds.shape[-1]))
        self.assertTrue(one_hot_to_numeric(candidate_analytic).shape == (1, f.dim))

        # round the mc candidate to allow for comparison
        candidate_mc = init_exact_rounding_func(candidate_mc)

        self.assertAllClose(candidate_analytic, candidate_exhaustive, rtol=0.1)
        self.assertAllClose(acq_values_analytic, acq_values_exhaustive, rtol=0.1)
        self.assertAllClose(candidate_mc, candidate_exhaustive, rtol=0.1)
        self.assertAllClose(acq_values_mc, acq_values_exhaustive, rtol=0.1)

    def test_probabilistic_reparameterization_categorical_qLogEI(self):
        self.test_probabilistic_reparameterization_categorical(
            base_acq_func_cls=qLogExpectedImprovement,
        )
