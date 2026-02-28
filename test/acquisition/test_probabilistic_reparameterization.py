import itertools
from typing import Any

import torch
from botorch.acquisition import (
    AcquisitionFunction,
    LogExpectedImprovement,
    qLogExpectedImprovement,
)
from botorch.acquisition.probabilistic_reparameterization import (
    AnalyticProbabilisticReparameterization,
    AnalyticProbabilisticReparameterizationInputTransform,
    get_probabilistic_reparameterization_input_transform,
    MCProbabilisticReparameterization,
    MCProbabilisticReparameterizationInputTransform,
)
from botorch.generation.gen import gen_candidates_scipy, gen_candidates_torch
from botorch.models import MixedSingleTaskGP
from botorch.models.transforms.factory import get_rounding_input_transform
from botorch.models.transforms.input import (
    ChainedInputTransform,
    Normalize,
    OneHotToNumeric,
)
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.test_functions.synthetic import Ackley, AckleyMixed
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.test_helpers import get_model
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior
from botorch.utils.transforms import unnormalize


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

        # no gaps allowed between categorical features
        with self.assertRaisesRegex(ValueError, "rightmost"):
            pr_transform_cls(
                one_hot_bounds=bounds,
                integer_indices=integer_indices,
                categorical_features={4: 2, 7: 2},
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

        X = torch.rand(4, 1, 1, bounds.shape[1], **self.tkwargs)

        with self.assertRaisesRegex(ValueError, "3 dimensions"):
            tf.transform(X[0, 0, ...])

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

        tf_analytic_discrete = AnalyticProbabilisticReparameterizationInputTransform(
            one_hot_bounds=bounds,
            integer_indices=[0, 1, 2, 3],
            categorical_features=categorical_features,
            **self.analytic_params,
        )
        X_transformed_analytic_discrete = tf_analytic_discrete.transform(X)
        expected_shape = [2 * 2 * 5 * 6 * 2 * 3, 1, bounds.shape[-1]]
        self.assertEqual(
            X_transformed_analytic_discrete.shape, torch.Size(expected_shape)
        )

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

    def test_probabilistic_reparameterization_transform_get_probs(self):
        bounds = self.one_hot_bounds
        dtype = torch.float32
        integer_indices = [2, 3]
        categorical_features = {4: 2, 6: 3}

        X = torch.tensor(
            [[[0.0, 0.0, 1.1, 2.5, 0.6, 0.4, 1.0, 0.0, 0.0]]],
            dtype=dtype,
            device=self.device,
        )

        tf_analytic = AnalyticProbabilisticReparameterizationInputTransform(
            one_hot_bounds=bounds,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
        )

        probs = tf_analytic.get_probs(X).squeeze(0)  # n_discrete x 1
        all_discrete_options = tf_analytic.all_discrete_options  # n_discrete x d
        # get the indices of any discrete options that have zero probability of
        # being sampled by X
        zero_prob_options = torch.any(
            (all_discrete_options[:, integer_indices] - X[..., integer_indices]).abs()
            >= 1.0,
            dim=-1,
        ).squeeze(0)

        self.assertEqual(probs.shape[0], all_discrete_options.shape[0])
        self.assertTrue(torch.all(probs[zero_prob_options] == 0.0))
        self.assertTrue(torch.all(probs[~zero_prob_options] > 0.0))
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=4)

        tf_mc = MCProbabilisticReparameterizationInputTransform(
            one_hot_bounds=bounds,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
        )

        rounding_prob = tf_mc.get_rounding_prob(X)

        num_cont = min(integer_indices)
        num_discrete = bounds.shape[1] - num_cont
        self.assertEqual(rounding_prob.shape, torch.Size([*X.shape[:-1], num_discrete]))

        for cat, card in categorical_features.items():
            # get_rounding_prob only returns the integer indices; need to offset
            start, end = cat - num_cont, cat - num_cont + card
            self.assertAlmostEqual(
                rounding_prob[..., start:end].sum().item(), 1.0, places=4
            )

    def test_probabilistic_reparameterization_transform_equality(self):
        bounds = self.one_hot_bounds
        integer_indices = [2, 3]
        categorical_features = {4: 2, 6: 3}

        all_tf_kwargs = (
            dict(
                one_hot_bounds=bounds,
                integer_indices=integer_indices,
                categorical_features=categorical_features,
            )
            | self.mc_params
            | {"resample": False}
        )

        tf_mc1 = MCProbabilisticReparameterizationInputTransform(**all_tf_kwargs)
        tf_mc2 = MCProbabilisticReparameterizationInputTransform(**all_tf_kwargs)
        self.assertTrue(tf_mc1.equals(tf_mc2))

        updated_tf_kwargs = all_tf_kwargs | {"resample": True}
        tf_mc3 = MCProbabilisticReparameterizationInputTransform(**updated_tf_kwargs)
        self.assertFalse(tf_mc1.equals(tf_mc3))

        updated_tf_kwargs = all_tf_kwargs | {"integer_indices": [3]}
        tf_mc4 = MCProbabilisticReparameterizationInputTransform(**updated_tf_kwargs)
        self.assertFalse(tf_mc1.equals(tf_mc4))

        all_tf_kwargs_analytic = (
            dict(
                one_hot_bounds=bounds,
                integer_indices=integer_indices,
                categorical_features=categorical_features,
            )
            | self.analytic_params
        )
        tf_analytic1 = AnalyticProbabilisticReparameterizationInputTransform(
            **all_tf_kwargs_analytic
        )
        tf_analytic2 = AnalyticProbabilisticReparameterizationInputTransform(
            **all_tf_kwargs_analytic
        )
        self.assertTrue(tf_analytic1.equals(tf_analytic2))
        self.assertFalse(tf_analytic1.equals(tf_mc1))

        # test comparison of base_samples
        X = torch.rand(
            4, 1, 1, bounds.shape[1], dtype=torch.float64, device=self.device
        )
        tf_mc1.transform(X)
        tf_mc2.transform(X)
        self.assertFalse(tf_mc1.equals(tf_mc2))


class TestGetProbabilisticReparameterizationInputTransform(BotorchTestCase):
    def test_get_probabilistic_reparameterization_input_transform(self):
        for dtype in (torch.float, torch.double):
            for use_analytic in (True, False):
                with self.subTest(dtype=dtype, use_analytic=use_analytic):
                    self._test_get_probabilistic_reparameterization_input_transform(
                        dtype=dtype, use_analytic=use_analytic
                    )

    def _test_get_probabilistic_reparameterization_input_transform(
        self, dtype: torch.dtype, use_analytic: bool
    ):
        expected_pr_transform_cls = (
            AnalyticProbabilisticReparameterizationInputTransform
            if use_analytic
            else MCProbabilisticReparameterizationInputTransform
        )

        one_hot_bounds = torch.tensor(
            [
                [0, 5],
                [0, 4],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
            ],
            dtype=dtype,
            device=self.device,
        ).t()

        with self.assertRaises(ValueError):
            # test no integer or categorical
            get_probabilistic_reparameterization_input_transform(
                one_hot_bounds=one_hot_bounds,
            )

        categorical_features = {2: 2, 4: 3}
        integer_indices = [1]
        mc_samples = 64
        resample = False
        tau = 0.2
        # test both integer and categorical
        tf = get_probabilistic_reparameterization_input_transform(
            one_hot_bounds=one_hot_bounds,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
            use_analytic=use_analytic,
            mc_samples=mc_samples,
            resample=resample,
            tau=tau,
        )
        self.assertIsInstance(tf, ChainedInputTransform)
        tfs = list(tf.items())

        self.assertEqual(len(tfs), 3)
        tf_name_i, tf_i = tfs[0]
        self.assertEqual(tf_name_i, "unnormalize")
        self.assertIsInstance(tf_i, Normalize)
        self.assertTrue(tf_i.reverse)
        bounds = one_hot_bounds[:, integer_indices]
        offset = bounds[:1, :]
        coefficient = bounds[1:2, :] - offset
        self.assertTrue(torch.equal(tf_i.coefficient, coefficient))
        self.assertTrue(torch.equal(tf_i.offset, offset))
        self.assertEqual(tf_i._d, one_hot_bounds.shape[1])
        self.assertEqual(
            tf_i.indices, torch.tensor(integer_indices, device=self.device)
        )

        tf_name_i, tf_i = tfs[1]
        self.assertEqual(tf_name_i, "round")
        self.assertIsInstance(tf_i, expected_pr_transform_cls)
        self.assertEqual(tf_i.categorical_features, categorical_features)
        self.assertEqual(tf_i.tau, tau)
        if not use_analytic:
            self.assertEqual(tf_i.mc_samples, mc_samples)
            self.assertEqual(tf_i.resample, resample)

        tf_name_i, tf_i = tfs[2]
        self.assertEqual(tf_name_i, "normalize")
        self.assertIsInstance(tf_i, Normalize)
        self.assertFalse(tf_i.reverse)
        self.assertTrue(torch.equal(tf_i.coefficient, coefficient))
        self.assertTrue(torch.equal(tf_i.offset, offset))
        self.assertEqual(tf_i._d, one_hot_bounds.shape[1])

        # test forward
        X = torch.rand(
            4, 1, 1, one_hot_bounds.shape[1], dtype=dtype, device=self.device
        )
        X_tf = tf(X)
        # assert the continuous param is unaffected
        expected_samples = (
            tf["round"].all_discrete_options.shape[0]
            if use_analytic
            else tf["round"].mc_samples
        )
        self.assertEqual(expected_samples, X_tf.shape[-3])
        X_expanded = X.expand(*X.shape[:-3], expected_samples, *X.shape[-2:])
        self.assertTrue(torch.equal(X_tf[..., 0], X_expanded[..., 0]))
        # check that integer params are rounded
        X_tf_int = unnormalize(X_tf, bounds=one_hot_bounds)[..., integer_indices]
        self.assertTrue(torch.all(X_tf_int % 1 == 0))
        # check that categoricals are discretized - check onehot constraint
        for start, card in categorical_features.items():
            end = start + card
            self.assertTrue(torch.all(X_tf[..., start:end].sum(dim=-1) == 1))
        # test transform on train/eval/fantasize
        for tf_i in tf.values():
            self.assertFalse(tf_i.transform_on_train)
            self.assertTrue(tf_i.transform_on_eval)
            # self.assertFalse(tf_i.transform_on_fantasize)

        # test categoricals, no integers
        tf = get_probabilistic_reparameterization_input_transform(
            one_hot_bounds=one_hot_bounds,
            integer_indices=None,
            categorical_features=categorical_features,
            use_analytic=use_analytic,
            mc_samples=mc_samples,
            resample=resample,
            tau=tau,
        )
        tfs = list(tf.items())
        self.assertEqual(len(tfs), 1)
        tf_name_i, tf_i = tfs[0]
        self.assertEqual(tf_name_i, "round")
        self.assertIsInstance(tf_i, expected_pr_transform_cls)
        self.assertIsNone(tf_i.integer_indices)
        self.assertEqual(tf_i.categorical_features, categorical_features)

        # test integers, no categoricals
        integer_indices = [one_hot_bounds.shape[1] - 1]
        tf = get_probabilistic_reparameterization_input_transform(
            one_hot_bounds=one_hot_bounds,
            integer_indices=integer_indices,
            categorical_features=None,
            use_analytic=use_analytic,
            mc_samples=mc_samples,
            resample=resample,
            tau=tau,
        )

        tfs = list(tf.items())
        self.assertEqual(len(tfs), 3)
        _, tf_i = tfs[1]
        self.assertEqual(tf_i.integer_indices.tolist(), integer_indices)
        self.assertIsNone(tf_i.categorical_features)


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

    def test_probabilistic_reparameterization_continuous(self):
        bounds = torch.zeros((2, 5))
        bounds[1, :] = 1.0

        mm = MockModel(MockPosterior(samples=torch.rand(1, 1)))
        base_acq_func = qLogExpectedImprovement(model=mm, best_f=0.0)
        for pr_cls in (
            MCProbabilisticReparameterization,
            AnalyticProbabilisticReparameterization,
        ):
            with self.assertRaisesRegex(
                NotImplementedError, "Categorical features or integer indices"
            ):
                pr_cls(
                    acq_function=base_acq_func,
                    one_hot_bounds=bounds,
                )

    def test_probabilistic_reparameterization_discrete(self):
        bounds = torch.zeros((2, 5), **self.tkwargs)
        bounds[1, :] = 5.0
        # test problem with no continuous features
        integer_indices = list(range(bounds.shape[1]))

        mc_samples, batch_limit = 128, self.acqf_params["batch_limit"]
        # posterior samples have base shape according to forward call
        # in _MCProbabilisticReparameterization
        samples = torch.tensor(10.0, **self.tkwargs)
        base_shape = torch.Size([1, batch_limit, 1, 1])
        mm = MockModel(MockPosterior(samples=samples, base_shape=base_shape))
        base_acq_func = qLogExpectedImprovement(model=mm, best_f=0.0)

        pr_acqf_params = dict(
            acq_function=base_acq_func,
            one_hot_bounds=bounds,
            integer_indices=integer_indices,
            mc_samples=mc_samples,
            **self.acqf_params,
        )
        pr_acqf = MCProbabilisticReparameterization(**pr_acqf_params)

        X = torch.tensor(
            [[[1.0, 2.0, 4.0, 0.0, 1.0]]], requires_grad=True, **self.tkwargs
        )
        loss = -pr_acqf(X).sum()
        grad = torch.autograd.grad(loss, X)[0]
        self.assertEqual(grad.shape, X.shape)

        # test removing batch limit
        # also test turning off ma baseline
        base_shape = torch.Size([1, mc_samples, 1, 1])
        mm = MockModel(MockPosterior(samples=samples, base_shape=base_shape))
        base_acq_func = qLogExpectedImprovement(model=mm, best_f=0.0)
        pr_acqf_params_no_batch_limit = pr_acqf_params | dict(
            acq_function=base_acq_func,
            batch_limit=None,
            use_ma_baseline=False,
        )
        pr_acqf = MCProbabilisticReparameterization(**pr_acqf_params_no_batch_limit)
        loss = -pr_acqf(X).sum()
        grad = torch.autograd.grad(loss, X)[0]
        self.assertEqual(grad.shape, X.shape)

        # test categorical
        bounds = torch.zeros((2, 5))
        bounds[1, :] = 1.0
        categorical_features = {0: 3, 3: 2}
        pr_acqf_params_cat = pr_acqf_params | dict(
            categorical_features=categorical_features,
            integer_indices=None,
            apply_numeric=True,
        )
        pr_acqf = MCProbabilisticReparameterization(**pr_acqf_params_cat)

        # check that apply_numeric is properly propagated
        X = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]], **self.tkwargs)
        X_oh2n = torch.tensor([[1.0, 0.0]], **self.tkwargs)
        self.assertIsNotNone(pr_acqf.one_hot_to_numeric)
        self.assertAllClose(pr_acqf.one_hot_to_numeric(X), X_oh2n)

    def test_probabilistic_reparameterization_optim(self):
        for base_acq_func_cls in (qLogExpectedImprovement, LogExpectedImprovement):
            with self.subTest("binary", base_acq_func_cls=base_acq_func_cls):
                self._test_probabilistic_reparameterization_optim_binary_mixed(
                    base_acq_func_cls=base_acq_func_cls
                )

            with self.subTest("categorical", base_acq_func_cls=base_acq_func_cls):
                self._test_probabilistic_reparameterization_optim_categorical_mixed(
                    base_acq_func_cls=base_acq_func_cls
                )

    def _test_probabilistic_reparameterization_optim_binary_mixed(
        self,
        base_acq_func_cls: type[AcquisitionFunction],
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

        # test that a purely continuous problem raises an error
        with self.assertRaises(NotImplementedError):
            continuous_params = pr_acq_func_params | {"integer_indices": None}
            AnalyticProbabilisticReparameterization(**continuous_params)

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
            {
                feat_dim: val
                for feat_dim, val in enumerate(vals, start=min(f.discrete_inds))
            }
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

    def _test_probabilistic_reparameterization_optim_categorical_mixed(
        self,
        base_acq_func_cls: type[AcquisitionFunction],
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
        candidate_mc_rnd = init_exact_rounding_func(candidate_mc)

        self.assertAllClose(candidate_analytic, candidate_exhaustive, rtol=0.1)
        self.assertAllClose(acq_values_analytic, acq_values_exhaustive, rtol=0.1)
        self.assertAllClose(candidate_mc_rnd, candidate_exhaustive, rtol=0.1)
        self.assertAllClose(acq_values_mc, acq_values_exhaustive, rtol=0.1)

    def test_probabilistic_reparameterization_sample_candidates(self):
        torch.manual_seed(0)
        bounds = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 4.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            **self.tkwargs,
        )
        integer_indices = [2, 3]
        categorical_features = {4: 2, 6: 3}

        candidate = torch.tensor(
            [[0.1, 0.2, 1.1, 2.5, 0.8, 0.4, 1.0, 0.0, 0.0]], **self.tkwargs
        )
        mm = MockModel(MockPosterior(samples=torch.rand(1, 1)))
        base_acq_func = qLogExpectedImprovement(model=mm, best_f=0.0)
        pr_acq = MCProbabilisticReparameterization(
            acq_function=base_acq_func,
            one_hot_bounds=bounds,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
            **self.acqf_params,
        )

        num_candidate_samples = 256
        candidate_expanded = candidate.expand(num_candidate_samples, -1)
        candidate_samples = pr_acq.sample_candidates(candidate_expanded)
        self.assertEqual(
            candidate_samples.shape,
            torch.Size([num_candidate_samples, bounds.shape[1]]),
        )

        # continuous parameters should not be rounded
        cont_indices = list(range(min(integer_indices)))
        self.assertAllClose(
            candidate_expanded[..., cont_indices], candidate_samples[..., cont_indices]
        )

        # categorical parameters should be one-hot encoded
        ones = torch.ones_like(candidate_samples[..., -1], dtype=torch.long)
        for cat, card in categorical_features.items():
            start, end = cat, cat + card
            self.assertAllClose(
                (candidate_samples[..., start:end] == 1.0).sum(dim=-1), ones
            )
            self.assertAllClose(
                (candidate_samples[..., start:end] == 0.0).sum(dim=-1),
                (card - 1) * ones,
            )

        # all proposed integers should be within [x.floor, x.ceil]
        int_within_range = (
            candidate_samples[:, integer_indices]
            - candidate_expanded[..., integer_indices]
        ).abs() < 1.0
        # FIXME: the line below will currently fail, since `sample_candidates` passes
        # the candidate through an (un)normalization transform. Either the logic
        # in the method needs to change, or this test needs to change.
        self.assertTrue(torch.all(int_within_range))
