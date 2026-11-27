#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.acquisition import (
    ExpectedImprovement,
    qExpectedImprovement,
    qMultiStepLookahead,
)
from botorch.acquisition.multi_step_lookahead import make_best_f, warmstart_multistep
from botorch.acquisition.objective import IdentityMCObjective
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase


class TestMultiStepLookahead(BotorchTestCase):
    def test_qMS_init(self):
        d = 2
        q = 1
        num_data = 3
        q_batch_sizes = [1, 1, 1]
        num_fantasies = [2, 2, 1]
        t_batch_size = [2]
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0], [1]], device=self.device, dtype=dtype)
            bounds = bounds.repeat(1, d)
            train_X = torch.rand(num_data, d, device=self.device, dtype=dtype)
            train_Y = torch.rand(num_data, 1, device=self.device, dtype=dtype)
            model = SingleTaskGP(train_X, train_Y)

            # exactly one of samplers or num_fantasies
            with self.assertRaises(UnsupportedError):
                qMultiStepLookahead(
                    model=model,
                    batch_sizes=q_batch_sizes,
                    valfunc_cls=[qExpectedImprovement] * 4,
                    valfunc_argfacs=[make_best_f] * 4,
                    inner_mc_samples=[2] * 4,
                )

            # cannot use qMS as its own valfunc_cls
            with self.assertRaises(UnsupportedError):
                qMultiStepLookahead(
                    model=model,
                    batch_sizes=q_batch_sizes,
                    valfunc_cls=[qMultiStepLookahead] * 4,
                    valfunc_argfacs=[make_best_f] * 4,
                    num_fantasies=num_fantasies,
                    inner_mc_samples=[2] * 4,
                )

            # construct using samplers
            samplers = [
                SobolQMCNormalSampler(sample_shape=torch.Size([nf]))
                for nf in num_fantasies
            ]
            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=q_batch_sizes,
                valfunc_cls=[qExpectedImprovement] * 4,
                valfunc_argfacs=[make_best_f] * 4,
                inner_mc_samples=[2] * 4,
                samplers=samplers,
            )
            self.assertEqual(qMS.num_fantasies, num_fantasies)

            # use default valfunc_cls, valfun_argfacs, inner_mc_samples
            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=q_batch_sizes,
                samplers=samplers,
            )
            self.assertEqual(len(qMS._valfunc_cls), 4)
            self.assertEqual(len(qMS.inner_samplers), 4)
            self.assertEqual(len(qMS._valfunc_argfacs), 4)

            # _construct_inner_samplers error catching tests below
            # AnalyticAcquisitionFunction with MCAcquisitionObjective
            with self.assertRaises(UnsupportedError):
                qMultiStepLookahead(
                    model=model,
                    objective=IdentityMCObjective(),
                    batch_sizes=q_batch_sizes,
                    valfunc_cls=[ExpectedImprovement] * 4,
                    valfunc_argfacs=[make_best_f] * 4,
                    num_fantasies=num_fantasies,
                )
            # AnalyticAcquisitionFunction and q > 1
            with self.assertRaises(UnsupportedError):
                qMultiStepLookahead(
                    model=model,
                    batch_sizes=[2, 2, 2],
                    valfunc_cls=[ExpectedImprovement] * 4,
                    valfunc_argfacs=[make_best_f] * 4,
                    num_fantasies=num_fantasies,
                    inner_mc_samples=[2] * 4,
                )
            # AnalyticAcquisitionFunction and inner_mc_samples
            with self.assertWarns(Warning):
                qMultiStepLookahead(
                    model=model,
                    batch_sizes=q_batch_sizes,
                    valfunc_cls=[ExpectedImprovement] * 4,
                    valfunc_argfacs=[make_best_f] * 4,
                    num_fantasies=num_fantasies,
                    inner_mc_samples=[2] * 4,
                )

            # test warmstarting
            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=q_batch_sizes,
                samplers=samplers,
            )
            q_prime = qMS.get_augmented_q_batch_size(q)
            eval_X = torch.rand(
                t_batch_size + [q_prime, d], device=self.device, dtype=dtype
            )
            warmstarted_X = warmstart_multistep(
                acq_function=qMS,
                bounds=bounds,
                num_restarts=5,
                raw_samples=10,
                full_optimizer=eval_X,
            )
            self.assertEqual(warmstarted_X.shape, torch.Size([5, q_prime, d]))

        with self.assertRaisesRegex(
            UnsupportedError,
            "`qMultiStepLookahead` got a non-MC `objective`. This is not supported."
            " Use `posterior_transform` and `objective=None` instead.",
        ):
            qMultiStepLookahead(model=model, batch_sizes=q_batch_sizes, objective="cat")

    def test_qMS(self):
        d = 2
        q = 1
        num_data = 3
        q_batch_sizes = [1, 1, 1]
        num_fantasies = [2, 2, 1]
        t_batch_size = [2]
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0], [1]], device=self.device, dtype=dtype)
            bounds = bounds.repeat(1, d)
            train_X = torch.rand(num_data, d, device=self.device, dtype=dtype)
            train_Y = torch.rand(num_data, 1, device=self.device, dtype=dtype)
            model = SingleTaskGP(train_X, train_Y)

            # default evaluation testsÎ
            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=[1, 1, 1],
                num_fantasies=num_fantasies,
            )
            q_prime = qMS.get_augmented_q_batch_size(q)
            eval_X = torch.rand(
                t_batch_size + [q_prime, d], device=self.device, dtype=dtype
            )
            result = qMS(eval_X)
            self.assertEqual(result.shape, torch.Size(t_batch_size))

            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=q_batch_sizes,
                valfunc_cls=[qExpectedImprovement] * 4,
                valfunc_argfacs=[make_best_f] * 4,
                num_fantasies=num_fantasies,
                inner_mc_samples=[2] * 4,
            )
            result = qMS(eval_X)
            self.assertEqual(result.shape, torch.Size(t_batch_size))

            # get induced fantasy model, with collapse_fantasy_base_samples
            fant_model = qMS.get_induced_fantasy_model(eval_X)
            self.assertEqual(
                fant_model.train_inputs[0].shape,
                torch.Size(
                    num_fantasies[::-1]
                    + t_batch_size
                    + [num_data + sum(q_batch_sizes), d]
                ),
            )

            # collapse fantasy base samples
            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=q_batch_sizes,
                valfunc_cls=[qExpectedImprovement] * 4,
                valfunc_argfacs=[make_best_f] * 4,
                num_fantasies=num_fantasies,
                inner_mc_samples=[2] * 4,
                collapse_fantasy_base_samples=False,
            )
            q_prime = qMS.get_augmented_q_batch_size(q)
            eval_X = torch.rand(
                t_batch_size + [q_prime, d], device=self.device, dtype=dtype
            )
            result = qMS(eval_X)
            self.assertEqual(result.shape, torch.Size(t_batch_size))
            self.assertEqual(
                qMS.samplers[0]._get_batch_range(model.posterior(eval_X)), (-3, -2)
            )

            # get induced fantasy model, without collapse_fantasy_base_samples
            fant_model = qMS.get_induced_fantasy_model(eval_X)
            self.assertEqual(
                fant_model.train_inputs[0].shape,
                torch.Size(
                    num_fantasies[::-1]
                    + t_batch_size
                    + [num_data + sum(q_batch_sizes), d]
                ),
            )

            # X_pending
            X_pending = torch.rand(5, d, device=self.device, dtype=dtype)
            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=q_batch_sizes,
                valfunc_cls=[qExpectedImprovement] * 4,
                valfunc_argfacs=[make_best_f] * 4,
                num_fantasies=num_fantasies,
                inner_mc_samples=[2] * 4,
                X_pending=X_pending,
            )
            q_prime = qMS.get_augmented_q_batch_size(q)
            eval_X = torch.rand(
                t_batch_size + [q_prime, d], device=self.device, dtype=dtype
            )
            result = qMS(eval_X)
            self.assertEqual(result.shape, torch.Size(t_batch_size))

            # add dummy base_weights to samplers
            samplers = [
                SobolQMCNormalSampler(sample_shape=torch.Size([nf]))
                for nf in num_fantasies
            ]
            for s in samplers:
                s.base_weights = torch.ones(
                    s.sample_shape[0], 1, device=self.device, dtype=dtype
                )

            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=[1, 1, 1],
                samplers=samplers,
            )
            q_prime = qMS.get_augmented_q_batch_size(q)
            eval_X = torch.rand(
                t_batch_size + [q_prime, d], device=self.device, dtype=dtype
            )
            result = qMS(eval_X)
            self.assertEqual(result.shape, torch.Size(t_batch_size))

            # extract candidates
            cand = qMS.extract_candidates(eval_X)
            self.assertEqual(cand.shape, torch.Size(t_batch_size + [q, d]))

    def test_make_best_f(self):
        # ``best_f`` must be in the original outcome space, i.e. that of the
        # posterior, also if the model uses an outcome transform (as by default).
        tkwargs = {"device": self.device, "dtype": torch.double}
        train_X = torch.rand(8, 2, **tkwargs)
        train_Y = 100 + 5 * torch.sin(6 * train_X)
        for num_outputs in (1, 2):
            model = SingleTaskGP(train_X, train_Y[:, :num_outputs]).eval()
            best_f = make_best_f(model=model, X=train_X)["best_f"]
            expected = train_Y[:, :num_outputs].amax(dim=0)
            self.assertAllClose(
                best_f, expected.squeeze(0) if num_outputs == 1 else expected
            )
        # fantasy model with batched train targets
        model = SingleTaskGP(train_X, train_Y[:, :1]).eval()
        model.posterior(train_X)
        fantasy_model = model.condition_on_observations(
            X=torch.rand(2, 1, 2, **tkwargs),
            Y=torch.tensor([[[110.0]], [[90.0]]], **tkwargs),
        )
        best_f = make_best_f(model=fantasy_model, X=train_X)["best_f"]
        self.assertAllClose(
            best_f, torch.stack([torch.tensor(110.0, **tkwargs), train_Y[:, 0].max()])
        )
