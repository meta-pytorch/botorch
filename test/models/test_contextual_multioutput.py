#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.contextual_multioutput import LCEMGP
from botorch.models.multitask import MultiTaskGP
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.test_helpers import gen_multi_task_dataset
from botorch.utils.testing import BotorchTestCase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import MaternKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from linear_operator.operators import LinearOperator
from linear_operator.operators.interpolated_linear_operator import (
    InterpolatedLinearOperator,
)
from torch import Tensor


class ContextualMultiOutputTest(BotorchTestCase):
    def test_LCEMGP(self):
        for dtype, fixed_noise in ((torch.float, True), (torch.double, False)):
            _, (train_x, train_y, train_yvar) = gen_multi_task_dataset(
                yvar=0.01 if fixed_noise else None, dtype=dtype, device=self.device
            )
            task_feature = 0
            model = LCEMGP(
                train_X=train_x,
                train_Y=train_y,
                task_feature=task_feature,
                train_Yvar=train_yvar,
            )

            self.assertIsInstance(model, LCEMGP)
            self.assertIsInstance(model, MultiTaskGP)
            self.assertIsNone(model.context_emb_feature)
            self.assertIsInstance(model.context_cat_feature, Tensor)
            self.assertEqual(model.context_cat_feature.shape, torch.Size([2, 1]))
            self.assertEqual(len(model.emb_layers), 1)
            self.assertEqual(model.emb_dims, [(2, 1)])

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 1}})

            context_covar = model._eval_context_covar()
            self.assertIsInstance(context_covar, LinearOperator)
            self.assertEqual(context_covar.shape, torch.Size([2, 2]))

            embeddings = model._task_embeddings()
            self.assertIsInstance(embeddings, Tensor)
            self.assertEqual(embeddings.shape, torch.Size([2, 1]))

            test_x = train_x[:5]
            self.assertIsInstance(model(test_x), MultivariateNormal)

            # test posterior
            posterior_f = model.posterior(test_x[:, task_feature + 1 :])
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.distribution, MultitaskMultivariateNormal)

            # test posterior w/ single output index
            posterior_f = model.posterior(
                test_x[:, task_feature + 1 :], output_indices=[0]
            )
            self.assertIsInstance(posterior_f, GPyTorchPosterior)
            self.assertIsInstance(posterior_f.distribution, MultivariateNormal)

            # test input embs_dim_list (one categorical feature)
            # test input pre-trained emb context_emb_feature
            model2 = LCEMGP(
                train_X=train_x,
                train_Y=train_y,
                task_feature=task_feature,
                embs_dim_list=[2],  # increase dim from 1 to 2
                context_emb_feature=torch.tensor([[0.2], [0.3]]),
            )
            self.assertIsInstance(model2, LCEMGP)
            self.assertIsInstance(model2, MultiTaskGP)
            self.assertIsNotNone(model2.context_emb_feature)
            self.assertIsInstance(model2.context_cat_feature, Tensor)
            self.assertEqual(model2.context_cat_feature.shape, torch.Size([2, 1]))
            self.assertEqual(len(model2.emb_layers), 1)
            self.assertEqual(model2.emb_dims, [(2, 2)])

            embeddings2 = model2._task_embeddings()
            self.assertIsInstance(embeddings2, Tensor)
            self.assertEqual(embeddings2.shape, torch.Size([2, 3]))

            # Check task_covar_matrix against previous implementation.
            task_idcs = torch.randint(
                low=0, high=2, size=torch.Size([8, 32, 1]), device=self.device
            )
            covar_matrix = model._eval_context_covar()
            previous_covar = InterpolatedLinearOperator(
                base_linear_op=covar_matrix,
                left_interp_indices=task_idcs,
                right_interp_indices=task_idcs,
            ).to_dense()
            self.assertAllClose(previous_covar, model.task_covar_module(task_idcs))
            custom_covar_module = MaternKernel()
            model_custom_covar = LCEMGP(
                train_X=train_x,
                train_Y=train_y,
                task_feature=task_feature,
                embs_dim_list=[2],  # increase dim from 1 to 2
                context_emb_feature=torch.tensor([[0.2], [0.3]]),
                covar_module=custom_covar_module,
            )
            self.assertIsInstance(model_custom_covar.covar_module, MaternKernel)

    def test_construct_inputs(self) -> None:
        for with_embedding_inputs, yvar, skip_task_features_in_datasets in zip(
            (True, False), (None, 0.01), (True, False), strict=True
        ):
            dataset, (train_x, train_y, train_yvar) = gen_multi_task_dataset(
                yvar=yvar,
                skip_task_features_in_datasets=skip_task_features_in_datasets,
                dtype=torch.double,
                device=self.device,
            )
            model_inputs = LCEMGP.construct_inputs(
                training_data=dataset,
                task_feature=0,
                embs_dim_list=[2] if with_embedding_inputs else None,
                context_emb_feature=(
                    torch.tensor([[0.2], [0.3]]) if with_embedding_inputs else None
                ),
                context_cat_feature=(
                    torch.tensor([[0.4], [0.5]]) if with_embedding_inputs else None
                ),
            )
            # Check that the model inputs are valid.
            model = LCEMGP(**model_inputs)
            # Check that the model inputs are as expected.
            self.assertEqual(model.all_tasks, [0, 1])
            if skip_task_features_in_datasets:
                # In this case, the task feature is appended at the end.
                self.assertAllClose(model_inputs.pop("train_X"), train_x[..., [1, 0]])
                # all_tasks is inferred from data when task features are omitted.
                self.assertEqual(model_inputs.pop("all_tasks"), [0, 1])
            else:
                self.assertAllClose(model_inputs.pop("train_X"), train_x)
            self.assertAllClose(model_inputs.pop("train_Y"), train_y)
            if train_yvar is not None:
                self.assertAllClose(model_inputs.pop("train_Yvar"), train_yvar)
            if with_embedding_inputs:
                self.assertEqual(model_inputs.pop("embs_dim_list"), [2])
                self.assertAllClose(
                    model_inputs.pop("context_emb_feature"),
                    torch.tensor([[0.2], [0.3]]),
                )
                self.assertAllClose(
                    model_inputs.pop("context_cat_feature"),
                    torch.tensor([[0.4], [0.5]]),
                )
            self.assertEqual(model_inputs.pop("task_feature"), 0)
            self.assertIsNone(model_inputs.pop("output_tasks"))
            # Check that there are no unexpected inputs.
            self.assertEqual(model_inputs, {})
