#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.models.kernels.contextual_lcea import (
    get_order,
    get_permutation,
    is_contiguous,
    LCEAKernel,
)

from botorch.models.kernels.contextual_sac import SACKernel
from botorch.utils.testing import BotorchTestCase
from gpytorch.kernels.rbf_kernel import RBFKernel
from torch import Tensor
from torch.nn import ModuleDict


class ContextualKernelTest(BotorchTestCase):
    def test_SACKernel(self):
        decomposition = {"1": [0, 3], "2": [1, 2]}
        kernel = SACKernel(decomposition=decomposition, batch_shape=torch.Size([]))

        self.assertIsInstance(kernel.kernel_dict, ModuleDict)
        self.assertIsInstance(kernel.base_kernel, RBFKernel)
        self.assertDictEqual(kernel.decomposition, decomposition)

        # test diag works well for lazy tensor
        x1 = torch.rand(5, 4)
        x2 = torch.rand(5, 4)
        res = kernel(x1, x2).to_dense()
        res_diag = kernel(x1, x2, diag=True)
        self.assertLess(torch.linalg.norm(res_diag - res.diag()), 1e-4)

        # test raise of ValueError
        with self.assertRaises(ValueError):
            SACKernel(decomposition={"1": [0, 3], "2": [1]}, batch_shape=torch.Size([]))

    def testLCEAKernel(self):
        decomposition = {"1": [0, 3], "2": [1, 2]}
        num_contexts = len(decomposition)
        kernel = LCEAKernel(decomposition=decomposition, batch_shape=torch.Size([]))
        # test init
        self.assertListEqual(kernel.context_list, ["1", "2"])

        self.assertIsInstance(kernel.base_kernel, RBFKernel)
        self.assertIsInstance(kernel.task_covar_module, RBFKernel)
        self.assertEqual(kernel.permutation, [0, 3, 1, 2])

        # test raise of ValueError
        with self.assertRaisesRegex(
            ValueError, "The number of parameters needs to be same across all contexts."
        ):
            LCEAKernel(
                decomposition={"1": [0, 1], "2": [2]}, batch_shape=torch.Size([])
            )

        # test set_outputscale_list
        kernel.initialize(outputscale_list=[0.5, 0.5])
        actual_value = torch.tensor([0.5, 0.5]).view_as(kernel.outputscale_list)
        self.assertLess(torch.linalg.norm(kernel.outputscale_list - actual_value), 1e-5)

        self.assertTrue(kernel.train_embedding)
        self.assertEqual(kernel.num_contexts, num_contexts)
        self.assertEqual(kernel.n_embs, 1)
        self.assertIsNone(kernel.context_emb_feature)
        self.assertIsInstance(kernel.context_cat_feature, Tensor)
        self.assertEqual(len(kernel.emb_layers), 1)
        self.assertListEqual(kernel.emb_dims, [(num_contexts, 1)])

        context_covar = kernel._eval_context_covar()
        self.assertIsInstance(context_covar, Tensor)
        self.assertEqual(context_covar.shape, torch.Size([num_contexts, num_contexts]))

        embeddings = kernel._task_embeddings()
        self.assertIsInstance(embeddings, Tensor)
        self.assertEqual(embeddings.shape, torch.Size([num_contexts, 1]))

        self.assertIsInstance(kernel.outputscale_list, Tensor)
        self.assertEqual(kernel.outputscale_list.shape, torch.Size([num_contexts]))

        # test diag works well for lazy tensor
        num_obs, num_contexts, input_dim = 5, 2, 2
        x1 = torch.rand(num_obs, num_contexts * input_dim)
        x2 = torch.rand(num_obs, num_contexts * input_dim)
        res = kernel(x1, x2).to_dense()
        res_diag = kernel(x1, x2, diag=True)
        self.assertAllClose(res_diag, res.diag(), atol=1e-4)

        # test batch evaluation
        batch_dim = 3
        x1 = torch.rand(batch_dim, num_obs, num_contexts * input_dim)
        x2 = torch.rand(batch_dim, num_obs, num_contexts * input_dim)
        res = kernel(x1, x2).to_dense()
        self.assertEqual(res.shape, torch.Size([batch_dim, num_obs, num_obs]))

        # testing efficient `einsum` with naive `sum` implementation
        context_covar = kernel._eval_context_covar()
        if x1.dim() > context_covar.dim():
            context_covar = context_covar.expand(
                x1.shape[:-1] + torch.Size([x2.shape[-2]]) + context_covar.shape
            )
        base_covar_perm = kernel._eval_base_covar_perm(x1, x2)
        expected_res = (context_covar * base_covar_perm).sum(dim=-2).sum(dim=-1)
        self.assertAllClose(expected_res, res)

        # diagonal batch evaluation
        res_diag = kernel(x1, x2, diag=True).to_dense()
        expected_res_diag = torch.diagonal(expected_res, dim1=-1, dim2=-2)
        self.assertAllClose(expected_res_diag, res_diag)

        # test input context_weight,
        # test input embs_dim_list (one categorical feature)
        # test input context_cat_feature
        embs_dim_list = [2]
        kernel2 = LCEAKernel(
            decomposition=decomposition,
            context_weight_dict={"1": 0.5, "2": 0.8},
            cat_feature_dict={"1": [0], "2": [1]},
            embs_dim_list=embs_dim_list,  # increase dim from 1 to 2
            batch_shape=torch.Size([]),
        )

        self.assertEqual(kernel2.num_contexts, num_contexts)
        self.assertEqual(kernel2.n_embs, 2)
        self.assertIsNone(kernel2.context_emb_feature)
        self.assertIsInstance(kernel2.context_cat_feature, Tensor)
        self.assertEqual(
            kernel2.context_cat_feature.shape, torch.Size([num_contexts, 1])
        )
        self.assertEqual(len(kernel2.emb_layers), 1)
        self.assertListEqual(kernel2.emb_dims, [(num_contexts, embs_dim_list[0])])

        context_covar2 = kernel2._eval_context_covar()
        self.assertIsInstance(context_covar2, Tensor)
        self.assertEqual(context_covar2.shape, torch.Size([num_contexts, num_contexts]))

        # test input pre-trained embedding
        kernel3 = LCEAKernel(
            decomposition=decomposition,
            embs_feature_dict={"1": [0.2], "2": [0.5]},
            batch_shape=torch.Size([]),
        )
        self.assertEqual(kernel3.num_contexts, num_contexts)
        self.assertEqual(kernel3.n_embs, 2)
        self.assertIsNotNone(kernel3.context_emb_feature)
        self.assertIsInstance(kernel3.context_emb_feature, Tensor)
        self.assertIsInstance(kernel3.context_cat_feature, Tensor)
        self.assertEqual(
            kernel3.context_cat_feature.shape, torch.Size([num_contexts, 1])
        )
        self.assertListEqual(kernel3.emb_dims, [(num_contexts, 1)])
        embeddings3 = kernel3._task_embeddings()
        self.assertEqual(embeddings3.shape, torch.Size([num_contexts, 2]))

        # test only use pre-trained embedding
        kernel4 = LCEAKernel(
            decomposition=decomposition,
            train_embedding=False,
            embs_feature_dict={"1": [0.2], "2": [0.5]},
            batch_shape=torch.Size([]),
        )
        self.assertEqual(kernel4.n_embs, 1)
        self.assertIsNotNone(kernel4.context_emb_feature)
        self.assertIsInstance(kernel4.context_emb_feature, Tensor)
        self.assertIsInstance(kernel4.context_cat_feature, Tensor)
        embeddings4 = kernel4._task_embeddings()
        self.assertEqual(embeddings4.shape, torch.Size([num_contexts, 1]))

        # test batch
        kernel5 = LCEAKernel(decomposition=decomposition, batch_shape=torch.Size([3]))
        self.assertEqual(kernel5.n_embs, 1)  # one dim cat
        self.assertListEqual(kernel5.emb_dims, [(num_contexts, 1)])

        embeddings_batch = kernel5._task_embeddings_batch()
        self.assertIsInstance(embeddings_batch, Tensor)
        self.assertEqual(embeddings_batch.shape, torch.Size([3, num_contexts, 1]))

        context_covar5 = kernel5._eval_context_covar()
        self.assertIsInstance(context_covar5, Tensor)
        self.assertEqual(
            context_covar5.shape, torch.Size([3, num_contexts, num_contexts])
        )

        # test batch with pre-trained features
        kernel6 = LCEAKernel(
            decomposition=decomposition,
            batch_shape=torch.Size([3]),
            embs_feature_dict={"1": [0.2], "2": [0.5]},
        )
        self.assertEqual(kernel6.n_embs, 2)  # one dim cat + one dim pre-train
        self.assertListEqual(kernel6.emb_dims, [(num_contexts, 1)])  # one dim for cat

        embeddings_batch = kernel6._task_embeddings_batch()
        self.assertIsInstance(embeddings_batch, Tensor)
        self.assertEqual(
            embeddings_batch.shape, torch.Size([3, num_contexts, num_contexts])
        )

        context_covar6 = kernel6._eval_context_covar()
        self.assertIsInstance(context_covar6, Tensor)
        self.assertEqual(
            context_covar6.shape, torch.Size([3, num_contexts, num_contexts])
        )

    def test_get_permutation(self):
        decomp = {"a": [0, 1], "b": [2, 3]}
        permutation = get_permutation(decomp)
        self.assertIsNone(permutation)
        # order mismatch
        decomp = {"a": [1, 0], "b": [2, 3]}
        permutation = get_permutation(decomp)
        self.assertEqual(permutation, [0, 1, 2, 3])
        # non-contiguous
        decomp = {"a": [0, 2], "b": [1, 3]}
        permutation = get_permutation(decomp)
        self.assertEqual(permutation, [0, 2, 1, 3])

    def test_is_contiguous(self):
        self.assertFalse(is_contiguous([0, 2]))
        self.assertTrue(is_contiguous([0, 1]))

    def test_get_order(self):
        self.assertEqual(get_order([1, 10, 3]), [1, 1, 0])
