# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.utils.testing import BotorchTestCase
from botorch_community.utils.stat_dist import mvn_hellinger_distance, mvn_kl_divergence


class TestStatDist(BotorchTestCase):
    def test_mvn_kl_divergence(self):
        q_shapes = [(1,), (5,)]
        mvn1_batch_shapes = [(4,), (4, 5), (4, 5), (4, 5, 7), (1, 5, 9, 2), (1, 2, 3)]
        mvn2_batch_shapes = [(4,), (4, 5), (1, 5), (4, 5, 1), (7, 5, 9, 2), (3, 2, 1)]

        for mvn1_shape, mvn2_shape in zip(mvn1_batch_shapes, mvn2_batch_shapes):
            for qdim in q_shapes:
                dist1_mean = torch.zeros(mvn1_shape + qdim + (1,))
                dist2_mean = torch.zeros(mvn2_shape + qdim + (1,))

                dist1_cov = torch.diag_embed(
                    torch.ones(dist1_mean.shape[:-1]), dim1=-2, dim2=-1
                )
                dist2_cov = torch.diag_embed(
                    torch.ones(dist2_mean.shape[:-1]), dim1=-2, dim2=-1
                )

                # creating a diagonal matrix of covariances
                res = mvn_kl_divergence(dist1_mean, dist2_mean, dist1_cov, dist2_cov)

                correct_shape = torch.broadcast_shapes(mvn1_shape, mvn2_shape) + (1,)
                self.assertEqual(res.shape, correct_shape)
                self.assertTrue(torch.all(res == 0))

        perturb_mean = dist1_mean + 1
        perturb_cov = dist2_cov + 0.01
        permean_res = mvn_kl_divergence(perturb_mean, dist2_mean, dist1_cov, dist2_cov)
        percov_res = mvn_kl_divergence(dist1_mean, dist2_mean, dist1_cov, perturb_cov)
        self.assertTrue(torch.all(permean_res > 0))
        self.assertTrue(torch.all(percov_res > 0))

    def test_mvn_hellinger_distance(self):
        q_shapes = [(1,), (5,)]
        mvn1_batch_shapes = [(4,), (4, 5), (4, 5), (4, 5, 7), (1, 5, 9, 2), (1, 2, 3)]
        mvn2_batch_shapes = [(4,), (4, 5), (1, 5), (4, 5, 1), (7, 5, 9, 2), (3, 2, 1)]

        for mvn1_shape, mvn2_shape in zip(mvn1_batch_shapes, mvn2_batch_shapes):
            for qdim in q_shapes:
                dist1_mean = torch.zeros(mvn1_shape + qdim + (1,))
                dist2_mean = torch.zeros(mvn2_shape + qdim + (1,))

                dist1_cov = torch.diag_embed(
                    torch.ones(dist1_mean.shape[:-1]), dim1=-2, dim2=-1
                )
                dist2_cov = torch.diag_embed(
                    torch.ones(dist2_mean.shape[:-1]), dim1=-2, dim2=-1
                )

                # creating a diagonal matrix of covariances
                res = mvn_hellinger_distance(
                    dist1_mean, dist2_mean, dist1_cov, dist2_cov
                )

                correct_shape = torch.broadcast_shapes(mvn1_shape, mvn2_shape) + (1,)
                self.assertEqual(res.shape, correct_shape)
                self.assertTrue(torch.all(res == 0))

        perturb_mean = dist1_mean + 1
        perturb_cov = dist2_cov + 0.01
        permean_res = mvn_hellinger_distance(
            perturb_mean, dist2_mean, dist1_cov, dist2_cov
        )
        percov_res = mvn_hellinger_distance(
            dist1_mean, dist2_mean, dist1_cov, perturb_cov
        )
        self.assertTrue(torch.all(permean_res > 0))
        self.assertTrue(torch.all(percov_res > 0))

    def test_mvn_hellinger_distance_numerical_stability(self):
        # Two almost equal distributions. Distance is approximatly 0.
        dist1_mean = torch.tensor([[-0.4615126826879162]], dtype = torch.float64)
        dist2_mean = torch.tensor([[-0.46151268268791173]], dtype = torch.float64)
        dist1_cov = torch.tensor([[0.12132352941175625]], dtype = torch.float64)
        dist2_cov = torch.tensor([[0.12132352941176472]], dtype = torch.float64)
        
        res = mvn_hellinger_distance(
            dist1_mean, dist2_mean, dist1_cov, dist2_cov
        )
        self.assertFalse(res.isnan().any())
        self.assertAllClose(res, torch.tensor([0.0], dtype = torch.float64))
