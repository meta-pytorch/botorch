#! /usr/bin/env python3

import unittest

import torch
from botorch.acquisition.batch_modules import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)

from ..mock import MockModel, MockPosterior


class TestQExpectedImprovement(unittest.TestCase):
    def test_q_expected_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            samples = torch.zeros(1, 1, 1, device=device, dtype=dtype)
            # the event shape in MockPosterior will be `b x q x t` = 1 x 1 x 1
            mm = MockModel(MockPosterior(samples=samples))
            # X is `q x d` = 1 x 1
            X = torch.zeros(1, 1, device=device, dtype=dtype)  # dummy for typing

            # basic test
            acq_module = qExpectedImprovement(
                model=mm, best_f=0, mc_samples=2, qmc=False
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res = acq_module(X)
            self.assertEqual(res.item(), 0)

            # basic test, fixed seed
            acq_module = qExpectedImprovement(
                model=mm, best_f=0, mc_samples=2, qmc=False, seed=12345
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res = acq_module(X)
            self.assertEqual(res.item(), 0)
            self.assertEqual(acq_module._base_samples_q_batch_size, 1)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 1, 1]))

            # basic test, qmc
            acq_module = qExpectedImprovement(
                model=mm, best_f=0, mc_samples=2, qmc=True
            )
            res1 = acq_module(X)
            self.assertEqual(res1.item(), 0)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 1, 1]))
            bs1 = acq_module.base_samples.clone()
            acq_module(X)
            self.assertFalse(torch.equal(acq_module.base_samples, bs1))

            # basic test, qmc, fixed seed
            acq_module = qExpectedImprovement(
                model=mm, best_f=0, mc_samples=2, qmc=True, seed=12345
            )
            res1 = acq_module(X)
            self.assertEqual(res1.item(), 0)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 1, 1]))
            bs1 = acq_module.base_samples.clone()
            acq_module(X)
            self.assertTrue(torch.equal(acq_module.base_samples, bs1))

            # test X_pending
            samples = torch.zeros(1, 2, 1, device=device, dtype=dtype)
            # the event shape in MockPosterior is will be `b x q x t` = 1 x 2 x 1
            mm = MockModel(MockPosterior(samples=samples))
            # X_pending has shape `m x d` = 1 x 1
            X_pending = torch.zeros(1, 1, device=device, dtype=dtype)

            # basic test X_pending
            acq_module = qExpectedImprovement(
                model=mm, best_f=0, mc_samples=2, X_pending=X_pending, qmc=False
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res2 = acq_module(X)
            self.assertEqual(res2.item(), 0)

            # basic test X_pending, fixed seed
            acq_module = qExpectedImprovement(
                model=mm,
                best_f=0,
                mc_samples=2,
                X_pending=X_pending,
                qmc=False,
                seed=12345,
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res2 = acq_module(X)
            self.assertEqual(res2.item(), 0)
            self.assertEqual(acq_module._base_samples_q_batch_size, 2)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 2, 1]))

            # test X_pending, qmc
            acq_module = qExpectedImprovement(
                model=mm, best_f=0, mc_samples=2, X_pending=X_pending, qmc=True
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res2 = acq_module(X)
            self.assertEqual(res2.item(), 0)
            self.assertEqual(acq_module._base_samples_q_batch_size, 2)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs2 = acq_module.base_samples.clone()
            acq_module(X)
            self.assertFalse(torch.equal(acq_module.base_samples, bs2))

            # test X_pending, qmc, fixed seed
            acq_module = qExpectedImprovement(
                model=mm,
                best_f=0,
                mc_samples=2,
                X_pending=X_pending,
                qmc=True,
                seed=12345,
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res2 = acq_module(X)
            self.assertEqual(res2.item(), 0)
            self.assertEqual(acq_module._base_samples_q_batch_size, 2)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs2 = acq_module.base_samples.clone()
            acq_module(X)
            self.assertTrue(torch.equal(acq_module.base_samples, bs2))

            # test X_pending w/ batch mode
            # the event shape is `b x q x t` = 2 x 2 x 1
            samples = torch.zeros(2, 2, 1, device=device, dtype=dtype)
            samples[0, 0, 0] = 1
            mm = MockModel(MockPosterior(samples=samples))

            # test X_pending w/ batch mode
            acq_module = qExpectedImprovement(
                model=mm, best_f=0, mc_samples=2, X_pending=X_pending, qmc=False
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res3 = acq_module(X.unsqueeze(0))
            self.assertEqual(res3[0].item(), 1)
            self.assertEqual(res3[1].item(), 0)

            # test X_pending w/ batch mode, fixed seed
            acq_module = qExpectedImprovement(
                model=mm,
                best_f=0,
                mc_samples=2,
                X_pending=X_pending,
                qmc=False,
                seed=12345,
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res3 = acq_module(X.unsqueeze(0))  # 1-dim batch
            self.assertEqual(res3[0].item(), 1)
            self.assertEqual(res3[1].item(), 0)
            self.assertEqual(acq_module._base_samples_q_batch_size, 2)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs3 = acq_module.base_samples.clone()
            acq_module(X.unsqueeze(0))
            self.assertTrue(torch.equal(acq_module.base_samples, bs3))
            res3 = acq_module(X.expand(2, 1, 1))  # 2-dim batch
            self.assertEqual(res3[0].item(), 1)
            self.assertEqual(res3[1].item(), 0)
            self.assertEqual(acq_module._base_samples_q_batch_size, 2)
            # the base samples should have the batch dim collapsed
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs3 = acq_module.base_samples.clone()
            acq_module(X.expand(2, 1, 1))
            self.assertTrue(torch.equal(acq_module.base_samples, bs3))

            # test X_pending w/ batch mode, qmc
            acq_module = qExpectedImprovement(
                model=mm, best_f=0, mc_samples=2, X_pending=X_pending, qmc=True
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res3 = acq_module(X.unsqueeze(0))
            self.assertEqual(res3[0].item(), 1)
            self.assertEqual(res3[1].item(), 0)
            self.assertEqual(acq_module._base_samples_q_batch_size, 2)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs3 = acq_module.base_samples.clone()
            acq_module(X.unsqueeze(0))
            self.assertFalse(torch.equal(acq_module.base_samples, bs3))

            # test X_pending w/ batch mode, qmc, fixed seed
            acq_module = qExpectedImprovement(
                model=mm,
                best_f=0,
                mc_samples=2,
                X_pending=X_pending,
                qmc=True,
                seed=12345,
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res3 = acq_module(X.unsqueeze(0))  # 1-dim batch
            self.assertEqual(res3[0].item(), 1)
            self.assertEqual(res3[1].item(), 0)
            self.assertEqual(acq_module._base_samples_q_batch_size, 2)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs3 = acq_module.base_samples.clone()
            acq_module(X.unsqueeze(0))
            self.assertTrue(torch.equal(acq_module.base_samples, bs3))
            res3 = acq_module(X.expand(2, 1, 1))  # 2-dim batch
            self.assertEqual(res3[0].item(), 1)
            self.assertEqual(res3[1].item(), 0)
            self.assertEqual(acq_module._base_samples_q_batch_size, 2)
            # the base samples should have the batch dim collapsed
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs3 = acq_module.base_samples.clone()
            acq_module(X.expand(2, 1, 1))
            self.assertTrue(torch.equal(acq_module.base_samples, bs3))

    def test_q_expected_improvement_cuda(self):
        if torch.cuda.is_available():
            self.test_q_expected_improvement(cuda=True)


class TestQNoisyExpectedImprovement(unittest.TestCase):
    def test_q_noisy_expected_improvement(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            # the event shape is `b x q x t` = 1 x 2 x 1
            samples_noisy = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
            samples_noisy = samples_noisy.view(1, 2, 1)
            # X_observed is `q' x d` = 1 x 1
            X_observed = torch.zeros(1, 1, device=device, dtype=dtype)
            mm_noisy = MockModel(MockPosterior(samples=samples_noisy))
            # X is `q x d` = 1 x 1
            X = torch.zeros(1, 1, device=device, dtype=dtype)

            # basic test
            acq_module = qNoisyExpectedImprovement(
                model=mm_noisy, X_observed=X_observed, mc_samples=2, qmc=False
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res = acq_module(X)
            self.assertEqual(res.item(), 1)

            # basic test, fixed seed
            acq_module = qNoisyExpectedImprovement(
                model=mm_noisy,
                X_observed=X_observed,
                mc_samples=2,
                qmc=False,
                seed=12345,
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res = acq_module(X)
            self.assertEqual(res.item(), 1)
            self.assertEqual(acq_module._base_samples_q_batch_size, 2)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 2, 1]))

            # basic test, qmc
            acq_module = qNoisyExpectedImprovement(
                model=mm_noisy, X_observed=X_observed, mc_samples=2, qmc=True
            )
            res1 = acq_module(X)
            self.assertEqual(res1.item(), 1)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs1 = acq_module.base_samples.clone()
            acq_module(X)
            self.assertFalse(torch.equal(acq_module.base_samples, bs1))

            # basic test, qmc, fixed seed
            acq_module = qNoisyExpectedImprovement(
                model=mm_noisy,
                X_observed=X_observed,
                mc_samples=2,
                qmc=True,
                seed=12345,
            )
            res1 = acq_module(X)
            self.assertEqual(res1.item(), 1)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 2, 1]))
            bs1 = acq_module.base_samples.clone()
            acq_module(X)
            self.assertTrue(torch.equal(acq_module.base_samples, bs1))

            # test X_pending
            # the event shape is `b x q x t` = 1 x 3 x 1
            samples_noisy = torch.tensor(
                [1.0, 0.0, 0.0], device=device, dtype=dtype
            ).view(1, 3, 1)
            mm_noisy = MockModel(MockPosterior(samples=samples_noisy))
            # X_pending has shape `m x d` = 1 x 1
            X_pending = torch.zeros(1, 1, device=device, dtype=dtype)

            # basic test X_pending
            acq_module = qNoisyExpectedImprovement(
                model=mm_noisy,
                X_observed=X_observed,
                mc_samples=2,
                X_pending=X_pending,
                qmc=False,
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res2 = acq_module(X)
            self.assertEqual(res2.item(), 1)

            # basic test X_pending, fixed seed
            acq_module = qNoisyExpectedImprovement(
                model=mm_noisy,
                X_observed=X_observed,
                mc_samples=2,
                X_pending=X_pending,
                qmc=False,
                seed=12345,
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res2 = acq_module(X)
            self.assertEqual(res2.item(), 1)
            self.assertEqual(acq_module._base_samples_q_batch_size, 3)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 3, 1]))

            # test X_pending, qmc
            acq_module = qNoisyExpectedImprovement(
                model=mm_noisy,
                X_observed=X_observed,
                mc_samples=2,
                X_pending=X_pending,
                qmc=True,
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res2 = acq_module(X)
            self.assertEqual(res2.item(), 1)
            self.assertEqual(acq_module._base_samples_q_batch_size, 3)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 3, 1]))
            bs2 = acq_module.base_samples.clone()
            acq_module(X)
            self.assertFalse(torch.equal(acq_module.base_samples, bs2))

            # test X_pending, qmc, fixed seed
            acq_module = qNoisyExpectedImprovement(
                model=mm_noisy,
                X_observed=X_observed,
                mc_samples=2,
                X_pending=X_pending,
                qmc=True,
                seed=12345,
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res2 = acq_module(X)
            self.assertEqual(res2.item(), 1)
            self.assertEqual(acq_module._base_samples_q_batch_size, 3)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 3, 1]))
            bs2 = acq_module.base_samples.clone()
            acq_module(X)
            self.assertTrue(torch.equal(acq_module.base_samples, bs2))

            # test X_pending w/ batch mode
            # the event shape is `b x q x t` = 2 x 3 x 1
            samples_noisy = torch.zeros(2, 3, 1, device=device, dtype=dtype)
            samples_noisy[0, 0, 0] = 1.0
            mm_noisy = MockModel(MockPosterior(samples=samples_noisy))
            X_pending = torch.zeros(1, 1, device=device, dtype=dtype)

            # test X_pending w/ batch mode
            acq_module = qNoisyExpectedImprovement(
                model=mm_noisy,
                X_observed=X_observed,
                mc_samples=2,
                X_pending=X_pending,
                qmc=False,
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res3 = acq_module(X.unsqueeze(0))
            self.assertEqual(res3[0].item(), 1)
            self.assertEqual(res3[1].item(), 0)

            # test X_pending w/ batch mode, fixed seed
            acq_module = qNoisyExpectedImprovement(
                model=mm_noisy,
                X_observed=X_observed,
                mc_samples=2,
                X_pending=X_pending,
                qmc=False,
                seed=12345,
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res3 = acq_module(X.unsqueeze(0))  # 1-dim batch
            self.assertEqual(res3[0].item(), 1)
            self.assertEqual(res3[1].item(), 0)
            self.assertEqual(acq_module._base_samples_q_batch_size, 3)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 3, 1]))
            bs3 = acq_module.base_samples.clone()
            acq_module(X.unsqueeze(0))
            self.assertTrue(torch.equal(acq_module.base_samples, bs3))
            res3 = acq_module(X.expand(2, 1, 1))  # 2-dim batch
            self.assertEqual(res3[0].item(), 1)
            self.assertEqual(res3[1].item(), 0)
            self.assertEqual(acq_module._base_samples_q_batch_size, 3)
            # the base samples should have the batch dim collapsed
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 3, 1]))
            bs3 = acq_module.base_samples.clone()
            acq_module(X.expand(2, 1, 1))
            self.assertTrue(torch.equal(acq_module.base_samples, bs3))

            # test X_pending w/ batch mode, qmc
            acq_module = qNoisyExpectedImprovement(
                model=mm_noisy,
                X_observed=X_observed,
                mc_samples=2,
                X_pending=X_pending,
                qmc=True,
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res3 = acq_module(X.unsqueeze(0))
            self.assertEqual(res3[0].item(), 1)
            self.assertEqual(res3[1].item(), 0)
            self.assertEqual(acq_module._base_samples_q_batch_size, 3)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 3, 1]))
            bs3 = acq_module.base_samples.clone()
            acq_module(X.unsqueeze(0))
            self.assertFalse(torch.equal(acq_module.base_samples, bs3))

            # test X_pending w/ batch mode, qmc, fixed seed
            acq_module = qNoisyExpectedImprovement(
                model=mm_noisy,
                X_observed=X_observed,
                mc_samples=2,
                X_pending=X_pending,
                qmc=True,
                seed=12345,
            )
            self.assertIsNone(acq_module._base_samples_q_batch_size)
            res3 = acq_module(X.unsqueeze(0))  # 1-dim batch
            self.assertEqual(res3[0].item(), 1)
            self.assertEqual(res3[1].item(), 0)
            self.assertEqual(acq_module._base_samples_q_batch_size, 3)
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 3, 1]))
            bs3 = acq_module.base_samples.clone()
            acq_module(X.unsqueeze(0))
            self.assertTrue(torch.equal(acq_module.base_samples, bs3))
            res3 = acq_module(X.expand(2, 1, 1))  # 2-dim batch
            self.assertEqual(res3[0].item(), 1)
            self.assertEqual(res3[1].item(), 0)
            self.assertEqual(acq_module._base_samples_q_batch_size, 3)
            # the base samples should have the batch dim collapsed
            self.assertEqual(acq_module.base_samples.shape, torch.Size([2, 1, 3, 1]))
            bs3 = acq_module.base_samples.clone()
            acq_module(X.expand(2, 1, 1))
            self.assertTrue(torch.equal(acq_module.base_samples, bs3))

    def test_q_noisy_expected_improvement_cuda(self):
        if torch.cuda.is_available():
            self.test_q_noisy_expected_improvement(cuda=True)
