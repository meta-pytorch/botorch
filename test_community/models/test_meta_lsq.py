#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from botorch.acquisition import (
    LogExpectedImprovement,
    qKnowledgeGradient,
    qLogExpectedImprovement,
)
from botorch.optim import optimize_acqf
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase
from botorch_community.models.meta_lsq import MetaLSQ, MetaLSQModel, MetaLSQTask
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal


def _sinusoid_task(gen, n=10, n_query=50, grad=True, **tkwargs):
    r"""A task from the sinusoid family ``a * sin(x + p)`` on ``[-5, 5]``."""
    amp = torch.rand(1, generator=gen).item() * 4.9 + 0.1
    phase = torch.rand(1, generator=gen).item() * math.pi
    X = (torch.rand(n, 1, generator=gen) * 10 - 5).to(**tkwargs)
    Xq = (torch.rand(n_query, 1, generator=gen) * 10 - 5).to(**tkwargs)
    Y = amp * torch.sin(X + phase)
    Yq = amp * torch.sin(Xq + phase)
    dY = (amp * torch.cos(X + phase)).unsqueeze(-1) if grad else None
    return MetaLSQTask(X, Y, Xq, Yq, dY)


def _two_output_data(n=8, **tkwargs):
    gen = torch.Generator().manual_seed(1)
    X = (torch.rand(n, 2, generator=gen) * 2 - 1).to(**tkwargs)
    Y = torch.stack([X.sin().sum(-1), X.cos().prod(-1)], dim=-1)
    J = torch.stack(
        [
            X.cos(),
            torch.stack(
                [-X[:, 0].sin() * X[:, 1].cos(), -X[:, 0].cos() * X[:, 1].sin()], dim=-1
            ),
        ],
        dim=-2,
    )  # n x m x d
    return X, Y, J


class TestMetaLSQ(BotorchTestCase):
    def test_init_and_validation(self) -> None:
        tkwargs = {"dtype": torch.double, "device": self.device}
        meta = MetaLSQ(dim=2, num_outputs=2, num_features=30, ard=True, **tkwargs)
        self.assertEqual(meta.sigma.shape, torch.Size([2]))
        self.assertEqual(meta.frequencies.shape, torch.Size([2, 30]))
        self.assertIsNone(meta.beta0)
        self.assertIn("N=30", repr(meta))
        self.assertIn("w=", repr(meta))
        self.assertTrue(meta.log_w.requires_grad)
        meta = MetaLSQ(dim=1, use_gradients=False, prior_mean=True, **tkwargs)
        self.assertFalse(meta.log_w.requires_grad)
        self.assertEqual(meta.beta0.shape, torch.Size([200, 1]))
        self.assertNotIn("w=", repr(meta))
        self.assertIn("sigma=[1, 1]", repr(MetaLSQ(dim=2, ard=True, **tkwargs)))
        with self.assertRaises(ValueError):
            MetaLSQ(dim=0)

        meta = MetaLSQ(dim=2, num_outputs=2, num_features=30, **tkwargs)
        X, Y, J = _two_output_data(**tkwargs)
        with self.assertRaisesRegex(ValueError, "dimension"):
            meta.adapt(X[:, :1], Y)
        with self.assertRaisesRegex(ValueError, "observations"):
            meta.adapt(X, Y[:, 0])
        with self.assertRaisesRegex(ValueError, "numbers of points"):
            meta.adapt(X, Y[:-1])
        with self.assertRaisesRegex(ValueError, "gradients"):
            meta.adapt(X, Y, J.transpose(-1, -2)[..., :1])
        no_grad = MetaLSQ(dim=2, num_outputs=2, use_gradients=False, **tkwargs)
        with self.assertRaisesRegex(ValueError, "use_gradients"):
            no_grad.adapt(X, Y, J)
        with self.assertRaisesRegex(ValueError, "noise_var"):
            meta.condition(X[:0], Y[:0])
        with self.assertRaisesRegex(ValueError, "empty"):
            meta.meta_fit([], num_steps=1)

    def test_adapt_matches_model_and_closed_form(self) -> None:
        tkwargs = {"dtype": torch.double, "device": self.device}
        for prior_mean, ard in ((False, False), (True, True)):
            meta = MetaLSQ(
                dim=2,
                num_outputs=2,
                num_features=40,
                mu=1e-3,
                w=2.0,
                ard=ard,
                prior_mean=prior_mean,
                **tkwargs,
            )
            if prior_mean:
                with torch.no_grad():
                    meta.beta0.normal_()
            if ard:
                with torch.no_grad():
                    meta.log_sigma.copy_(torch.tensor([2.0, 0.5], **tkwargs).log())
            sqrt_w = meta.w.sqrt().detach()
            X, Y, J = _two_output_data(**tkwargs)
            A, t = meta.design(X, Y, J)
            # 8 value rows and 8 * 2 gradient rows
            self.assertEqual(A.shape, torch.Size([24, 40]))
            self.assertEqual(t.shape, torch.Size([24, 2]))
            # the gradient rows are the gradients of the value rows
            H = torch.autograd.functional.jacobian(
                lambda x, meta=meta, Y=Y: meta.design(x, Y)[0].sum(0), X
            )  # N x n x d
            self.assertAllClose(A[8:], (sqrt_w * H.permute(1, 2, 0)).reshape(16, 40))
            self.assertAllClose(t[8:], (sqrt_w * J.transpose(-1, -2)).reshape(16, 2))
            beta = meta.adapt(X, Y, J)
            b0 = torch.zeros(40, 2, **tkwargs) if meta.beta0 is None else meta.beta0
            eye = torch.eye(40, **tkwargs)
            expected = b0 + torch.linalg.solve(
                A.T @ A + meta.mu * eye, A.T @ (t - A @ b0)
            )
            self.assertAllClose(beta.detach(), expected.detach())
            model = meta.condition(X, Y, J)
            self.assertIsInstance(model, MetaLSQModel)
            self.assertAllClose(model.coefficients, beta.detach())
            # the ML noise estimate equals t' (I + A A' / mu)^{-1} t / r
            r = t - A @ b0
            K = torch.eye(24, **tkwargs) + A @ A.T / meta.mu
            s2 = (r * torch.linalg.solve(K, r)).sum(0) / 24
            self.assertAllClose(model.noise_variance, s2.detach())
            # posterior mean and covariance
            Xt = torch.rand(5, 2, **tkwargs)
            post = model.posterior(Xt)
            self.assertIsInstance(post, GPyTorchPosterior)
            self.assertIsInstance(post.distribution, MultitaskMultivariateNormal)
            self.assertAllClose(post.mean, meta.predict(Xt, beta).detach())
            Phi = meta.design(Xt, torch.zeros(5, 2, **tkwargs))[0].detach()
            S = torch.linalg.inv(A.T @ A + meta.mu * eye).detach()
            for i in range(2):
                cov = model.noise_variance[i] * Phi @ S @ Phi.T
                sub = model.posterior(Xt, output_indices=[i])
                self.assertIsInstance(sub.distribution, MultivariateNormal)
                self.assertAllClose(sub.distribution.covariance_matrix, cov)
                noisy = model.posterior(Xt, output_indices=[i], observation_noise=True)
                self.assertAllClose(
                    noisy.variance[:, 0], cov.diag() + model.noise_variance[i]
                )
            # subset_output agrees with output_indices
            sub_model = model.subset_output([1])
            self.assertEqual(sub_model.num_outputs, 1)
            self.assertAllClose(
                sub_model.posterior(Xt).mean, post.mean[:, 1:], atol=1e-8
            )
            noise = torch.tensor([0.1, 0.2], **tkwargs)
            fixed = meta.condition(X, Y, J, noise_var=noise)
            self.assertAllClose(fixed.subset_output([1]).noise_variance, noise[1:])
        # more rows than features: the primal solve
        meta = MetaLSQ(dim=2, num_outputs=2, num_features=10, **tkwargs)
        self.assertAllClose(
            meta.adapt(X, Y, J).detach(), meta.condition(X, Y, J).coefficients
        )

    def test_gradients_help(self) -> None:
        # Fitting slopes as well as values improves a 4-point fit of a sinusoid.
        tkwargs = {"dtype": torch.double, "device": self.device}
        gen = torch.Generator().manual_seed(0)
        meta = MetaLSQ(dim=1, sigma=1.0, mu=1e-4, **tkwargs)
        errs = {True: [], False: []}
        for _ in range(20):
            task = _sinusoid_task(gen, n=4, **tkwargs)
            for grad in (True, False):
                dY = task.train_dY if grad else None
                mean = (
                    meta.condition(task.train_X, task.train_Y, dY)
                    .posterior(task.test_X)
                    .mean
                )
                errs[grad].append((mean - task.test_Y).pow(2).mean())
        self.assertLess(
            torch.stack(errs[True]).mean(), 0.5 * torch.stack(errs[False]).mean()
        )

    def test_meta_fit(self) -> None:
        tkwargs = {"dtype": torch.double, "device": self.device}
        gen = torch.Generator().manual_seed(123)
        heldout = [_sinusoid_task(gen, **tkwargs) for _ in range(50)]

        def sampler(k, g):
            return [_sinusoid_task(g, **tkwargs) for _ in range(k)]

        meta = MetaLSQ(dim=1, sigma=2.0, mu=1e-2, **tkwargs)
        with torch.no_grad():
            before = meta.query_loss(heldout).item()
        history = meta.meta_fit(sampler, num_steps=150, lr=2e-2)
        self.assertEqual(len(history), 150)
        with torch.no_grad():
            after = meta.query_loss(heldout).item()
        self.assertLess(after, 0.5 * before)
        # a fixed list of tasks and a custom loss
        meta = MetaLSQ(dim=1, use_gradients=False, prior_mean=True, **tkwargs)
        pool = [_sinusoid_task(gen, grad=False, **tkwargs) for _ in range(8)]
        calls = []

        def loss_fn(model, tasks):
            calls.append(len(tasks))
            return model.query_loss(tasks)

        meta.meta_fit(pool, num_steps=3, batch_size=2, loss_fn=loss_fn)
        self.assertEqual(calls, [2, 2, 2])
        # during the value warm-up the gradient weight gets no gradient
        warm = MetaLSQ(dim=1, **tkwargs)
        warm.meta_fit(sampler, num_steps=3, value_warmup_steps=3)
        self.assertEqual(warm.w.item(), 1.0)
        warm.meta_fit(sampler, num_steps=3, value_warmup_steps=2)
        self.assertNotEqual(warm.w.item(), 1.0)
        self.assertGreater(meta.beta0.abs().sum().item(), 0.0)

    def test_model_shapes_and_batches(self) -> None:
        tkwargs = {"dtype": torch.double, "device": self.device}
        X, Y, J = _two_output_data(**tkwargs)
        meta = MetaLSQ(dim=2, num_outputs=2, num_features=50, **tkwargs)
        model = meta.condition(X, Y, J)
        self.assertEqual(model.num_outputs, 2)
        self.assertEqual(model.batch_shape, torch.Size([]))
        post = model.posterior(torch.rand(4, 3, 2, **tkwargs))
        self.assertEqual(post.mean.shape, torch.Size([4, 3, 2]))
        self.assertTrue((post.variance > 0).all())
        samples = post.rsample(torch.Size([6]))
        self.assertEqual(samples.shape, torch.Size([6, 4, 3, 2]))
        # a fixed noise variance and observation noise as a tensor
        fixed = meta.condition(X, Y, J, noise_var=0.01)
        self.assertAllClose(fixed.noise_variance, torch.full((2,), 0.01, **tkwargs))
        Xt = torch.rand(3, 2, **tkwargs)
        noise = torch.full((1, 2), 0.5, **tkwargs)
        p0 = fixed.posterior(Xt)
        p1 = fixed.posterior(Xt, observation_noise=noise)
        self.assertAllClose(p1.variance, p0.variance + 0.5)
        # batched training data
        Xb, Yb = (
            X.expand(3, -1, -1),
            Y.expand(3, -1, -1) * torch.arange(1, 4, **tkwargs).view(3, 1, 1),
        )
        batched = meta.condition(Xb, Yb)
        self.assertEqual(batched.batch_shape, torch.Size([3]))
        self.assertEqual(batched.posterior(Xt).mean.shape, torch.Size([3, 3, 2]))
        single = meta.condition(X, 2 * Y)
        self.assertAllClose(batched.posterior(Xt).mean[1], single.posterior(Xt).mean)
        # an empty data set with a given noise variance gives the prior
        empty = meta.condition(X[:0], Y[:0], noise_var=1e-4)
        self.assertAllClose(empty.posterior(Xt).mean, torch.zeros(3, 2, **tkwargs))
        # posterior transform
        from botorch.acquisition.objective import ScalarizedPosteriorTransform

        tf = ScalarizedPosteriorTransform(weights=torch.tensor([1.0, -1.0], **tkwargs))
        self.assertEqual(
            model.posterior(Xt, posterior_transform=tf).mean.shape,
            torch.Size([3, 1]),
        )
        # float32 smoke test
        meta32 = MetaLSQ(dim=2, num_outputs=2, num_features=50, dtype=torch.float)
        m32 = meta32.condition(X.float(), Y.float(), J.float())
        self.assertEqual(m32.posterior(Xt.float()).mean.dtype, torch.float)

    def test_condition_and_fantasize(self) -> None:
        tkwargs = {"dtype": torch.double, "device": self.device}
        X, Y, J = _two_output_data(n=8, **tkwargs)
        meta = MetaLSQ(dim=2, num_outputs=2, num_features=50, **tkwargs)
        Xt = torch.rand(4, 2, **tkwargs)
        # conditioning in two steps equals conditioning at once
        full = meta.condition(X, Y, J)
        part = meta.condition(X[:5], Y[:5], J[:5]).condition_on_observations(
            X[5:], Y[5:], dY=J[5:]
        )
        self.assertAllClose(part.posterior(Xt).mean, full.posterior(Xt).mean)
        self.assertAllClose(part.noise_variance, full.noise_variance)
        self.assertAllClose(
            full.posterior(Xt).mean, meta.condition(X, Y, J).posterior(Xt).mean
        )
        # fantasize adds a batch dimension
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([3]))
        fm = full.fantasize(torch.rand(2, 2, **tkwargs), sampler)
        self.assertEqual(fm.batch_shape, torch.Size([3]))
        self.assertEqual(fm.posterior(Xt).mean.shape, torch.Size([3, 4, 2]))
        fm = full.fantasize(torch.rand(5, 2, 2, **tkwargs), sampler)
        self.assertEqual(fm.batch_shape, torch.Size([3, 5]))
        fm = full.fantasize(torch.rand(0, 2, **tkwargs), sampler)
        self.assertEqual(fm.batch_shape, torch.Size([3]))

    def test_acquisition_functions(self) -> None:
        tkwargs = {"dtype": torch.double, "device": self.device}
        gen = torch.Generator().manual_seed(0)
        task = _sinusoid_task(gen, n=6, **tkwargs)
        model = MetaLSQ(dim=1, **tkwargs).condition(
            task.train_X, task.train_Y, task.train_dY
        )
        bounds = torch.tensor([[-5.0], [5.0]], **tkwargs)
        best_f = task.train_Y.max()
        for acqf, q in (
            (LogExpectedImprovement(model, best_f=best_f), 1),
            (qLogExpectedImprovement(model, best_f=best_f), 2),
            (qKnowledgeGradient(model, num_fantasies=4), 1),
        ):
            cand, _ = optimize_acqf(
                acqf, bounds=bounds, q=q, num_restarts=2, raw_samples=16
            )
            self.assertEqual(cand.shape, torch.Size([q, 1]))
            self.assertTrue(((cand >= -5) & (cand <= 5)).all())
