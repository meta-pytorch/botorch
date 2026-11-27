#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.testing import BotorchTestCase
from botorch_community.models.cake import (
    algebraic_crossover,
    algebraic_mutation,
    AlgebraicLLMClient,
    BASE_KERNEL_NAMES,
    CAKE,
    count_base_kernels,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    fit_kernel_gp,
    KernelExpressionError,
    OllamaLLMClient,
    OpenAILLMClient,
    parse_kernel_expression,
    parse_llm_kernel_response,
    ScriptedLLMClient,
    tokenize_kernel_expression,
)
from gpytorch.kernels import (
    AdditiveKernel,
    MaternKernel,
    PeriodicKernel,
    ProductKernel,
    RBFKernel,
    ScaleKernel,
)


def _tiny_data(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [[0.1], [0.3], [0.5], [0.7], [0.9]], device=device, dtype=dtype
    )
    train_Y = torch.sin(6.0 * train_X)
    return train_X, train_Y


class TestKernelParsing(BotorchTestCase):
    def test_tokenize_and_count(self) -> None:
        tokens = tokenize_kernel_expression("se + (per * lin)")
        self.assertEqual(tokens, ["SE", "+", "(", "PER", "*", "LIN", ")"])
        self.assertEqual(count_base_kernels("SE + (PER * LIN)"), 3)

    def test_invalid_expression(self) -> None:
        with self.assertRaises(KernelExpressionError):
            tokenize_kernel_expression("")
        with self.assertRaises(KernelExpressionError):
            parse_kernel_expression("SE / PER", ard_num_dims=1)
        with self.assertRaises(KernelExpressionError):
            parse_kernel_expression("SE +", ard_num_dims=1)
        with self.assertRaises(KernelExpressionError):
            parse_kernel_expression("(SE + PER", ard_num_dims=1)

    def test_parse_base_and_composition(self) -> None:
        se = parse_kernel_expression("SE", ard_num_dims=2)
        self.assertIsInstance(se, ScaleKernel)
        self.assertIsInstance(se.base_kernel, RBFKernel)
        self.assertEqual(se.base_kernel.ard_num_dims, 2)

        additive = parse_kernel_expression("SE + M5", ard_num_dims=1)
        self.assertIsInstance(additive.base_kernel, AdditiveKernel)

        product = parse_kernel_expression("PER * LIN", ard_num_dims=1)
        self.assertIsInstance(product.base_kernel, ProductKernel)
        self.assertIsInstance(product.base_kernel.kernels[0], PeriodicKernel)

        nested = parse_kernel_expression("(SE + PER) * M3", ard_num_dims=1)
        self.assertIsInstance(nested.base_kernel, ProductKernel)
        matern_terms = [k for k in nested.modules() if isinstance(k, MaternKernel)]
        self.assertTrue(any(k.nu == 1.5 for k in matern_terms))

    def test_repeated_kernels_do_not_share_modules(self) -> None:
        kernel = parse_kernel_expression("SE + SE", ard_num_dims=1)
        left, right = kernel.base_kernel.kernels
        self.assertIsNot(left, right)

    def test_algebraic_operators(self) -> None:
        crossed = algebraic_crossover("SE", "PER", operator="*")
        self.assertEqual(crossed, "(SE) * (PER)")
        mutated = algebraic_mutation("SE + PER", replacement="M5")
        self.assertEqual(mutated, "M5 + PER")
        with self.assertRaises(KernelExpressionError):
            algebraic_crossover("SE", "PER", operator="-")


class TestLLMHelpers(BotorchTestCase):
    def test_parse_llm_kernel_response(self) -> None:
        response = "Analysis first.\nKernel: SE + PER\nAnalysis: additive structure."
        self.assertEqual(parse_llm_kernel_response(response), "SE + PER")
        with self.assertRaises(KernelExpressionError):
            parse_llm_kernel_response("no kernel here")

    def test_scripted_client(self) -> None:
        client = ScriptedLLMClient(["Kernel: SE * LIN"])
        self.assertIn("SE * LIN", client.complete("sys", "user"))
        with self.assertRaises(RuntimeError):
            client.complete("sys", "user")

    def test_algebraic_client_crossover_and_mutation(self) -> None:
        client = AlgebraicLLMClient(seed=0)
        crossover = client.complete(
            "sys",
            "You are given two parent kernels and their BIC fitness scores "
            "(lower is better):\nSE (1.0), PER (2.0)",
        )
        self.assertRegex(crossover, r"Kernel: \(SE\) [+*] \(PER\)")
        mutation = client.complete(
            "sys",
            "You are given a kernel and its BIC fitness score (lower is better):\n"
            "SE (1.0)",
        )
        self.assertRegex(mutation, r"Kernel: (SE|PER|LIN|RQ|M3|M5)")

        parenthesized = client.complete(
            "sys",
            "You are given two parent kernels and their BIC fitness scores "
            "(lower is better):\n(SE) + (PER) (1.0), (M5) * (LIN) (2.0)",
        )
        self.assertRegex(
            parenthesized,
            r"Kernel: \(\(SE\) \+ \(PER\)\) [+*] \(\(M5\) \* \(LIN\)\)",
        )

    def test_openai_client_requires_package_and_key(self) -> None:
        with patch("os.getenv", return_value=None):
            try:
                OpenAILLMClient(api_key=None)
            except ImportError as err:
                self.assertIn("openai", str(err))
            except ValueError as err:
                self.assertIn("OPENAI_API_KEY", str(err))

    def test_openai_and_ollama_client_defaults(self) -> None:
        fake_openai = MagicMock()
        fake_mod = MagicMock()
        fake_mod.OpenAI = fake_openai
        with patch.dict("sys.modules", {"openai": fake_mod}):
            openai_client = OpenAILLMClient(api_key="sk-test")
            self.assertEqual(openai_client.model, DEFAULT_OPENAI_MODEL)
            self.assertIsNone(openai_client.temperature)
            fake_openai.assert_called_with(api_key="sk-test")

            ollama_client = OllamaLLMClient()
            self.assertEqual(ollama_client.model, DEFAULT_OLLAMA_MODEL)
            self.assertEqual(ollama_client.base_url, DEFAULT_OLLAMA_BASE_URL)
            fake_openai.assert_called_with(
                api_key="ollama", base_url=DEFAULT_OLLAMA_BASE_URL
            )


class TestFitKernelGP(BotorchTestCase):
    def test_fit_and_bic(self) -> None:
        for dtype in (torch.float, torch.double):
            train_X, train_Y = _tiny_data(self.device, dtype)
            model, bic = fit_kernel_gp(train_X, train_Y, kernel_expression="SE")
            self.assertIsInstance(model, SingleTaskGP)
            self.assertTrue(torch.isfinite(torch.tensor(bic)))
            posterior = model.posterior(train_X)
            self.assertEqual(posterior.mean.shape[-2:], torch.Size([5, 1]))

    def test_invalid_shapes(self) -> None:
        train_X = torch.rand(3, 2, 2, device=self.device)
        train_Y = torch.rand(3, 1, device=self.device)
        with self.assertRaises(UnsupportedError):
            fit_kernel_gp(train_X, train_Y, kernel_expression="SE")


class TestCAKE(BotorchTestCase):
    def _make_cake(self, **kwargs: object) -> CAKE:
        defaults: dict[str, object] = {
            "llm_client": ScriptedLLMClient(["Kernel: SE + PER", "Kernel: M5"]),
            "num_crossover": 1,
            "mutation_prob": 1.0,
            "population_size": 3,
            "base_kernels": ("SE", "M5"),
            "num_restarts": 2,
            "raw_samples": 8,
            "seed": 0,
        }
        defaults.update(kwargs)
        return CAKE(**defaults)  # pyre-ignore[6]

    def test_invalid_init(self) -> None:
        with self.assertRaises(ValueError):
            CAKE(population_size=0)
        with self.assertRaises(ValueError):
            CAKE(mutation_prob=1.5)
        with self.assertRaises(ValueError):
            CAKE(base_kernels=("NOT_A_KERNEL",))

    def test_fit_evolves_population(self) -> None:
        train_X, train_Y = _tiny_data(self.device, torch.double)
        cake = self._make_cake()
        best = cake.fit(train_X, train_Y)
        self.assertIn(best, cake.population)
        self.assertLessEqual(len(cake.population), cake.population_size)
        self.assertIsInstance(cake.best_model, SingleTaskGP)
        self.assertTrue(torch.isfinite(torch.tensor(cake.best_bic)))
        assert cake.population_prob is not None
        self.assertAlmostEqual(float(cake.population_prob.sum()), 1.0, places=5)
        self.assertTrue((cake.population_prob >= 0).all())

    def test_algebraic_fallback_without_llm(self) -> None:
        train_X, train_Y = _tiny_data(self.device, torch.double)
        cake = self._make_cake(llm_client=None, mutation_prob=1.0)
        best = cake.fit(train_X, train_Y)
        self.assertIsInstance(best, str)
        self.assertGreaterEqual(len(cake.population), 1)

    def test_invalid_llm_response_falls_back(self) -> None:
        train_X, train_Y = _tiny_data(self.device, torch.double)
        cake = self._make_cake(
            llm_client=ScriptedLLMClient(["not a kernel", "still not"]),
            mutation_prob=1.0,
        )
        cake.fit(train_X, train_Y)
        self.assertGreaterEqual(len(cake.population), 1)

    def test_baker_query_shape(self) -> None:
        train_X, train_Y = _tiny_data(self.device, torch.double)
        cake = self._make_cake()
        cake.fit(train_X, train_Y)
        bounds = torch.tensor([[0.0], [1.0]], device=self.device, dtype=torch.double)
        query = cake.get_next_query(bounds)
        self.assertEqual(query.X.shape, torch.Size([1, 1]))
        self.assertTrue((query.X >= 0).all() and (query.X <= 1).all())
        self.assertIn(query.kernel_expression, cake.population)
        self.assertTrue(torch.isfinite(query.baker_score))

    def test_get_next_query_requires_fit(self) -> None:
        cake = self._make_cake()
        bounds = torch.tensor([[0.0], [1.0]], device=self.device)
        with self.assertRaises(RuntimeError):
            cake.get_next_query(bounds)
        with self.assertRaisesRegex(RuntimeError, "fit"):
            _ = cake.best_kernel

    def test_best_kernel_is_lowest_bic(self) -> None:
        train_X, train_Y = _tiny_data(self.device, torch.double)
        cake = self._make_cake(llm_client=None, mutation_prob=0.0, num_crossover=0)
        cake.fit(train_X, train_Y)
        expected = min(cake.population, key=lambda k: cake.population[k].bic)
        self.assertEqual(cake.best_kernel, expected)
        self.assertEqual(set(cake.population), set(cake.base_kernels))
        self.assertTrue(all(k in BASE_KERNEL_NAMES for k in cake.population))
