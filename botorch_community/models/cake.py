#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Context-Aware Kernel Evolution (CAKE) as introduced in [Suwandi2025cake]_.

CAKE uses an LLM as the crossover and mutation operators of an evolutionary
search over compositional Gaussian-process kernels. At every BO iteration the
current observations are written into a system prompt, candidate kernels are
scored with the Bayesian information criterion (BIC), and the next query is
chosen with BIC-Acquisition Kernel Ranking (BAKER). BAKER multiplies a
fitness weight proportional to ``exp(-BIC)`` by the expected improvement of
each kernel's proposed query, then selects the highest-scoring pair.

This module is a BoTorch-native reimplementation of the public research code
from [Suwandi2025cake]_. The LLM is optional: tests and smoke runs can inject a
scripted or algebraic client. Cloud and local chat backends share the OpenAI
Chat Completions protocol (OpenAI, Ollama, vLLM, LM Studio) and import
``openai`` lazily, so it is not a hard dependency of BoTorch.

References:

.. [Suwandi2025cake]
    R. C. Suwandi, F. Yin, J. Wang, R. Li, T.-H. Chang, and S. Theodoridis.
    Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with
    LLMs. Advances in Neural Information Processing Systems (NeurIPS), 2025.
    https://proceedings.neurips.cc/paper_files/paper/2025/file/c03a2610bca2712b984b331fd4f7bb6f-Paper-Conference.pdf

Contributor: richardcsuwandi
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.exceptions.errors import UnsupportedError
from botorch.fit import fit_gpytorch_mll
from botorch.logging import logger
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from gpytorch.kernels import (
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
)
from gpytorch.kernels.kernel import Kernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor

BASE_KERNEL_NAMES: tuple[str, ...] = ("SE", "PER", "LIN", "RQ", "M3", "M5")
KERNEL_OPERATORS: tuple[str, ...] = ("+", "*")
_TOKEN_PATTERN = re.compile(r"SE|PER|LIN|RQ|M[135]|[()+*]")
_KERNEL_LINE_PATTERN = re.compile(r"Kernel:\s*(.+)", flags=re.IGNORECASE)
_BASE_TOKEN_PATTERN = re.compile(r"SE|PER|LIN|RQ|M[135]")
# Parenthesized numeric BIC after a kernel, e.g. ``(SE + PER) (12.3)``.
_KERNEL_WITH_BIC_PATTERN = re.compile(
    r"(.+?)\s+\(([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\)"
)

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert in machine learning, specializing in Gaussian processes. \
Here are the observations we have collected so far:
{observations}

Please analyze these observations to identify patterns in the data that can \
be captured by a kernel function.
You can use any of the following base kernels: {base_kernels}, and combine \
these kernels using the following operators: {operators}.
Your goal is to construct a kernel expression that best explains the observed \
data. The kernel will be evaluated using a fitness score derived from the \
Bayesian information criterion (BIC), where lower BIC is better.
Always respond in exactly this format:
Kernel: <expression using only {base_kernels} and operators {operators}>
Analysis: <brief reasoning>
"""

CROSSOVER_PROMPT_TEMPLATE = """\
You are given two parent kernels and their BIC fitness scores \
(lower is better):
{parent_kernel1} ({fitness1}), {parent_kernel2} ({fitness2})

Please propose a new kernel that has a potentially better (lower) BIC. \
You may combine the parent kernels using any of the operators from: \
{operators}. Briefly explain your reasoning behind the proposed kernel.
Respond in exactly this format:
Kernel: <expression>
Analysis: <brief reasoning>
"""

MUTATION_PROMPT_TEMPLATE = """\
You are given a kernel and its BIC fitness score (lower is better):
{kernel} ({fitness})

Please propose a new kernel that has a potentially better (lower) BIC. \
You may replace a base kernel in the current expression with another base \
kernel from the set: {base_kernels}. Briefly explain your reasoning.
Respond in exactly this format:
Kernel: <expression>
Analysis: <brief reasoning>
"""


class KernelExpressionError(ValueError):
    """Raised when a compositional kernel expression cannot be parsed."""


class LLMClient(Protocol):
    """Minimal chat-completion interface used by CAKE."""

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Return the assistant message text.

        Args:
            system_prompt: System prompt with observations and kernel grammar.
            user_message: Crossover or mutation instruction.

        Returns:
            Raw assistant text, expected to contain a ``Kernel:`` line.
        """


def _make_base_kernel(name: str, ard_num_dims: int) -> Kernel:
    if name == "SE":
        return RBFKernel(ard_num_dims=ard_num_dims)
    if name == "PER":
        return PeriodicKernel(ard_num_dims=ard_num_dims)
    if name == "LIN":
        return LinearKernel(ard_num_dims=ard_num_dims)
    if name == "RQ":
        return RQKernel(ard_num_dims=ard_num_dims)
    if name == "M1":
        return MaternKernel(nu=0.5, ard_num_dims=ard_num_dims)
    if name == "M3":
        return MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
    if name == "M5":
        return MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
    raise KernelExpressionError(f"Unknown base kernel '{name}'.")


def tokenize_kernel_expression(expression: str) -> list[str]:
    """Split a kernel expression into base kernels, operators, and parens.

    Args:
        expression: Kernel string such as ``SE + (PER * LIN)``.

    Returns:
        Token list in the original order.

    Raises:
        KernelExpressionError: If the expression contains unknown symbols.
    """
    cleaned = re.sub(r"\s+", "", expression).upper()
    if not cleaned:
        raise KernelExpressionError("Kernel expression is empty.")
    tokens = _TOKEN_PATTERN.findall(cleaned)
    if "".join(tokens) != cleaned:
        raise KernelExpressionError(
            f"Could not tokenize kernel expression '{expression}'."
        )
    return tokens


def count_base_kernels(expression: str) -> int:
    """Count base-kernel tokens in a compositional expression.

    Args:
        expression: Kernel string such as ``SE + PER``.

    Returns:
        Number of base-kernel tokens.
    """
    return len(_BASE_TOKEN_PATTERN.findall(re.sub(r"\s+", "", expression).upper()))


def parse_kernel_expression(expression: str, ard_num_dims: int) -> ScaleKernel:
    """Parse a compositional kernel string into a GPyTorch ``ScaleKernel``.

    Addition and multiplication are left-associative and have equal
    precedence, matching the original CAKE implementation. Each occurrence of
    a base kernel is a fresh GPyTorch module so repeated terms do not share
    hyperparameters.

    Args:
        expression: Kernel string using ``SE``, ``PER``, ``LIN``, ``RQ``,
            ``M3``, ``M5`` and operators ``+`` / ``*``.
        ard_num_dims: Input dimension passed to each ARD base kernel.

    Returns:
        A ``ScaleKernel`` wrapping the parsed composition.

    Raises:
        KernelExpressionError: If the expression is invalid.
    """
    if ard_num_dims < 1:
        raise KernelExpressionError(
            f"ard_num_dims must be >= 1, but got {ard_num_dims}."
        )
    tokens = tokenize_kernel_expression(expression)
    kernel, index = _parse_expr(tokens, 0, ard_num_dims)
    if index != len(tokens):
        raise KernelExpressionError(
            f"Unexpected token '{tokens[index]}' in '{expression}'."
        )
    return ScaleKernel(kernel)


def _parse_expr(tokens: list[str], index: int, ard_num_dims: int) -> tuple[Kernel, int]:
    left, index = _parse_term(tokens, index, ard_num_dims)
    while index < len(tokens) and tokens[index] in KERNEL_OPERATORS:
        operator = tokens[index]
        right, index = _parse_term(tokens, index + 1, ard_num_dims)
        left = left + right if operator == "+" else left * right
    return left, index


def _parse_term(tokens: list[str], index: int, ard_num_dims: int) -> tuple[Kernel, int]:
    if index >= len(tokens):
        raise KernelExpressionError("Unexpected end of kernel expression.")
    token = tokens[index]
    if token == "(":
        inner, index = _parse_expr(tokens, index + 1, ard_num_dims)
        if index >= len(tokens) or tokens[index] != ")":
            raise KernelExpressionError("Unbalanced parentheses in kernel expression.")
        return inner, index + 1
    if token in {")", "+", "*"}:
        raise KernelExpressionError(f"Unexpected token '{token}'.")
    return _make_base_kernel(token, ard_num_dims), index + 1


def parse_llm_kernel_response(response: str) -> str:
    """Extract a kernel expression from an LLM response.

    Args:
        response: Raw assistant text.

    Returns:
        The parsed kernel expression, stripped of surrounding whitespace.

    Raises:
        KernelExpressionError: If no ``Kernel:`` line is found.
    """
    match = _KERNEL_LINE_PATTERN.search(response)
    if match is None:
        raise KernelExpressionError("LLM response did not contain a 'Kernel:' line.")
    expression = match.group(1).strip().strip("`").strip()
    if not expression:
        raise KernelExpressionError("LLM returned an empty kernel expression.")
    return expression


def compute_bic(model: SingleTaskGP) -> float:
    """Compute BIC of a fitted exact GP on its transformed training targets.

    Args:
        model: A fitted ``SingleTaskGP``.

    Returns:
        BIC value. Lower is better.
    """
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    output = model(*model.train_inputs)
    mll_mean = mll(output, model.train_targets)
    num_data = model.train_targets.numel()
    log_likelihood = float(mll_mean.item()) * num_data
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bic = -2.0 * log_likelihood + num_params * math.log(num_data)
    model.eval()
    return bic


def fit_kernel_gp(
    train_X: Tensor,
    train_Y: Tensor,
    kernel_expression: str,
    train_Yvar: Tensor | None = None,
) -> tuple[SingleTaskGP, float]:
    """Fit a ``SingleTaskGP`` with a compositional CAKE kernel and return BIC.

    Args:
        train_X: An ``n x d`` tensor of training inputs.
        train_Y: An ``n x 1`` or ``n`` tensor of training observations.
        kernel_expression: Compositional kernel string.
        train_Yvar: Optional ``n x 1`` observation noise.

    Returns:
        The fitted model and its BIC.
    """
    train_X, train_Y, train_Yvar = _validate_train_data(
        train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar
    )
    covar_module = parse_kernel_expression(
        expression=kernel_expression, ard_num_dims=train_X.shape[-1]
    )
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        covar_module=covar_module,
        outcome_transform=Standardize(m=1),
        input_transform=Normalize(d=train_X.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model, compute_bic(model)


def _validate_train_data(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    if train_X.ndim != 2:
        raise UnsupportedError(
            f"CAKE currently supports 2-D train_X of shape n x d, got {train_X.shape}."
        )
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)
    if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
        raise UnsupportedError(
            "CAKE currently supports single-output train_Y of shape n or n x 1, "
            f"got {tuple(train_Y.shape)}."
        )
    if train_X.shape[0] != train_Y.shape[0]:
        raise UnsupportedError(
            "train_X and train_Y must have the same n, got "
            f"{train_X.shape[0]} and {train_Y.shape[0]}."
        )
    if train_Yvar is not None:
        if train_Yvar.ndim == 1:
            train_Yvar = train_Yvar.unsqueeze(-1)
        if train_Yvar.shape != train_Y.shape:
            raise UnsupportedError(
                "train_Yvar must match train_Y shape, got "
                f"{tuple(train_Yvar.shape)} vs {tuple(train_Y.shape)}."
            )
    return train_X, train_Y, train_Yvar


def algebraic_crossover(parent_a: str, parent_b: str, operator: str = "+") -> str:
    """Combine two kernel expressions with addition or multiplication.

    Args:
        parent_a: First parent kernel expression.
        parent_b: Second parent kernel expression.
        operator: Either ``+`` or ``*``.

    Returns:
        A parenthesized compositional expression.
    """
    if operator not in KERNEL_OPERATORS:
        raise KernelExpressionError(
            f"operator must be one of {KERNEL_OPERATORS}, got '{operator}'."
        )
    return f"({parent_a}) {operator} ({parent_b})"


def algebraic_mutation(expression: str, replacement: str) -> str:
    """Replace the first base kernel in ``expression`` with ``replacement``.

    Args:
        expression: Kernel expression to mutate.
        replacement: Replacement base-kernel name.

    Returns:
        Mutated kernel expression.
    """
    if replacement not in BASE_KERNEL_NAMES and replacement != "M1":
        raise KernelExpressionError(f"Unknown replacement kernel '{replacement}'.")
    mutated, count = _BASE_TOKEN_PATTERN.subn(replacement, expression, count=1)
    if count == 0:
        raise KernelExpressionError(
            f"No base kernel found to mutate in '{expression}'."
        )
    return mutated


DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"


class OpenAICompatibleLLMClient:
    """Chat Completions client for OpenAI-compatible HTTP servers.

    Used by OpenAI, Ollama, vLLM, and LM Studio. The ``openai`` package is
    imported lazily. Install it with ``pip install openai``.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        require_api_key: bool = True,
    ) -> None:
        r"""Initialize an OpenAI-compatible chat client.

        Args:
            model: Chat model name on the target server.
            api_key: API key. Defaults to ``OPENAI_API_KEY`` when
                ``require_api_key`` is True.
            base_url: Optional Chat Completions base URL. Omit for OpenAI.
            temperature: Sampling temperature. ``None`` omits the field,
                which is required for some GPT-5 chat models.
            top_p: Nucleus sampling parameter. ``None`` omits the field.
            require_api_key: If True, raise when no API key is available.
        """
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError(
                "CAKE chat backends require the `openai` package. Install "
                "it with `pip install openai`."
            ) from err
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if require_api_key and not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Pass api_key or export the "
                "environment variable."
            )
        client_kwargs: dict[str, Any] = {"api_key": resolved_key or "ollama"}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self._client = OpenAI(**client_kwargs)

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Call the chat completions API.

        Args:
            system_prompt: System prompt with observations and kernel grammar.
            user_message: Crossover or mutation instruction.

        Returns:
            Assistant message content.
        """
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        if self.temperature is not None:
            create_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            create_kwargs["top_p"] = self.top_p
        response = self._client.chat.completions.create(**create_kwargs)
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Chat completion returned empty content.")
        return content


class OpenAILLMClient(OpenAICompatibleLLMClient):
    """OpenAI API client. Defaults to ``gpt-5-mini``.

    The paper experiments in [Suwandi2025cake]_ used ``gpt-4o-mini``. Pass
    that name explicitly to reproduce the original setup.
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> None:
        r"""Initialize the OpenAI chat client.

        Args:
            model: OpenAI chat model. Defaults to ``gpt-5-mini``.
            api_key: API key. Defaults to the ``OPENAI_API_KEY`` environment
                variable.
            base_url: Optional custom API base URL.
            temperature: Sampling temperature. Omitted by default because
                GPT-5 chat models often reject non-default values.
            top_p: Nucleus sampling parameter. Omitted by default.
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            require_api_key=True,
        )


class OllamaLLMClient(OpenAICompatibleLLMClient):
    """Local Ollama client via the OpenAI-compatible ``/v1`` endpoint.

    Requires a running ``ollama serve`` and a pulled model, for example
    ``ollama pull llama3.2``. No extra Python package beyond ``openai``.
    The dummy API key is required by the OpenAI SDK and ignored by Ollama.
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        api_key: str = "ollama",
        temperature: float | None = 0.7,
        top_p: float | None = 1.0,
    ) -> None:
        r"""Initialize a local Ollama chat client.

        Args:
            model: Ollama model tag, such as ``llama3.2`` or ``qwen2.5``.
            base_url: OpenAI-compatible Ollama endpoint.
            api_key: Dummy key required by the OpenAI SDK. Ignored locally.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            require_api_key=False,
        )


class ScriptedLLMClient:
    """Deterministic LLM stand-in for tests and notebooks."""

    def __init__(self, responses: Sequence[str]) -> None:
        r"""Initialize from a finite list of canned responses.

        Args:
            responses: Assistant strings, each containing a ``Kernel:`` line.
        """
        if not responses:
            raise ValueError("responses must be a non-empty sequence.")
        self._responses = list(responses)
        self.num_calls = 0

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Return the next scripted response.

        Args:
            system_prompt: Unused; accepted for API compatibility.
            user_message: Unused; accepted for API compatibility.

        Returns:
            The next canned assistant string.

        Raises:
            RuntimeError: If no scripted responses remain.
        """
        del system_prompt, user_message
        if self.num_calls >= len(self._responses):
            raise RuntimeError("ScriptedLLMClient has no remaining responses.")
        response = self._responses[self.num_calls]
        self.num_calls += 1
        return response


class AlgebraicLLMClient:
    """LLM-free genetic operators that emit valid CAKE kernel strings."""

    def __init__(
        self,
        operators: Sequence[str] = KERNEL_OPERATORS,
        base_kernels: Sequence[str] = BASE_KERNEL_NAMES,
        seed: int = 0,
    ) -> None:
        r"""Initialize algebraic crossover and mutation.

        Args:
            operators: Allowed binary operators.
            base_kernels: Allowed replacement kernels for mutation.
            seed: RNG seed used to pick operators and replacements.
        """
        self.operators = tuple(operators)
        self.base_kernels = tuple(base_kernels)
        self._generator = torch.Generator().manual_seed(seed)

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Propose a crossover or mutation from the user prompt text.

        Args:
            system_prompt: Unused; accepted for API compatibility.
            user_message: Crossover or mutation prompt containing kernels.

        Returns:
            A response with a ``Kernel:`` line.
        """
        del system_prompt
        expressions = _extract_kernel_names_from_prompt(user_message)
        if "two parent kernels" in user_message.lower():
            if len(expressions) < 2:
                raise KernelExpressionError(
                    "Crossover prompt did not contain two parent kernels."
                )
            operator = self.operators[int(self._randint(len(self.operators)))]
            kernel = algebraic_crossover(expressions[0], expressions[1], operator)
        else:
            if not expressions:
                raise KernelExpressionError(
                    "Mutation prompt did not contain a kernel expression."
                )
            replacement = self.base_kernels[int(self._randint(len(self.base_kernels)))]
            kernel = algebraic_mutation(expressions[0], replacement)
        return f"Kernel: {kernel}\nAnalysis: algebraic operator"

    def _randint(self, n: int) -> int:
        return int(torch.randint(n, (1,), generator=self._generator).item())


def _extract_kernel_names_from_prompt(user_message: str) -> list[str]:
    # Crossover/mutation prompts append a parenthesized BIC after each expression.
    expressions: list[str] = []
    for raw_expression, _bic in _KERNEL_WITH_BIC_PATTERN.findall(user_message):
        expression = " ".join(raw_expression.replace(",", " ").split())
        if not expression:
            continue
        try:
            tokenize_kernel_expression(expression)
        except KernelExpressionError:
            continue
        expressions.append(expression)
    return expressions


@dataclass
class KernelRecord:
    """A fitted kernel in the CAKE population."""

    expression: str
    model: SingleTaskGP
    bic: float


@dataclass
class CAKEQuery:
    """Next evaluation proposed by BAKER."""

    X: Tensor
    baker_score: Tensor
    kernel_expression: str
    model: SingleTaskGP


class CAKE:
    r"""LLM-guided compositional kernel search with BAKER query selection.

    Default hyperparameters are a lighter setting than the paper. The
    experiments in [Suwandi2025cake]_ used ``num_crossover=5``,
    ``population_size=10``, and ``mutation_prob=0.7``.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        num_crossover: int = 1,
        mutation_prob: float = 0.7,
        population_size: int = 6,
        base_kernels: Sequence[str] = BASE_KERNEL_NAMES,
        operators: Sequence[str] = KERNEL_OPERATORS,
        max_base_kernels: int = 5,
        max_prompt_observations: int | None = 64,
        num_restarts: int = 10,
        raw_samples: int = 32,
        maximize: bool = True,
        seed: int | None = None,
    ) -> None:
        r"""Initialize CAKE.

        Args:
            llm_client: Chat client used for crossover and mutation. If
                omitted, algebraic operators are used. Pass
                :class:`OpenAILLMClient` for the OpenAI API (default
                ``gpt-5-mini``) or :class:`OllamaLLMClient` for a local
                Ollama server. The paper used ``gpt-4o-mini``.
            num_crossover: Number of crossover proposals per generation.
            mutation_prob: Probability of mutating the current best kernel.
            population_size: Number of kernels retained after selection.
            base_kernels: Initial population and mutation vocabulary.
            operators: Allowed binary kernel operators.
            max_base_kernels: Maximum number of base kernels in a proposal.
            max_prompt_observations: If set, include at most this many most
                recent observations in the LLM prompt.
            num_restarts: Restarts used by :func:`optimize_acqf` in BAKER.
            raw_samples: Raw samples used by :func:`optimize_acqf` in BAKER.
            maximize: If True, treat the problem as maximization.
            seed: Optional RNG seed for parent sampling and algebraic
                fallbacks.
        """
        if num_crossover < 0:
            raise ValueError(f"num_crossover must be >= 0, got {num_crossover}.")
        if not 0.0 <= mutation_prob <= 1.0:
            raise ValueError(f"mutation_prob must be in [0, 1], got {mutation_prob}.")
        if population_size < 1:
            raise ValueError(f"population_size must be >= 1, got {population_size}.")
        if max_base_kernels < 1:
            raise ValueError(f"max_base_kernels must be >= 1, got {max_base_kernels}.")
        if num_restarts < 1:
            raise ValueError(f"num_restarts must be >= 1, got {num_restarts}.")
        if raw_samples < 1:
            raise ValueError(f"raw_samples must be >= 1, got {raw_samples}.")
        unknown = set(base_kernels) - set(BASE_KERNEL_NAMES + ("M1",))
        if unknown:
            raise ValueError(f"Unknown base kernels: {sorted(unknown)}.")
        if any(op not in KERNEL_OPERATORS for op in operators):
            raise ValueError(f"operators must be a subset of {KERNEL_OPERATORS}.")
        self.llm_client = llm_client
        self.num_crossover = num_crossover
        self.mutation_prob = mutation_prob
        self.population_size = population_size
        self.base_kernels = tuple(base_kernels)
        self.operators = tuple(operators)
        self.max_base_kernels = max_base_kernels
        self.max_prompt_observations = max_prompt_observations
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.maximize = maximize
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)
        self.population: dict[str, KernelRecord] = {}
        self.population_prob: Tensor | None = None
        self.train_X: Tensor | None = None
        self.train_Y: Tensor | None = None
        self.train_Yvar: Tensor | None = None
        self.system_prompt: str | None = None

    @property
    def best_kernel(self) -> str:
        """Kernel expression with the lowest BIC in the current population."""
        if not self.population:
            raise RuntimeError("Call fit() before accessing best_kernel.")
        return min(self.population, key=lambda expr: self.population[expr].bic)

    @property
    def best_model(self) -> SingleTaskGP:
        """Fitted GP corresponding to :attr:`best_kernel`."""
        return self.population[self.best_kernel].model

    @property
    def best_bic(self) -> float:
        """BIC of :attr:`best_kernel`."""
        return self.population[self.best_kernel].bic

    def fit(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
    ) -> str:
        """Run one CAKE generation on the current data.

        Args:
            train_X: An ``n x d`` tensor of observed inputs.
            train_Y: An ``n x 1`` or ``n`` tensor of observed values.
            train_Yvar: Optional observation noise.

        Returns:
            The best kernel expression after selection.
        """
        self.train_X, self.train_Y, self.train_Yvar = _validate_train_data(
            train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        self.system_prompt = self._build_system_prompt()
        self._initialize_population()
        self._propose_kernels()
        self._select_survivors()
        return self.best_kernel

    def get_next_query(self, bounds: Tensor) -> CAKEQuery:
        """Select the next query with BAKER.

        LogEI is optimized for each surviving kernel (numerically stable).
        Ranking uses ``log(weight) + LogEI``, which is equivalent to the
        paper's ``weight * EI`` comparison.

        Args:
            bounds: A ``2 x d`` tensor of lower and upper bounds.

        Returns:
            The BAKER-selected query, score, kernel, and fitted model.
        """
        if not self.population or self.population_prob is None:
            raise RuntimeError("Call fit() before get_next_query().")
        if self.train_X is None or self.train_Y is None:
            raise RuntimeError("CAKE has no training data.")
        if bounds.shape != torch.Size([2, self.train_X.shape[-1]]):
            raise ValueError(
                "bounds must have shape 2 x d, got "
                f"{tuple(bounds.shape)} for d={self.train_X.shape[-1]}."
            )
        bounds = bounds.to(device=self.train_X.device, dtype=self.train_X.dtype)
        best_f = self.train_Y.amax() if self.maximize else self.train_Y.amin()
        best_query: CAKEQuery | None = None
        best_score = torch.tensor(
            float("-inf"), device=self.train_X.device, dtype=self.train_X.dtype
        )
        expressions = list(self.population.keys())
        min_prob = torch.finfo(self.population_prob.dtype).tiny
        log_weights = self.population_prob.clamp(min=min_prob).log()
        for i, expression in enumerate(expressions):
            record = self.population[expression]
            acqf = LogExpectedImprovement(
                model=record.model, best_f=best_f, maximize=self.maximize
            )
            try:
                candidate, log_ei = optimize_acqf(
                    acq_function=acqf,
                    bounds=bounds,
                    q=1,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                    retry_on_optimization_warning=True,
                )
            except Exception as err:
                logger.debug("Skipping kernel %s in BAKER: %s", expression, err)
                continue
            if log_ei is None:
                continue
            score = log_weights[i].to(log_ei) + log_ei
            if torch.isfinite(score) and score > best_score:
                best_score = score
                best_query = CAKEQuery(
                    X=candidate,
                    baker_score=score.detach(),
                    kernel_expression=expression,
                    model=record.model,
                )
        if best_query is None:
            raise RuntimeError("BAKER failed to optimize any kernel in the population.")
        return best_query

    def _build_system_prompt(self) -> str:
        assert self.train_X is not None and self.train_Y is not None
        xs = self.train_X
        ys = self.train_Y
        if (
            self.max_prompt_observations is not None
            and xs.shape[0] > self.max_prompt_observations
        ):
            xs = xs[-self.max_prompt_observations :]
            ys = ys[-self.max_prompt_observations :]
        lines = [
            f"x = {x.tolist()}, y = {y.tolist()}"
            for x, y in zip(xs, ys.squeeze(-1), strict=True)
        ]
        return SYSTEM_PROMPT_TEMPLATE.format(
            observations="\n".join(lines),
            base_kernels=", ".join(self.base_kernels),
            operators=", ".join(self.operators),
        )

    def _initialize_population(self) -> None:
        previous = dict(self.population)
        self.population = {}
        for expression in self.base_kernels:
            self._try_add_kernel(expression)
        # Keep previously evolved kernels that still fit the new data.
        for expression in previous:
            if expression not in self.population:
                self._try_add_kernel(expression)
        if not self.population:
            raise RuntimeError("Failed to fit any kernel in the CAKE population.")
        self._refresh_probabilities()

    def _try_add_kernel(self, expression: str) -> bool:
        assert self.train_X is not None and self.train_Y is not None
        if expression in self.population:
            return False
        if count_base_kernels(expression) > self.max_base_kernels:
            return False
        try:
            parse_kernel_expression(expression, ard_num_dims=self.train_X.shape[-1])
            model, bic = fit_kernel_gp(
                train_X=self.train_X,
                train_Y=self.train_Y,
                kernel_expression=expression,
                train_Yvar=self.train_Yvar,
            )
        except Exception as err:
            logger.debug("Discarding kernel %s: %s", expression, err)
            return False
        if not math.isfinite(bic):
            logger.debug("Discarding kernel %s: non-finite BIC %s.", expression, bic)
            return False
        self.population[expression] = KernelRecord(
            expression=expression, model=model, bic=bic
        )
        return True

    def _propose_kernels(self) -> None:
        self._crossover()
        if self._uniform() < self.mutation_prob:
            self._mutate()
        self._refresh_probabilities()

    def _crossover(self) -> None:
        expressions = list(self.population.keys())
        if len(expressions) < 2:
            return
        assert self.population_prob is not None
        for _ in range(self.num_crossover):
            idx = torch.multinomial(
                self.population_prob,
                num_samples=2,
                replacement=False,
                generator=self._generator,
            )
            parent_a = expressions[int(idx[0].item())]
            parent_b = expressions[int(idx[1].item())]
            operator = self.operators[int(self._randint(len(self.operators)))]
            proposed = self._llm_or_algebraic(
                user_message=CROSSOVER_PROMPT_TEMPLATE.format(
                    parent_kernel1=parent_a,
                    parent_kernel2=parent_b,
                    fitness1=self.population[parent_a].bic,
                    fitness2=self.population[parent_b].bic,
                    operators=", ".join(self.operators),
                ),
                fallback=lambda a=parent_a, b=parent_b, op=operator: (
                    algebraic_crossover(a, b, operator=op)
                ),
            )
            self._try_add_kernel(proposed)

    def _mutate(self) -> None:
        parent = self.best_kernel
        replacement = self.base_kernels[int(self._randint(len(self.base_kernels)))]
        proposed = self._llm_or_algebraic(
            user_message=MUTATION_PROMPT_TEMPLATE.format(
                kernel=parent,
                fitness=self.population[parent].bic,
                base_kernels=", ".join(self.base_kernels),
            ),
            fallback=lambda expr=parent, repl=replacement: algebraic_mutation(
                expr, replacement=repl
            ),
        )
        self._try_add_kernel(proposed)

    def _llm_or_algebraic(self, user_message: str, fallback: Callable[[], str]) -> str:
        if self.llm_client is None or self.system_prompt is None:
            return fallback()
        try:
            response = self.llm_client.complete(self.system_prompt, user_message)
            return parse_llm_kernel_response(response)
        except Exception as err:
            logger.debug("LLM proposal failed (%s); using algebraic fallback.", err)
            return fallback()

    def _select_survivors(self) -> None:
        ranked = sorted(self.population.items(), key=lambda item: item[1].bic)
        self.population = dict(ranked[: self.population_size])
        self._refresh_probabilities()

    def _refresh_probabilities(self) -> None:
        bics = torch.tensor(
            [record.bic for record in self.population.values()],
            dtype=torch.double,
        ).cpu()
        centered = bics - bics.mean()
        if bics.numel() > 1:
            std = bics.std(unbiased=False)
            if torch.isfinite(std) and std > 0:
                centered = centered / std
        self.population_prob = torch.softmax(-centered, dim=0)

    def _uniform(self) -> float:
        return float(torch.rand(1, generator=self._generator).item())

    def _randint(self, n: int) -> int:
        return int(torch.randint(n, (1,), generator=self._generator).item())
