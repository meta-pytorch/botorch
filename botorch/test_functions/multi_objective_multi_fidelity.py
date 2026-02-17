#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Multi-objective multi-fidelity optimization benchmark problems.

References

.. [Irshad2021]
    F. Irshad, S. Karsch, and A. Döpp. Expected hypervolume improvement for
    simultaneous multi-objective and multi-fidelity optimization.
    arXiv preprint arXiv:2112.13901, 2021.
"""

import math

import torch
from botorch.test_functions.base import MultiObjectiveTestProblem
from torch import Tensor


class MOMFBraninCurrin(MultiObjectiveTestProblem):
    r"""Branin-Currin problem for multi-objective-multi-fidelity optimization.

    (2+1)-dimensional function with domain ``[0,1]^3`` where the last dimension
    is the fidelity parameter ``s``.
    Both functions assume minimization. See [Irshad2021]_ for more details.

    Modified Branin function::

        B(x,s) = 21-((
        15*x_2 - b(s) * (15 * x_1 - 5) ** 2 + c(s) * (15 * x_1 - 5) - 6 ) ** 2
        + 10 * (1 - t(s)) * cos(15 * x_1 - 5)+10)/22

    Here ``b``, ``c``, ``r`` and ``t`` are constants and ``s`` is the fidelity
    parameter, where:

        - ``b = 5.1 / (4 * math.pi ** 2) - 0.01(1-s)``
        - ``c = 5 / math.pi - 0.1*(1 - s)``
        - ``r = 6``
        - ``t = 1 / (8 * math.pi) + 0.05*(1-s)``

    Modified Currin function::

        C(x) = 14-((1 - 0.1(1-s)exp(-1 / (2 * x_2))) * (
        2300 * x_1 ** 3 + 1900 * x_1 ** 2 + 2092 * x_1 + 60
        ) / 100 * x_1 ** 3 + 500 * x_1 ** 2 + 4 * x_2 + 20)/15
    """

    dim = 3
    continuous_inds = list(range(3))
    num_objectives = 2
    _bounds = [(0.0, 1.0) for _ in range(dim)]
    _ref_point = [0, 0]
    _max_hv = 0.5235514158034145

    def _branin(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        s = X[..., 2]

        x11 = 15 * x1 - 5
        x22 = 15 * x2
        b = 5.1 / (4 * math.pi**2) - 0.01 * (1 - s)
        c = 5 / math.pi - 0.1 * (1 - s)
        r = 6
        t = 1 / (8 * math.pi) + 0.05 * (1 - s)
        y = (x22 - b * x11**2 + c * x11 - r) ** 2 + 10 * (1 - t) * torch.cos(x11) + 10
        B = 21 - y
        return B / 22

    def _currin(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        s = X[..., 2]
        A = 2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60
        B = 100 * x1**3 + 500 * x1**2 + 4 * x1 + 20
        y = (1 - 0.1 * (1 - s) * torch.exp(-1 / (2 * x2))) * A / B
        C = -y + 14
        return C / 15

    def _evaluate_true(self, X: Tensor) -> Tensor:
        branin = self._branin(X)
        currin = self._currin(X)
        return torch.stack([-branin, -currin], dim=-1)


class MOMFPark(MultiObjectiveTestProblem):
    r"""Modified Park test functions for multi-objective multi-fidelity optimization.

    (4+1)-dimensional function with domain ``[0,1]^5`` where the last dimension
    is the fidelity parameter ``s``. See [Irshad2021]_ for more details.

    The first modified Park function is

        P1(x, s)=A*(T1(x,s)+T2(x,s)-B)/22-0.8

    The second modified Park function is

        P2(x,s)=A*(5-2/3*exp(x1+x2)-x4*sin(x3)*A+x3-B)/4 - 0.7

    Here

        T_1(x,s) = (x1+0.001*(1-s))/2*sqrt(1+(x2+x3**2)*x4/(x1**2))

        T_2(x, s) = (x1+3*x4)*exp(1+sin(x3))

    and ``A(s)=(0.9+0.1*s)``, ``B(s)=0.1*(1-s)``.
    """

    dim = 5
    continuous_inds = list(range(5))
    num_objectives = 2
    _bounds = [(0.0, 1.0) for _ in range(dim)]
    _ref_point = [0, 0]
    _max_hv = 0.08551927363087991

    def _transform(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        x3 = X[..., 2]
        x4 = X[..., 3]
        s = X[..., 4]
        _x1 = 1 - 2 * (x1 - 0.6) ** 2
        _x2 = x2
        _x3 = 1 - 3 * (x3 - 0.5) ** 2
        _x4 = 1 - (x4 - 0.8) ** 2
        return torch.stack([_x1, _x2, _x3, _x4, s], dim=-1)

    def _park1(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        x3 = X[..., 2]
        x4 = X[..., 3]
        s = X[..., 4]
        T1 = (
            (x1 + 1e-3 * (1 - s))
            / 2
            * torch.sqrt(1 + (x2 + x3**2) * x4 / (x1**2 + 1e-4))
        )
        T2 = (x1 + 3 * x4) * torch.exp(1 + torch.sin(x3))
        A = 0.9 + 0.1 * s
        B = 0.1 * (1 - s)
        return A * (T1 + T2 - B) / 22 - 0.8

    def _park2(self, X: Tensor) -> Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        x3 = X[..., 2]
        x4 = X[..., 3]
        s = X[..., 4]
        A = 0.9 + 0.1 * s
        B = 0.1 * (1 - s)
        return (
            A * (5 - 2 / 3 * torch.exp(x1 + x2) + x4 * torch.sin(x3) * A - x3 + B) / 4
            - 0.7
        )

    def _evaluate_true(self, X: Tensor) -> Tensor:
        X = self._transform(X)
        park1 = self._park1(X)
        park2 = self._park2(X)
        return torch.stack([-park1, -park2], dim=-1)
