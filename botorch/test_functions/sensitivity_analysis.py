# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor


class Ishigami(SyntheticTestFunction):
    r"""Ishigami test function.

    three-dimensional function (usually evaluated on ``[-pi, pi]^3``):

        f(x) = sin(x_1) + a sin(x_2)^2 + b x_3^4 sin(x_1)

    Here ``a`` and ``b`` are constants where a=7 and b=0.1 or b=0.05
    Proposed to test sensitivity analysis methods because it exhibits strong
    nonlinearity and nonmonotonicity and a peculiar dependence on x_3.
    """

    def __init__(
        self,
        b: float = 0.1,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            b: the b constant, should be 0.1 or 0.05.
            noise_std: Standard deviation of the observation noise.
            negative: If True, negative the objective.
            dtype: The dtype that is used for the bounds of the function.
        """
        self._optimizers = None
        if b not in (0.1, 0.05):
            raise ValueError("b parameter should be 0.1 or 0.05")
        self.dim = 3
        self.continuous_inds = list(range(self.dim))
        if b == 0.1:
            self.si = [0.3138, 0.4424, 0]
            self.si_t = [0.558, 0.442, 0.244]
            self.s_ij = [0, 0.244, 0]
            self.dgsm_gradient = [-0.0004, -0.0004, -0.0004]
            self.dgsm_gradient_abs = [1.9, 4.45, 1.97]
            self.dgsm_gradient_square = [7.7, 24.5, 11]
        elif b == 0.05:
            self.si = [0.218, 0.687, 0]
            self.si_t = [0.3131, 0.6868, 0.095]
            self.s_ij = [0, 0.094, 0]
            self.dgsm_gradient = [-0.0002, -0.0002, -0.0002]
            self.dgsm_gradient_abs = [1.26, 4.45, 1.97]
            self.dgsm_gradient_square = [2.8, 24.5, 11]
        self._bounds = [(-math.pi, math.pi) for _ in range(self.dim)]
        self.b = b
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def compute_dgsm(self, X: Tensor) -> tuple[list[float], list[float], list[float]]:
        r"""Compute derivative global sensitivity measures.

        This function can be called separately to estimate the dgsm measure
        The exact global integrals of these values are already added under
        as attributes dgsm_gradient, dgsm_gradient_bas, and dgsm_gradient_square.

        Args:
            X: Set of points at which to compute derivative measures.

        Returns: The average gradient, absolute gradient, and square gradients.
        """
        dx_1 = torch.cos(X[..., 0]) * (1 + self.b * (X[..., 2] ** 4))
        dx_2 = 14 * torch.cos(X[..., 1]) * torch.sin(X[..., 1])
        dx_3 = 0.4 * (X[..., 2] ** 3) * torch.sin(X[..., 0])
        gradient_measure = [
            torch.mean(dx_1).item(),
            torch.mean(dx_1).item(),
            torch.mean(dx_1).item(),
        ]
        gradient_absolute_measure = [
            torch.mean(torch.abs(dx_1)).item(),
            torch.mean(torch.abs(dx_2)).item(),
            torch.mean(torch.abs(dx_3)).item(),
        ]
        gradient_square_measure = [
            torch.mean(torch.pow(dx_1, 2)).item(),
            torch.mean(torch.pow(dx_2, 2)).item(),
            torch.mean(torch.pow(dx_3, 2)).item(),
        ]
        return gradient_measure, gradient_absolute_measure, gradient_square_measure

    def _evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        t = (
            torch.sin(X[..., 0])
            + 7 * (torch.sin(X[..., 1]) ** 2)
            + self.b * (X[..., 2] ** 4) * torch.sin(X[..., 0])
        )
        return t


class Gsobol(SyntheticTestFunction):
    r"""Gsobol test function.

    d-dimensional function (usually evaluated on ``[0, 1]^d``):

        f(x) = Prod_{i=1}\^{d} ((\|4x_i-2\|+a_i)/(1+a_i)), a_i >=0

    common combinations of dimension and a vector:

        dim=8, a= [0, 1, 4.5, 9, 99, 99, 99, 99]
        dim=6, a=[0, 0.5, 3, 9, 99, 99]
        dim = 15, a= [1, 2, 5, 10, 20, 50, 100, 500, 1000, ..., 1000]

    Proposed to test sensitivity analysis methods
    First order Sobol indices have closed form expression S_i=V_i/V with :

        V_i= 1/(3(1+a_i)\^2)
        V= Prod_{i=1}\^{d} (1+V_i) - 1

    """

    def __init__(
        self,
        dim: int,
        a: list = None,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            dim: Dimensionality of the problem. If 6, 8, or 15, will use standard a.
            a: a parameter, unless dim is 6, 8, or 15.
            noise_std: Standard deviation of observation noise.
            negate: Return negative of function.
            dtype: The dtype that is used for the bounds of the function.
        """
        self._optimizers = None
        self.dim = dim
        self.continuous_inds = list(range(dim))
        self._bounds = [(0, 1) for _ in range(self.dim)]
        if self.dim == 6:
            self.a = [0, 0.5, 3, 9, 99, 99]
        elif self.dim == 8:
            self.a = [0, 1, 4.5, 9, 99, 99, 99, 99]
        elif self.dim == 15:
            self.a = [
                1,
                2,
                5,
                10,
                20,
                50,
                100,
                500,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
            ]
        else:
            self.a = a
        self.optimal_sobol_indicies()
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def optimal_sobol_indicies(self):
        vi = []
        for i in range(self.dim):
            vi.append(1 / (3 * ((1 + self.a[i]) ** 2)))
        self.vi = Tensor(vi)
        self.V = torch.prod(1 + self.vi) - 1
        self.si = self.vi / self.V
        si_t = []
        for i in range(self.dim):
            si_t.append(
                (
                    self.vi[i]
                    * torch.prod(self.vi[:i] + 1)
                    * torch.prod(self.vi[i + 1 :] + 1)
                )
                / self.V
            )
        self.si_t = Tensor(si_t)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        t = 1
        for i in range(self.dim):
            t = t * (torch.abs(4 * X[..., i] - 2) + self.a[i]) / (1 + self.a[i])
        return t


class Morris(SyntheticTestFunction):
    r"""Morris test function.

    20-dimensional function (usually evaluated on ``[0, 1]^20``):

        f(x) = sum_{i=1}\^20 beta_i w_i + sum_{i<j}\^20 beta_ij w_i w_j
        + sum_{i<j<l}\^20 beta_ijl w_i w_j w_l + 5w_1 w_2 w_3 w_4

    Proposed to test sensitivity analysis methods
    """

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of observation noise.
            negate: Return negative of function.
            dtype: The dtype that is used for the bounds of the function.
        """
        self._optimizers = None
        self.dim = 20
        self.continuous_inds = list(range(self.dim))
        self._bounds = [(0, 1) for _ in range(self.dim)]
        self.si = [
            0.005,
            0.008,
            0.017,
            0.009,
            0.016,
            0,
            0.069,
            0.1,
            0.15,
            0.1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

    @property
    def _optimal_value(self) -> float:
        raise NotImplementedError

    def _evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        W = []
        t1 = 0
        t2 = 0
        t3 = 0
        for i in range(self.dim):
            if i in [2, 4, 6]:
                wi = 2 * (1.1 * X[..., i] / (X[..., i] + 0.1) - 0.5)
            else:
                wi = 2 * (X[..., i] - 0.5)
            W.append(wi)
            if i < 10:
                betai = 20
            else:
                betai = (-1) ** (i + 1)
            t1 = t1 + betai * wi
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                if i < 6 or j < 6:
                    beta_ij = -15
                else:
                    beta_ij = (-1) ** (i + j + 2)
                t2 = t2 + beta_ij * W[i] * W[j]
                for k in range(j + 1, self.dim):
                    if i < 5 or j < 5 or k < 5:
                        beta_ijk = -10
                    else:
                        beta_ijk = 0
                    t3 = t3 + beta_ijk * W[i] * W[j] * W[k]
        t4 = 5 * W[0] * W[1] * W[2] * W[3]
        return t1 + t2 + t3 + t4
