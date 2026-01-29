#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Multi-objective optimization benchmark problems.

References

.. [Daulton2022]
    S. Daulton, S. Cakmak, M. Balandat, M. A. Osborne, E. Zhou, and E. Bakshy.
    Robust Multi-Objective Bayesian Optimization Under Input Noise.
    Proceedings of the 39th International Conference on Machine Learning, 2022.

.. [Deb2005dtlz]
    K. Deb, L. Thiele, M. Laumanns, E. Zitzler, A. Abraham, L. Jain, and
    R. Goldberg. Scalable test problems for evolutionary multi-objective
    optimization. Evolutionary Multiobjective Optimization, Springer-Verlag,
    pp. 105-145, 2005.

.. [Deb2005robust]
    K. Deb and H. Gupta. Searching for Robust Pareto-Optimal Solutions in
    Multi-objective Optimization. Evolutionary Multi-Criterion Optimization,
    Springer-Berlin, pp. 150-164, 2005.

.. [Frohlich2020]
    L. Frohlich, E. Klenske, J. Vinogradska, C. Daniel, and M. Zeilinger.
    Noisy-Input Entropy Search for Efficient Robust Bayesian Optimization.
    Proceedings of the Twenty Third International Conference on Artificial
    Intelligence and Statistics, PMLR 108:2262-2272, 2020.

.. [GarridoMerchan2020]
    E. C. Garrido-Merch ́an and D. Hern ́andez-Lobato. Parallel Predictive Entropy
    Search for Multi-objective Bayesian Optimization with Constraints.
    arXiv e-prints, arXiv:2004.00601, Apr. 2020.

.. [Gelbart2014]
    Michael A. Gelbart, Jasper Snoek, and Ryan P. Adams. 2014. Bayesian
    optimization with unknown constraints. In Proceedings of the Thirtieth
    Conference on Uncertainty in Artificial Intelligence (UAI’14).
    AUAI Press, Arlington, Virginia, USA, 250–259.

.. [Liang2021]
    Q. Liang and L. Lai, Scalable Bayesian Optimization Accelerates Process
    Optimization of Penicillin Production. NeurIPS 2021 AI for Science Workshop, 2021.

.. [Ma2019]
    Z. Ma and Y. Wang. Evolutionary Constrained Multiobjective Optimization:
    Test Suite Construction and Performance Comparisons. IEEE Transactions
    on Evolutionary Computation, 23(6):972–986, December 2019.

.. [Oszycka1995]
    A. Osyczka and S. Kundu. A new method to solve generalized
    multicriteria optimization problems using the simple genetic algorithm.
    In Structural Optimization 10. 94–99, 1995.

.. [Tanabe2020]
    Ryoji Tanabe and Hisao Ishibuchi. An easy-to-use real-world multi-objective
    optimization problem suite, Applied Soft Computing,Volume 89, 2020.

.. [Yang2019a]
    K. Yang, M. Emmerich, A. Deutz, and T. Bäck. Multi-Objective Bayesian
    Global Optimization using expected hypervolume improvement gradient.
    Swarm and evolutionary computation 44, pp. 945--956, 2019.

.. [Zitzler2000]
    E. Zitzler, K. Deb, and L. Thiele. Comparison of multiobjective
    evolutionary algorithms: Empirical results. Evolutionary Computation, vol.
    8, no. 2,pp. 173–195, 2000.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from math import pi

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.test_functions.base import (
    ConstrainedBaseTestProblem,
    MultiObjectiveTestProblem,
)
from botorch.test_functions.synthetic import Branin, Levy
from botorch.utils.sampling import sample_hypersphere, sample_simplex
from botorch.utils.transforms import unnormalize
from scipy.special import gamma
from torch import Tensor
from torch.distributions import MultivariateNormal


class BraninCurrin(MultiObjectiveTestProblem):
    r"""Two objective problem composed of the Branin and Currin functions.

    Branin (rescaled):

        f(x) = (
        15*x_1 - 5.1 * (15 * x_0 - 5) ** 2 / (4 * pi ** 2) + 5 * (15 * x_0 - 5)
        / pi - 5
        ) ** 2 + (10 - 10 / (8 * pi)) * cos(15 * x_0 - 5))

    Currin:

        f(x) = (1 - exp(-1 / (2 * x_1))) * (
        2300 * x_0 ** 3 + 1900 * x_0 ** 2 + 2092 * x_0 + 60
        ) / 100 * x_0 ** 3 + 500 * x_0 ** 2 + 4 * x_0 + 20

    """

    dim = 2
    continuous_inds = list(range(2))
    num_objectives = 2
    _bounds = [(0.0, 1.0), (0.0, 1.0)]
    _ref_point = [18.0, 6.0]
    _max_hv = 59.36011874867746  # this is approximated using NSGA-II

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
            dtype: The dtype that is used for the bounds of the function.
        """
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)
        self._branin = Branin()

    def _rescaled_branin(self, X: Tensor) -> Tensor:
        # return to Branin bounds
        x_0 = 15 * X[..., 0] - 5
        x_1 = 15 * X[..., 1]
        return self._branin(torch.stack([x_0, x_1], dim=-1))

    @staticmethod
    def _currin(X: Tensor) -> Tensor:
        x_0 = X[..., 0]
        x_1 = X[..., 1]
        factor1 = 1 - torch.exp(-1 / (2 * x_1))
        numer = 2300 * x_0.pow(3) + 1900 * x_0.pow(2) + 2092 * x_0 + 60
        denom = 100 * x_0.pow(3) + 500 * x_0.pow(2) + 4 * x_0 + 20
        return factor1 * numer / denom

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # branin rescaled with inputsto [0,1]^2
        branin = self._rescaled_branin(X=X)
        currin = self._currin(X=X)
        return torch.stack([branin, currin], dim=-1)


class DH(MultiObjectiveTestProblem, ABC):
    r"""Base class for DH problems for robust multi-objective optimization.

    In their paper, [Deb2005robust]_ consider these problems under a mean-robustness
    setting, and use uniformly distributed input perturbations from the box with
    edge lengths ``delta_0 = delta``, ``delta_i = 2 * delta, i > 0``, with
    ``delta`` ranging up to ``0.01`` for DH1 and DH2, and ``delta = 0.03`` for
    DH3 and DH4.

    These are d-dimensional problems with two objectives:

        f_0(x) = x_0
        f_1(x) = h(x) + g(x) * S(x) for DH1 and DH2
        f_1(x) = h(x) * (g(x) + S(x)) for DH3 and DH4

    The goal is to minimize both objectives. See [Deb2005robust]_ for more details
    on DH. The reference points were set using ``infer_reference_point``.
    """

    num_objectives = 2
    _ref_point: list[float] = [1.1, 1.1]
    _x_1_lb: float
    _area_under_curve: float
    _min_dim: int

    def __init__(
        self,
        dim: int,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        if dim < self._min_dim:
            raise ValueError(f"dim must be >= {self._min_dim}, but got dim={dim}!")
        self.dim = dim
        self.continuous_inds = list(range(dim))
        self._bounds = [(0.0, 1.0), (self._x_1_lb, 1.0)] + [
            (-1.0, 1.0) for _ in range(dim - 2)
        ]
        # max_hv is the area of the box minus the area of the curve formed by the PF.
        self._max_hv = self._ref_point[0] * self._ref_point[1] - self._area_under_curve
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

    @abstractmethod
    def _h(self, X: Tensor) -> Tensor:
        pass  # pragma: no cover

    @abstractmethod
    def _g(self, X: Tensor) -> Tensor:
        pass  # pragma: no cover

    @abstractmethod
    def _S(self, X: Tensor) -> Tensor:
        pass  # pragma: no cover


class DH1(DH):
    r"""DH1 test problem.

    d-dimensional problem evaluated on ``[0, 1] x [-1, 1]^{d-1}``:

        f_0(x) = x_0
        f_1(x) = h(x_0) + g(x) * S(x_0)
        h(x_0) = 1 - x_0^2
        g(x) = \sum_{i=1}^{d-1} (10 + x_i^2 - 10 * cos(4 * pi * x_i))
        S(x_0) = alpha / (0.2 + x_0) + beta * x_0^2

    where alpha = 1 and beta = 1.

    The Pareto front corresponds to the equation ``f_1 = 1 - f_0^2``, and it is found at
    ``x_i = 0`` for ``i > 0`` and any value of ``x_0`` in ``(0, 1]``.
    """

    alpha = 1.0
    beta = 1.0
    _x_1_lb = -1.0
    _area_under_curve = 2.0 / 3.0
    _min_dim = 2

    def _h(self, X: Tensor) -> Tensor:
        return 1 - X[..., 0].pow(2)

    def _g(self, X: Tensor) -> Tensor:
        x_1_to = X[..., 1:]
        return torch.sum(
            10 + x_1_to.pow(2) - 10 * torch.cos(4 * math.pi * x_1_to),
            dim=-1,
        )

    def _S(self, X: Tensor) -> Tensor:
        x_0 = X[..., 0]
        return self.alpha / (0.2 + x_0) + self.beta * x_0.pow(2)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        f_0 = X[..., 0]
        # This may encounter 0 / 0, which we set to 0.
        f_1 = self._h(X) + torch.nan_to_num(self._g(X) * self._S(X))
        return torch.stack([f_0, f_1], dim=-1)


class DH2(DH1):
    r"""DH2 test problem.

    This is identical to DH1 except for having ``beta = 10.0``.
    """

    beta = 10.0


class DH3(DH):
    r"""DH3 test problem.

    d-dimensional problem evaluated on ``[0, 1]^2 x [-1, 1]^{d-2}``:

        f_0(x) = x_0
        f_1(x) = h(x_1) * (g(x) + S(x_0))
        h(x_1) = 2 - 0.8 * exp(-((x_1 - 0.35) / 0.25)^2) - exp(-((x_1 - 0.85) / 0.03)^2)
        g(x) = \sum_{i=2}^{d-1} (50 * x_i^2)
        S(x_0) = 1 - sqrt(x_0)

    The Pareto front is found at ``x_i = 0`` for ``i > 1``. There's a local and a global
    Pareto front, which are found at ``x_1 = 0.35`` and ``x_1 = 0.85``, respectively.
    The approximate relationships between the objectives at local and global Pareto
    fronts are given by ``f_1 = 1.2 (1 - sqrt(f_0))`` and ``f_1 = 1 - f_0``,
    respectively. The specific values on the Pareto fronts can be found by varying
    ``x_0``.
    """

    _x_1_lb = 0.0
    _area_under_curve = 0.328449169794718
    _min_dim = 3

    @staticmethod
    def _exp_args(x: Tensor) -> tuple[Tensor, Tensor]:
        exp_arg_1 = -((x - 0.35) / 0.25).pow(2)
        exp_arg_2 = -((x - 0.85) / 0.03).pow(2)
        return exp_arg_1, exp_arg_2

    def _h(self, X: Tensor) -> Tensor:
        exp_arg_1, exp_arg_2 = self._exp_args(X[..., 1])
        return 2 - 0.8 * torch.exp(exp_arg_1) - torch.exp(exp_arg_2)

    def _g(self, X: Tensor) -> Tensor:
        return 50 * X[..., 2:].pow(2).sum(dim=-1)

    def _S(self, X: Tensor) -> Tensor:
        return 1 - X[..., 0].sqrt()

    def _evaluate_true(self, X: Tensor) -> Tensor:
        f_0 = X[..., 0]
        f_1 = self._h(X) * (self._g(X) + self._S(X))
        return torch.stack([f_0, f_1], dim=-1)


class DH4(DH3):
    r"""DH4 test problem.

    This is similar to DH3 except that it is evaluated on
    ``[0, 1] x [-0.15, 1] x [-1, 1]^{d-2}`` and:

        h(x_0, x_1) = 2 - x_0 - 0.8 * exp(-((x_0 + x_1 - 0.35) / 0.25)^2)
        - exp(-((x_0 + x_1 - 0.85) / 0.03)^2)

    The Pareto front is found at ``x_i = 0`` for ``i > 2``, with the local one being
    near ``x_0 + x_1 = 0.35`` and the global one near ``x_0 + x_1 = 0.85``.
    """

    _x_1_lb = -0.15
    _area_under_curve = 0.22845

    def _h(self, X: Tensor) -> Tensor:
        exp_arg_1, exp_arg_2 = self._exp_args(X[..., :2].sum(dim=-1))
        return 2 - X[..., 0] - 0.8 * torch.exp(exp_arg_1) - torch.exp(exp_arg_2)


class DTLZ(MultiObjectiveTestProblem):
    r"""Base class for DTLZ problems.

    See [Deb2005dtlz]_ for more details on DTLZ.
    """

    def __init__(
        self,
        dim: int,
        num_objectives: int = 2,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension of the function.
            num_objectives: Must be less than dim.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        if dim <= num_objectives:
            raise ValueError(
                f"dim must be > num_objectives, but got {dim} and {num_objectives}."
            )
        self.num_objectives = num_objectives
        self.dim = dim
        self.continuous_inds = list(range(dim))
        self.k = self.dim - self.num_objectives + 1
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self._ref_point = [self._ref_val for _ in range(num_objectives)]
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)


class DTLZ1(DTLZ):
    r"""DTLZ1 test problem.

    d-dimensional problem evaluated on ``[0, 1]^d``:

        f_0(x) = 0.5 * x_0 * (1 + g(x))
        f_1(x) = 0.5 * (1 - x_0) * (1 + g(x))
        g(x) = 100 * \sum_{i=m}^{d-1} (
        k + (x_i - 0.5)^2 - cos(20 * pi * (x_i - 0.5))
        )

    where k = d - m + 1.

    The pareto front is given by the line (or hyperplane) \sum_i f_i(x) = 0.5.
    The goal is to minimize both objectives. The reference point comes from [Yang2019]_.
    """

    _ref_val = 400.0

    @property
    def _max_hv(self) -> float:
        return self._ref_val**self.num_objectives - 1 / 2**self.num_objectives

    def _evaluate_true(self, X: Tensor) -> Tensor:
        X_m = X[..., -self.k :]
        X_m_minus_half = X_m - 0.5
        sum_term = (
            X_m_minus_half.pow(2) - torch.cos(20 * math.pi * X_m_minus_half)
        ).sum(dim=-1)
        g_X_m = 100 * (self.k + sum_term)
        g_X_m_term = 0.5 * (1 + g_X_m)
        fs = []
        for i in range(self.num_objectives):
            idx = self.num_objectives - 1 - i
            f_i = g_X_m_term * X[..., :idx].prod(dim=-1)
            if i > 0:
                f_i *= 1 - X[..., idx]
            fs.append(f_i)
        return torch.stack(fs, dim=-1)

    def gen_pareto_front(self, n: int) -> Tensor:
        r"""Generate ``n`` pareto optimal points.

        The pareto points randomly sampled from the hyperplane sum_i f(x_i) = 0.5.
        """
        f_X = 0.5 * sample_simplex(
            n=n,
            d=self.num_objectives,
            qmc=True,
            dtype=self.ref_point.dtype,
            device=self.ref_point.device,
        )
        if self.negate:
            f_X *= -1
        return f_X


class DTLZ2(DTLZ):
    r"""DTLZ2 test problem.

    d-dimensional problem evaluated on ``[0, 1]^d``:

        f_0(x) = (1 + g(x)) * cos(x_0 * pi / 2)
        f_1(x) = (1 + g(x)) * sin(x_0 * pi / 2)
        g(x) = \sum_{i=m}^{d-1} (x_i - 0.5)^2

    The pareto front is given by the unit hypersphere \sum{i} f_i^2 = 1.
    Note: the pareto front is completely concave. The goal is to minimize
    both objectives.
    """

    _ref_val = 1.1

    @property
    def _max_hv(self) -> float:
        # hypercube - volume of hypersphere in R^d such that all coordinates are
        # positive
        hypercube_vol = self._ref_val**self.num_objectives
        pos_hypersphere_vol = (
            math.pi ** (self.num_objectives / 2)
            / gamma(self.num_objectives / 2 + 1)
            / 2**self.num_objectives
        )
        return hypercube_vol - pos_hypersphere_vol

    def _evaluate_true(self, X: Tensor) -> Tensor:
        X_m = X[..., -self.k :]
        g_X = (X_m - 0.5).pow(2).sum(dim=-1)
        g_X_plus1 = 1 + g_X
        fs = []
        pi_over_2 = math.pi / 2
        for i in range(self.num_objectives):
            idx = self.num_objectives - 1 - i
            f_i = g_X_plus1.clone()
            f_i *= torch.cos(X[..., :idx] * pi_over_2).prod(dim=-1)
            if i > 0:
                f_i *= torch.sin(X[..., idx] * pi_over_2)
            fs.append(f_i)
        return torch.stack(fs, dim=-1)

    def gen_pareto_front(self, n: int) -> Tensor:
        r"""Generate ``n`` pareto optimal points.

        The pareto points are randomly sampled from the hypersphere's
        positive section.
        """
        f_X = sample_hypersphere(
            n=n,
            d=self.num_objectives,
            dtype=self.ref_point.dtype,
            device=self.ref_point.device,
            qmc=True,
        ).abs()
        if self.negate:
            f_X *= -1
        return f_X


class DTLZ3(DTLZ2):
    r"""DTLZ3 test problem.

    d-dimensional problem evaluated on ``[0, 1]^d``:

        f_0(x) = (1 + g(x)) * cos(x_0 * pi / 2)
        f_1(x) = (1 + g(x)) * sin(x_0 * pi / 2)
        g(x) = 100 * [k + \sum_{i=m}^{n-1} (x_i - 0.5)^2 - cos(20 * pi * (x_i - 0.5))]

    ``g(x)`` introduces (``3k−1``) local Pareto fronts that are parallel to
    the one global Pareto-optimal front.

    The global Pareto-optimal front corresponds to x_i = 0.5 for x_i in X_m.
    """

    _ref_val = 10000.0

    def _evaluate_true(self, X: Tensor) -> Tensor:
        X_m = X[..., -self.k :]
        g_X = 100 * (
            X_m.shape[-1]
            + ((X_m - 0.5).pow(2) - torch.cos(20 * math.pi * (X_m - 0.5))).sum(dim=-1)
        )
        g_X_plus1 = 1 + g_X
        fs = []
        pi_over_2 = math.pi / 2
        for i in range(self.num_objectives):
            idx = self.num_objectives - 1 - i
            f_i = g_X_plus1.clone()
            f_i *= torch.cos(X[..., :idx] * pi_over_2).prod(dim=-1)
            if i > 0:
                f_i *= torch.sin(X[..., idx] * pi_over_2)
            fs.append(f_i)
        return torch.stack(fs, dim=-1)


class DTLZ4(DTLZ2):
    r"""DTLZ4 test problem.

    This is the same as DTLZ2, but with alpha=100 as the exponent,
    resulting in dense solutions near the f_M-f_1 plane.

    The global Pareto-optimal front corresponds to x_i = 0.5 for x_i in X_m.
    """

    _alpha = 100.0


class DTLZ5(DTLZ):
    r"""DTLZ5 test problem.

    d-dimensional problem evaluated on ``[0, 1]^d``:

        f_0(x) = (1 + g(x)) * cos(theta_0 * pi / 2)
        f_1(x) = (1 + g(x)) * sin(theta_0 * pi / 2)
        theta_i = pi / (4 * (1 + g(X_m)) * (1 + 2 * g(X_m) * x_i)) for i = 1, ... , M-2
        g(x) = \sum_{i=m}^{d-1} (x_i - 0.5)^2

    The global Pareto-optimal front corresponds to x_i = 0.5 for x_i in X_m.
    """

    _ref_val = 10.0

    def _evaluate_true(self, X: Tensor) -> Tensor:
        X_m = X[..., -self.k :]
        X_ = X[..., : -self.k]
        g_X = (X_m - 0.5).pow(2).sum(dim=-1)
        theta = 1 / (2 * (1 + g_X.unsqueeze(-1))) * (1 + 2 * g_X.unsqueeze(-1) * X_)
        theta = torch.cat([X[..., :1], theta[..., 1:]], dim=-1)
        fs = []
        pi_over_2 = math.pi / 2
        g_X_plus1 = g_X + 1
        for i in range(self.num_objectives):
            f_i = g_X_plus1.clone()
            f_i *= torch.cos(theta[..., : theta.shape[-1] - i] * pi_over_2).prod(dim=-1)
            if i > 0:
                f_i *= torch.sin(theta[..., theta.shape[-1] - i] * pi_over_2)
            fs.append(f_i)
        return torch.stack(fs, dim=-1)


class DTLZ7(DTLZ):
    r"""DTLZ7 test problem.

    d-dimensional problem evaluated on ``[0, 1]^d``:
        f_0(x) = x_0
        f_1(x) = x_1
        ...
        f_{M-1}(x) = (1 + g(X_m)) * h(f_0, f_1, ..., f_{M-2}, g, x)
        h(f_0, f_1, ..., f_{M-2}, g, x) =
        M - sum_{i=0}^{M-2} f_i(x)/(1+g(x)) * (1 + sin(3 * pi * f_i(x)))

    This test problem has 2M-1 disconnected Pareto-optimal regions in the search space.

    The pareto frontier corresponds to X_m = 0.
    """

    _ref_val = 15.0

    def _evaluate_true(self, X):
        f = []
        for i in range(0, self.num_objectives - 1):
            f.append(X[..., i])
        f = torch.stack(f, dim=-1)

        g_X = 1 + 9 / self.k * torch.sum(X[..., -self.k :], dim=-1)
        h = self.num_objectives - torch.sum(
            f / (1 + g_X.unsqueeze(-1)) * (1 + torch.sin(3 * math.pi * f)), dim=-1
        )
        return torch.cat([f, ((1 + g_X) * h).unsqueeze(-1)], dim=-1)


class GMM(MultiObjectiveTestProblem):
    r"""A test problem where each objective is a Gaussian mixture model.

    This implementation is adapted from the single objective version (proposed by
    [Frohlich2020]_) at
    https://github.com/boschresearch/NoisyInputEntropySearch/blob/master/
    core/util/objectives.py.

    See [Daulton2022]_ for details on this multi-objective problem.
    """

    dim = 2
    continuous_inds = list(range(dim))
    _bounds = [(0.0, 1.0), (0.0, 1.0)]

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        num_objectives: int = 2,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
            num_objectives: The number of objectives.
            dtype: The dtype that is used for the bounds of the function.
        """
        if num_objectives not in (2, 3, 4):
            raise UnsupportedError("GMM only currently supports 2 to 4 objectives.")
        self._ref_point = [-0.2338, -0.2211]
        if num_objectives > 2:
            self._ref_point.append(-0.5180)
        if num_objectives > 3:
            self._ref_point.append(-0.1866)
        self.num_objectives = num_objectives
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)
        gmm_pos = torch.tensor(
            [
                [[0.2, 0.2], [0.8, 0.2], [0.5, 0.7]],
                [[0.07, 0.2], [0.4, 0.8], [0.85, 0.1]],
            ]
        )
        gmm_var = torch.tensor([[0.20, 0.10, 0.10], [0.2, 0.1, 0.05]]).pow(2)
        gmm_norm = 2 * pi * gmm_var * torch.tensor([0.5, 0.7, 0.7])
        if num_objectives > 2:
            gmm_pos = torch.cat(
                [gmm_pos, torch.tensor([[[0.08, 0.21], [0.45, 0.75], [0.86, 0.11]]])],
                dim=0,
            )
            gmm_var = torch.cat(
                [gmm_var, torch.tensor([[0.2, 0.1, 0.07]]).pow(2)], dim=0
            )
            gmm_norm = torch.cat(
                [
                    gmm_norm,
                    2 * pi * gmm_var[2] * torch.tensor([[0.5, 0.7, 0.9]]),
                ],
                dim=0,
            )
        if num_objectives > 3:
            gmm_pos = torch.cat(
                [gmm_pos, torch.tensor([[[0.09, 0.19], [0.44, 0.72], [0.89, 0.13]]])],
                dim=0,
            )
            gmm_var = torch.cat(
                [gmm_var, torch.tensor([[0.15, 0.07, 0.09]]).pow(2)], dim=0
            )
            gmm_norm = torch.cat(
                [
                    gmm_norm,
                    2 * pi * gmm_var[3] * torch.tensor([[0.5, 0.7, 0.9]]),
                ],
                dim=0,
            )
        gmm_covar = gmm_var.view(*gmm_var.shape, 1, 1) * torch.eye(
            2, dtype=gmm_var.dtype, device=gmm_var.device
        )
        self.register_buffer("gmm_pos", gmm_pos)
        self.register_buffer("gmm_covar", gmm_covar)
        self.register_buffer("gmm_norm", gmm_norm)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        r"""Evaluate the GMMs."""
        # This needs to be reinstantiated because MVN apparently does not
        # have a ``to`` method to make it device/dtype agnostic.
        mvn = MultivariateNormal(loc=self.gmm_pos, covariance_matrix=self.gmm_covar)
        view_shape = (
            X.shape[:-1]
            + torch.Size([1] * (self.gmm_pos.ndim - 1))
            + self.gmm_pos.shape[-1:]
        )
        expand_shape = X.shape[:-1] + self.gmm_pos.shape
        pdf_X = mvn.log_prob(X.view(view_shape).expand(expand_shape)).exp()
        # Multiply by -1 to make this a minimization problem by default
        return -(self.gmm_norm * pdf_X).sum(dim=-1)


class Penicillin(MultiObjectiveTestProblem):
    r"""A penicillin production simulator from [Liang2021]_.

    This implementation is adapted from
    https://github.com/HarryQL/TuRBO-Penicillin.

    The goal is to maximize the penicillin yield while minimizing
    time to ferment and the CO2 byproduct.

    The function is defined for minimization of all objectives.

    The reference point was set using the ``infer_reference_point`` heuristic
    on the Pareto frontier obtained via NSGA-II.
    """

    _max_hv = 2183455.909507436
    dim = 7
    continuous_inds = list(range(dim))
    num_objectives = 3
    _bounds = [
        (60.0, 120.0),
        (0.05, 18.0),
        (293.0, 303.0),
        (0.05, 18.0),
        (0.01, 0.5),
        (500.0, 700.0),
        (5.0, 6.5),
    ]
    _ref_point = [25.935, 57.612, 935.5]

    Y_xs = 0.45
    Y_ps = 0.90
    K_1 = 10 ** (-10)
    K_2 = 7 * 10 ** (-5)
    m_X = 0.014
    alpha_1 = 0.143
    alpha_2 = 4 * 10 ** (-7)
    alpha_3 = 10 ** (-4)
    mu_X = 0.092
    K_X = 0.15
    mu_p = 0.005
    K_p = 0.0002
    K_I = 0.10
    K = 0.04
    k_g = 7.0 * 10**3
    E_g = 5100.0
    k_d = 10.0**33
    E_d = 50000.0
    lambd = 2.5 * 10 ** (-4)
    T_v = 273.0  # Kelvin
    T_o = 373.0
    R = 1.9872  # CAL/(MOL K)
    V_max = 180.0

    @classmethod
    def penicillin_vectorized(cls, X_input: Tensor) -> Tensor:
        r"""Penicillin simulator, simplified and vectorized.

        The 7 input parameters are (in order): culture volume, biomass
        concentration, temperature, glucose concentration, substrate feed
        rate, substrate feed concentration, and H+ concentration.

        Args:
            X_input: A ``n x 7``-dim tensor of inputs.

        Returns:
            An ``n x 3``-dim tensor of (negative) penicillin yield, CO2 and time.
        """
        V, X, T, S, F, s_f, H_ = torch.split(X_input, 1, -1)
        P, CO2 = torch.zeros_like(V), torch.zeros_like(V)
        H = torch.full_like(H_, 10.0).pow(-H_)

        active = torch.ones_like(V).bool()
        t_tensor = torch.full_like(V, 2500)

        for t in range(1, 2501):
            if active.sum() == 0:
                break
            F_loss = (
                V[active]
                * cls.lambd
                * torch.special.expm1(5 * ((T[active] - cls.T_o) / (cls.T_v - cls.T_o)))
            )
            dV_dt = F[active] - F_loss
            mu = (
                (cls.mu_X / (1 + cls.K_1 / H[active] + H[active] / cls.K_2))
                * (S[active] / (cls.K_X * X[active] + S[active]))
                * (
                    (cls.k_g * torch.exp(-cls.E_g / (cls.R * T[active])))
                    - (cls.k_d * torch.exp(-cls.E_d / (cls.R * T[active])))
                )
            )
            dX_dt = mu * X[active] - (X[active] / V[active]) * dV_dt
            mu_pp = cls.mu_p * (
                S[active] / (cls.K_p + S[active] + S[active].pow(2) / cls.K_I)
            )
            dS_dt = (
                -(mu / cls.Y_xs) * X[active]
                - (mu_pp / cls.Y_ps) * X[active]
                - cls.m_X * X[active]
                + F[active] * s_f[active] / V[active]
                - (S[active] / V[active]) * dV_dt
            )
            dP_dt = (
                (mu_pp * X[active])
                - cls.K * P[active]
                - (P[active] / V[active]) * dV_dt
            )
            dCO2_dt = cls.alpha_1 * dX_dt + cls.alpha_2 * X[active] + cls.alpha_3

            # UPDATE
            P[active] = P[active] + dP_dt  # Penicillin concentration
            V[active] = V[active] + dV_dt  # Culture medium volume
            X[active] = X[active] + dX_dt  # Biomass concentration
            S[active] = S[active] + dS_dt  # Glucose concentration
            CO2[active] = CO2[active] + dCO2_dt  # CO2 concentration

            # Update active indices
            full_dpdt = torch.ones_like(P)
            full_dpdt[active] = dP_dt
            inactive = (V > cls.V_max) + (S < 0) + (full_dpdt < 10e-12)
            t_tensor[inactive] = torch.minimum(
                t_tensor[inactive], torch.full_like(t_tensor[inactive], t)
            )
            active[inactive] = 0

        return torch.stack([-P, CO2, t_tensor], dim=-1)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # This uses in-place operations. Hence, the clone is to avoid modifying
        # the original X in-place.
        return self.penicillin_vectorized(X.view(-1, self.dim).clone()).view(
            *X.shape[:-1], self.num_objectives
        )


class ToyRobust(MultiObjectiveTestProblem):
    r"""A 1D problem where the Pareto frontier is sensitive to input noise.

    Specifically, the pareto frontier over the nominal objectives is
    sensitive to input noise. The first objective is a mixture of a linear
    function and a sinusoidal function, and the second objective is a modified
    Levy function, where the second parameter is fixed.

    This function comes from [Daulton2022]_.

    The reference point was set using the ``infer_reference_point``
    heuristic on the Pareto frontier over a large discrete set of
    random designs.
    """

    dim = 1
    continuous_inds = list(range(dim))
    _bounds = [(0.0, 0.7)]
    _ref_point = [-6.1397, -8.1942]
    num_objectives = 2
    levy = Levy()

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        self.levy.bounds = self.levy.bounds.to(X)
        return super().forward(X=X, noise=noise)

    def f_1(self, X: Tensor) -> Tensor:
        p1 = 2.4 - 10 * X - 0.1 * X.pow(2)
        p2 = 2 * X - 0.1 * X.pow(2)
        smoother = (X - 0.5).pow(2) + torch.sin(30 * X) * 0.1
        x_mask = torch.sigmoid((0.2 - X) / 0.005)
        return -(p1 * x_mask + p2 * (1 - x_mask) + smoother) * 30 + 30

    def f_2(self, X: Tensor) -> Tensor:
        X = torch.cat(
            [X, torch.zeros_like(X)],
            dim=-1,
        )
        # Cut out the first part of the function.
        X = X * 0.95 + 0.03
        X = unnormalize(X, self.levy.bounds.to(X))
        Y = self.levy(X).unsqueeze(-1)
        Y -= X[..., :1].pow(2) * 0.75
        return Y

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return -torch.cat([self.f_1(X), self.f_2(X)], dim=-1)


class VehicleSafety(MultiObjectiveTestProblem):
    r"""Optimize Vehicle crash-worthiness.

    See [Tanabe2020]_ for details.

    The reference point is 1.1 * the nadir point from
    approximate front provided by [Tanabe2020]_.

    The maximum hypervolume is computed using the approximate
    pareto front from [Tanabe2020]_.
    """

    _ref_point = [1864.72022, 11.81993945, 0.2903999384]
    _max_hv = 246.81607081187002
    _bounds = [(1.0, 3.0)] * 5
    dim = 5
    continuous_inds = list(range(dim))
    num_objectives = 3

    def _evaluate_true(self, X: Tensor) -> Tensor:
        X1, X2, X3, X4, X5 = torch.split(X, 1, -1)
        f1 = (
            1640.2823
            + 2.3573285 * X1
            + 2.3220035 * X2
            + 4.5688768 * X3
            + 7.7213633 * X4
            + 4.4559504 * X5
        )
        f2 = (
            6.5856
            + 1.15 * X1
            - 1.0427 * X2
            + 0.9738 * X3
            + 0.8364 * X4
            - 0.3695 * X1 * X4
            + 0.0861 * X1 * X5
            + 0.3628 * X2 * X4
            - 0.1106 * X1.pow(2)
            - 0.3437 * X3.pow(2)
            + 0.1764 * X4.pow(2)
        )
        f3 = (
            -0.0551
            + 0.0181 * X1
            + 0.1024 * X2
            + 0.0421 * X3
            - 0.0073 * X1 * X2
            + 0.024 * X2 * X3
            - 0.0118 * X2 * X4
            - 0.0204 * X3 * X4
            - 0.008 * X3 * X5
            - 0.0241 * X2.pow(2)
            + 0.0109 * X4.pow(2)
        )
        f_X = torch.cat([f1, f2, f3], dim=-1)
        return f_X


class ZDT(MultiObjectiveTestProblem):
    r"""Base class for ZDT problems.

    See [Zitzler2000]_ for more details on ZDT.
    """

    _ref_point = [11.0, 11.0]

    def __init__(
        self,
        dim: int,
        num_objectives: int = 2,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension of the function.
            num_objectives: Number of objectives. Must not be larger than dim.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        if num_objectives != 2:
            raise NotImplementedError(
                f"{type(self).__name__} currently only supports 2 objectives."
            )
        if dim < num_objectives:
            raise ValueError(
                f"dim must be >= num_objectives, but got {dim} and {num_objectives}"
            )
        self.num_objectives = num_objectives
        self.dim = dim
        self.continuous_inds = list(range(dim))
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

    @staticmethod
    def _g(X: Tensor) -> Tensor:
        return 1 + 9 * X[..., 1:].mean(dim=-1)


class ZDT1(ZDT):
    r"""ZDT1 test problem.

    d-dimensional problem evaluated on ``[0, 1]^d``:

        f_0(x) = x_0
        f_1(x) = g(x) * (1 - sqrt(x_0 / g(x))
        g(x) = 1 + 9 / (d - 1) * \sum_{i=1}^{d-1} x_i

    The reference point comes from [Yang2019a]_.

    The pareto front is convex.
    """

    _max_hv = 120 + 2 / 3

    def _evaluate_true(self, X: Tensor) -> Tensor:
        f_0 = X[..., 0]
        g = self._g(X=X)
        f_1 = g * (1 - (f_0 / g).sqrt())
        return torch.stack([f_0, f_1], dim=-1)

    def gen_pareto_front(self, n: int) -> Tensor:
        f_0 = torch.linspace(
            0, 1, n, dtype=self.bounds.dtype, device=self.bounds.device
        )
        f_1 = 1 - f_0.sqrt()
        f_X = torch.stack([f_0, f_1], dim=-1)
        if self.negate:
            f_X *= -1
        return f_X


class ZDT2(ZDT):
    r"""ZDT2 test problem.

    d-dimensional problem evaluated on ``[0, 1]^d``:

        f_0(x) = x_0
        f_1(x) = g(x) * (1 - (x_0 / g(x))^2)
        g(x) = 1 + 9 / (d - 1) * \sum_{i=1}^{d-1} x_i

    The reference point comes from [Yang2019a]_.

    The pareto front is concave.
    """

    _max_hv = 120 + 1 / 3

    def _evaluate_true(self, X: Tensor) -> Tensor:
        f_0 = X[..., 0]
        g = self._g(X=X)
        f_1 = g * (1 - (f_0 / g).pow(2))
        return torch.stack([f_0, f_1], dim=-1)

    def gen_pareto_front(self, n: int) -> Tensor:
        f_0 = torch.linspace(
            0, 1, n, dtype=self.bounds.dtype, device=self.bounds.device
        )
        f_1 = 1 - f_0.pow(2)
        f_X = torch.stack([f_0, f_1], dim=-1)
        if self.negate:
            f_X *= -1
        return f_X


class ZDT3(ZDT):
    r"""ZDT3 test problem.

    d-dimensional problem evaluated on ``[0, 1]^d``:

        f_0(x) = x_0
        f_1(x) = 1 - sqrt(x_0 / g(x)) - x_0 / g * sin(10 * pi * x_0)
        g(x) = 1 + 9 / (d - 1) * \sum_{i=1}^{d-1} x_i

    The reference point comes from [Yang2019a]_.

    The pareto front consists of several discontinuous convex parts.
    """

    _max_hv = 128.77811613069076060
    _parts = [
        # this interval includes both end points
        [0, 0.0830015349],
        # this interval includes only the right end points
        [0.1822287280, 0.2577623634],
        [0.4093136748, 0.4538821041],
        [0.6183967944, 0.6525117038],
        [0.8233317983, 0.8518328654],
    ]
    # nugget to make sure linspace returns elements within the specified range
    _eps = 1e-6

    def _evaluate_true(self, X: Tensor) -> Tensor:
        f_0 = X[..., 0]
        g = self._g(X=X)
        f_1 = 1 - (f_0 / g).sqrt() - f_0 / g * torch.sin(10 * math.pi * f_0)
        return torch.stack([f_0, f_1], dim=-1)

    def gen_pareto_front(self, n: int) -> Tensor:
        n_parts = len(self._parts)
        n_per_part = torch.full(
            torch.Size([n_parts]),
            n // n_parts,
            dtype=torch.long,
            device=self.bounds.device,
        )
        left_over = n % n_parts
        n_per_part[:left_over] += 1
        f_0s = []
        for i, p in enumerate(self._parts):
            left, right = p
            f_0s.append(
                torch.linspace(
                    left + self._eps,
                    right - self._eps,
                    int(n_per_part[i]),
                    dtype=self.bounds.dtype,
                    device=self.bounds.device,
                )
            )
        f_0 = torch.cat(f_0s, dim=0)
        f_1 = 1 - f_0.sqrt() - f_0 * torch.sin(10 * math.pi * f_0)
        f_X = torch.stack([f_0, f_1], dim=-1)
        if self.negate:
            f_X *= -1
        return f_X


class CarSideImpact(MultiObjectiveTestProblem):
    r"""Car side impact problem.

    See [Tanabe2020]_ for details.

    The reference point is ``nadir + 0.1 * (ideal - nadir)``
    where the ideal and nadir points come from the approximate
    Pareto frontier from [Tanabe2020]_. The max_hv was computed
    based on the approximate Pareto frontier from [Tanabe2020]_.
    """

    num_objectives: int = 4
    dim: int = 7
    continuous_inds = list(range(dim))
    _bounds = [
        (0.5, 1.5),
        (0.45, 1.35),
        (0.5, 1.5),
        (0.5, 1.5),
        (0.875, 2.625),
        (0.4, 1.2),
        (0.4, 1.2),
    ]
    _ref_point = [45.4872, 4.5114, 13.3394, 10.3942]
    _max_hv = 484.72654347642793

    def _evaluate_true(self, X: Tensor) -> Tensor:
        X1, X2, X3, X4, X5, X6, X7 = torch.split(X, 1, -1)
        f1 = (
            1.98
            + 4.9 * X1
            + 6.67 * X2
            + 6.98 * X3
            + 4.01 * X4
            + 1.78 * X5
            + 10**-5 * X6
            + 2.73 * X7
        )
        f2 = 4.72 - 0.5 * X4 - 0.19 * X2 * X3
        V_MBP = 10.58 - 0.674 * X1 * X2 - 0.67275 * X2
        V_FD = 16.45 - 0.489 * X3 * X7 - 0.843 * X5 * X6
        f3 = 0.5 * (V_MBP + V_FD)
        g1 = 1 - 1.16 + 0.3717 * X2 * X4 + 0.0092928 * X3
        g2 = (
            0.32
            - 0.261
            + 0.0159 * X1 * X2
            + 0.06486 * X1
            + 0.019 * X2 * X7
            - 0.0144 * X3 * X5
            - 0.0154464 * X6
        )
        g3 = (
            0.32
            - 0.214
            - 0.00817 * X5
            + 0.045195 * X1
            + 0.0135168 * X1
            - 0.03099 * X2 * X6
            + 0.018 * X2 * X7
            - 0.007176 * X3
            - 0.023232 * X3
            + 0.00364 * X5 * X6
            + 0.018 * X2.pow(2)
        )
        g4 = 0.32 - 0.74 + 0.61 * X2 + 0.031296 * X3 + 0.031872 * X7 - 0.227 * X2.pow(2)
        g5 = 32 - 28.98 - 3.818 * X3 + 4.2 * X1 * X2 - 1.27296 * X6 + 2.68065 * X7
        g6 = (
            32
            - 33.86
            - 2.95 * X3
            + 5.057 * X1 * X2
            + 3.795 * X2
            + 3.4431 * X7
            - 1.45728
        )
        g7 = 32 - 46.36 + 9.9 * X2 + 4.4505 * X1
        g8 = 4 - f2
        g9 = 9.9 - V_MBP
        g10 = 15.7 - V_FD
        g = torch.cat([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10], dim=-1)
        zero = torch.tensor(0.0, dtype=X.dtype, device=X.device)
        g = torch.where(g < 0, -g, zero)
        f4 = g.sum(dim=-1, keepdim=True)
        return torch.cat([f1, f2, f3, f4], dim=-1)


# ------ Constrained Multi-Objective Test Problems ----- #


class BNH(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    r"""The constrained BNH problem.

    See [GarridoMerchan2020]_ for more details on this problem. Note that this is a
    minimization problem.
    """

    dim = 2
    continuous_inds = list(range(dim))
    num_objectives = 2
    num_constraints = 2
    _bounds = [(0.0, 5.0), (0.0, 3.0)]
    _ref_point = [0.0, 0.0]  # TODO: Determine proper reference point

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return torch.stack(
            [4.0 * X.pow(2).sum(dim=-1), (X - 5.0).pow(2).sum(dim=-1)], dim=-1
        )

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        c1 = 25.0 - (X[..., 0] - 5.0).pow(2) - X[..., 1].pow(2)
        c2 = (X[..., 0] - 8.0).pow(2) + (X[..., 1] + 3.0).pow(2) - 7.7
        return torch.stack([c1, c2], dim=-1)


class CONSTR(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    r"""The constrained CONSTR problem.

    See [GarridoMerchan2020]_ for more details on this problem. Note that this is a
    minimization problem.
    """

    dim = 2
    continuous_inds = list(range(dim))
    num_objectives = 2
    num_constraints = 2
    _bounds = [(0.1, 10.0), (0.0, 5.0)]
    _ref_point = [10.0, 10.0]

    def _evaluate_true(self, X: Tensor) -> Tensor:
        obj1 = X[..., 0]
        obj2 = (1.0 + X[..., 1]) / X[..., 0]
        return torch.stack([obj1, obj2], dim=-1)

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        c1 = 9.0 * X[..., 0] + X[..., 1] - 6.0
        c2 = 9.0 * X[..., 0] - X[..., 1] - 1.0
        return torch.stack([c1, c2], dim=-1)


class ConstrainedBraninCurrin(BraninCurrin, ConstrainedBaseTestProblem):
    r"""Constrained Branin Currin Function.

    This uses the disk constraint from [Gelbart2014]_.
    """

    dim = 2
    continuous_inds = list(range(dim))
    num_objectives = 2
    num_constraints = 1
    _bounds = [(0.0, 1.0), (0.0, 1.0)]
    _con_bounds = [(-5.0, 10.0), (0.0, 15.0)]
    _ref_point = [80.0, 12.0]
    _max_hv = 608.4004237022673  # from NSGA-II with 90k evaluations

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        constraint_noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise of the objectives.
            constraint_noise_std: Standard deviation of the observation noise of the
                constraint.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)
        con_bounds = torch.tensor(self._con_bounds, dtype=self.bounds.dtype).transpose(
            -1, -2
        )
        self.register_buffer("con_bounds", con_bounds)
        self.constraint_noise_std = constraint_noise_std

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self.con_bounds)
        return 50 - (X_tf[..., 0:1] - 2.5).pow(2) - (X_tf[..., 1:2] - 7.5).pow(2)


class C2DTLZ2(DTLZ2, ConstrainedBaseTestProblem):
    num_constraints = 1
    _r = 0.2
    # approximate from nsga-ii, TODO: replace with analytic
    _max_hv = 0.3996406303723544

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        if X.ndim > 2:
            raise NotImplementedError("Batch X is not supported.")
        f_X = self.evaluate_true(X=X)
        term1 = (f_X - 1).pow(2)
        mask = ~(torch.eye(f_X.shape[-1], device=f_X.device).bool())
        indices = torch.arange(f_X.shape[1], device=f_X.device).repeat(f_X.shape[1], 1)
        indexer = indices[mask].view(f_X.shape[1], f_X.shape[-1] - 1)
        term2_inner = (
            f_X.unsqueeze(1)
            .expand(f_X.shape[0], f_X.shape[-1], f_X.shape[-1])
            .gather(dim=-1, index=indexer.repeat(f_X.shape[0], 1, 1))
        )
        term2 = (term2_inner.pow(2) - self._r**2).sum(dim=-1)
        min1 = (term1 + term2).min(dim=-1).values
        min2 = ((f_X - 1 / math.sqrt(f_X.shape[-1])).pow(2) - self._r**2).sum(dim=-1)
        return -torch.min(min1, min2).unsqueeze(-1)


class DiscBrake(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    r"""The Disc Brake problem.

    There are 2 objectives and 4 constraints.

    Both objectives should be minimized.

    See [Tanabe2020]_ for details.

    The reference point was set using the ``infer_reference_point``
    heuristic on the Pareto frontier over a large discrete set of
    random designs.
    """

    dim = 4
    continuous_inds = list(range(dim))
    num_objectives = 2
    num_constraints = 4
    _bounds = [(55.0, 80.0), (75.0, 110.0), (1000.0, 3000.0), (11.0, 20.0)]
    _ref_point = [5.7771, 3.9651]

    def _evaluate_true(self, X: Tensor) -> Tensor:
        f = torch.zeros(
            *X.shape[:-1], self.num_objectives, dtype=X.dtype, device=X.device
        )

        X1, X2, X3, X4 = torch.split(X, 1, -1)
        sq_diff = X2.pow(2) - X1.pow(2)
        f[..., :1] = 4.9 * 1e-5 * sq_diff * (X4 - 1.0)
        f[..., 1:] = (9.82 * 1e6) * sq_diff / (X3 * X4 * (X2.pow(3) - X1.pow(3)))

        return f

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        g = torch.zeros(
            *X.shape[:-1], self.num_constraints, dtype=X.dtype, device=X.device
        )
        X1, X2, X3, X4 = torch.split(X, 1, -1)
        sq_diff = X2.pow(2) - X1.pow(2)
        cub_diff = X2.pow(3) - X1.pow(3)
        g[..., :1] = X2 - X1 - 20.0
        g[..., 1:2] = 0.4 - X3 / (3.14 * sq_diff)
        g[..., 2:3] = 1.0 - (2.22 * 1e-3 * X3 * cub_diff) / sq_diff.pow(2)
        g[..., 3:] = (2.66 * 1e-2 * X3 * X4 * cub_diff) / sq_diff - 900.0
        return g


class MW7(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    r"""The MW7 problem.

    This problem has 2 objectives, 2 constraints, and a disconnected Pareto
    frontier. It supports arbitrary input dimension > 1. See [Ma2019]_ for details.

    This implementation is adapted from:
    https://github.com/anyoptimization/pymoo/blob/master/pymoo/problems/multi/mw.py
    """

    num_constraints = 2
    num_objectives = 2
    _ref_point = [1.2, 1.2]

    def __init__(
        self,
        dim: int,
        noise_std: None | float | list[float] = None,
        constraint_noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension of the function. Must be at least 2.
            noise_std: Standard deviation of the observation noise of the objectives.
            constraint_noise_std: Standard deviation of the observation noise of the
                constraints.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        if dim < 2:
            raise ValueError("dim must be greater than or equal to 2.")
        self.dim = dim
        self.continuous_inds = list(range(dim))
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)
        self.constraint_noise_std = constraint_noise_std

    def LA2(self, A, B, C, D, theta) -> Tensor:
        return A * torch.sin(B * theta.pow(C)).pow(D)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        a = X[..., :-1] - 0.5
        contrib = 2 * (X[..., 1:] + a.pow(2) - 1).pow(2)
        g = 1 + contrib.sum(dim=-1)
        f0 = g * X[..., 0]
        f1 = g * torch.sqrt(1 - (f0 / g).pow(2))
        return torch.stack([f0, f1], dim=-1)

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        ff = self.evaluate_true(X=X)
        f0, f1 = ff[..., 0], ff[..., 1]
        atan = torch.arctan(f1 / f0)
        g0 = (
            f0.pow(2)
            + f1.pow(2)
            - (1.2 + (self.LA2(0.4, 4.0, 1.0, 16.0, atan)).abs()).pow(2)
        )
        g1 = (1.15 - self.LA2(0.2, 4.0, 1.0, 8.0, atan)).pow(2) - f0.pow(2) - f1.pow(2)
        return -torch.stack([g0, g1], dim=-1)


class OSY(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    r"""
    The OSY test problem from [Oszycka1995]_.
    Implementation from
    https://github.com/msu-coinlab/pymoo/blob/master/pymoo/problems/multi/osy.py
    Note that this implementation assumes minimization, so please choose negate=True.
    """

    dim = 6
    continuous_inds = list(range(dim))
    num_constraints = 6
    num_objectives = 2
    _bounds = [
        (0.0, 10.0),
        (0.0, 10.0),
        (1.0, 5.0),
        (0.0, 6.0),
        (1.0, 5.0),
        (0.0, 10.0),
    ]
    _ref_point = [-75.0, 75.0]

    def _evaluate_true(self, X: Tensor) -> Tensor:
        f1 = -(
            25 * (X[..., 0] - 2).pow(2)
            + (X[..., 1] - 2).pow(2)
            + (X[..., 2] - 1).pow(2)
            + (X[..., 3] - 4).pow(2)
            + (X[..., 4] - 1).pow(2)
        )
        f2 = X.pow(2).sum(-1)
        return torch.stack([f1, f2], dim=-1)

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        g1 = X[..., 0] + X[..., 1] - 2.0
        g2 = 6.0 - X[..., 0] - X[..., 1]
        g3 = 2.0 - X[..., 1] + X[..., 0]
        g4 = 2.0 - X[..., 0] + 3.0 * X[..., 1]
        g5 = 4.0 - (X[..., 2] - 3.0).pow(2) - X[..., 3]
        g6 = (X[..., 4] - 3.0).pow(2) + X[..., 5] - 4.0
        return torch.stack([g1, g2, g3, g4, g5, g6], dim=-1)


class SRN(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    r"""The constrained SRN problem.

    See [GarridoMerchan2020]_ for more details on this problem. Note that this is a
    minimization problem.
    """

    dim = 2
    continuous_inds = list(range(dim))
    num_objectives = 2
    num_constraints = 2
    _bounds = [(-20.0, 20.0), (-20.0, 20.0)]
    _ref_point = [0.0, 0.0]  # TODO: Determine proper reference point

    def _evaluate_true(self, X: Tensor) -> Tensor:
        obj1 = 2.0 + (X - 2.0).pow(2).sum(dim=-1)
        obj2 = 9.0 * X[..., 0] - (X[..., 1] - 1.0).pow(2)
        return torch.stack([obj1, obj2], dim=-1)

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        c1 = 225.0 - (X.pow(2)).pow(2).sum(dim=-1)
        c2 = -10.0 - X[..., 0] + 3 * X[..., 1]
        return torch.stack([c1, c2], dim=-1)


class WeldedBeam(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    r"""
    The Welded Beam multi-objective test problem. Similar to ``WeldedBeamSO`` in
    ``botorch.test_function.synthetic``, but with an additional output, somewhat
    modified constraints, and a different domain.

    Implementation from
    https://github.com/msu-coinlab/pymoo/blob/master/pymoo/problems/multi/welded_beam.py
    Note that this implementation assumes minimization, so please choose negate=True.
    """

    dim = 4
    continuous_inds = list(range(dim))
    num_constraints = 4
    num_objectives = 2
    _bounds = [
        (0.125, 5.0),
        (0.1, 10.0),
        (0.1, 10.0),
        (0.125, 5.0),
    ]
    _ref_point = [40, 0.015]

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # We could do the following, but the constraints are using somewhat
        # different numbers (see below).
        # f1 = WeldedBeam.evaluate_true(self, X)
        x1, x2, x3, x4 = X.unbind(-1)
        f1 = 1.10471 * x1.pow(2) * x2 + 0.04811 * x3 * x4 * (14.0 + x2)
        f2 = 2.1952 / (x4 * x3.pow(3))
        return torch.stack([f1, f2], dim=-1)

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        x1, x2, x3, x4 = X.unbind(-1)
        P = 6000.0
        L = 14.0
        t_max = 13600.0
        s_max = 30000.0

        # Ideally, we could just do the following, but the numbers in the
        # single-outcome WeldedBeam are different (see below)
        # g1_, g2_, g3_, _, _, g6_ = WeldedBeam.evaluate_slack_true(self, X)
        # g1 = g1_ / t_max
        # g2 = g2_ / s_max
        # g3 = 1 / (5 - 0.125) * g3_
        # g4 = 1 / P * g6_

        R = torch.sqrt(0.25 * (x2.pow(2) + (x1 + x3).pow(2)))
        M = P * (L + x2 / 2)
        # This ``J`` is different than the one in [CoelloCoello2002constraint]_
        # by a factor of 2 (sqrt(2) instead of sqrt(0.5))
        J = 2 * math.sqrt(0.5) * x1 * x2 * (x2.pow(2) / 12 + 0.25 * (x1 + x3).pow(2))
        t1 = P / (math.sqrt(2) * x1 * x2)
        t2 = M * R / J
        t = torch.sqrt(t1.pow(2) + t1 * t2 * x2 / R + t2.pow(2))
        s = 6 * P * L / (x4 * x3.pow(2))
        # These numbers are also different from [CoelloCoello2002constraint]_
        P_c = 64746.022 * (1 - 0.0282346 * x3) * x3 * x4.pow(3)

        g1 = (t - t_max) / t_max
        g2 = (s - s_max) / s_max
        g3 = 1 / (5 - 0.125) * (x1 - x4)
        g4 = (P - P_c) / P

        return torch.stack([g1, g2, g3, g4], dim=-1)
