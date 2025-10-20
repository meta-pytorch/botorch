#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Synthetic functions for multi-fidelity optimization benchmarks.

References:

.. [Chen2024]
    Chen, Y., et al. A latent variable approach for non-hierarchical
    multi-fidelity adaptive sampling. Computer Methods in Applied Mechanics
    and Engineering 421 (2024).
"""

from __future__ import annotations

import math

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor


class AugmentedBranin(SyntheticTestFunction):
    r"""Augmented Branin test function for multi-fidelity optimization.

    3-dimensional function with domain `[-5, 10] x [0, 15] * [0,1]`, where
    the last dimension of is the fidelity parameter:

        B(x) = (x_2 - (b - 0.1 * (1 - x_3))x_1^2 + c x_1 - r)^2 +
            10 (1-t) cos(x_1) + 10

    Here `b`, `c`, `r` and `t` are constants where `b = 5.1 / (4 * math.pi ** 2)`
    `c = 5 / math.pi`, `r = 6`, `t = 1 / (8 * math.pi)`.
    B has infinitely many minimizers with `x_1 = -pi, pi, 3pi`
    and `B_min = 0.397887`
    """

    dim = 3
    continuous_inds = list(range(dim))
    _bounds = [(-5.0, 10.0), (0.0, 15.0), (0.0, 1.0)]
    _optimal_value = 0.397887
    _optimizers = [  # this is a subset, ther are infinitely many optimizers
        (-math.pi, 12.275, 1),
        (math.pi, 1.3867356039019576, 0.1),
        (math.pi, 1.781519779945532, 0.5),
        (math.pi, 2.1763039559891064, 0.9),
    ]

    def _evaluate_true(self, X: Tensor) -> Tensor:
        t1 = (
            X[..., 1]
            - (5.1 / (4 * math.pi**2) - 0.1 * (1 - X[..., 2])) * X[..., 0].pow(2)
            + 5 / math.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[..., 0])
        return t1.pow(2) + t2 + 10


class AugmentedHartmann(SyntheticTestFunction):
    r"""Augmented Hartmann synthetic test function.

    7-dimensional function (typically evaluated on `[0, 1]^7`), where the last
    dimension is the fidelity parameter.

        H(x) = -(ALPHA_1 - 0.1 * (1-x_7)) * exp(- sum_{j=1}^6 A_1j (x_j - P_1j) ** 2) -
            sum_{i=2}^4 ALPHA_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij) ** 2)

    H has a unique global minimizer
    `x = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, 1.0]`

    with `H_min = -3.32237`
    """

    dim = 7
    continuous_inds = list(range(dim))
    _bounds = [(0.0, 1.0) for _ in range(dim)]
    _optimal_value = -3.32237
    _optimizers = [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, 1.0)]
    _check_grad_at_opt = False

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)
        self.register_buffer("ALPHA", torch.tensor([1.0, 1.2, 3.0, 3.2]))
        A = [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
        P = [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381.0],
        ]
        self.register_buffer("A", torch.tensor(A))
        self.register_buffer("P", torch.tensor(P))

    def _evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        inner_sum = torch.sum(
            self.A * (X[..., :6].unsqueeze(-2) - 0.0001 * self.P).pow(2), dim=-1
        )
        alpha1 = self.ALPHA[0] - 0.1 * (1 - X[..., 6])
        H = (
            -(torch.sum(self.ALPHA[1:] * torch.exp(-inner_sum)[..., 1:], dim=-1))
            - alpha1 * torch.exp(-inner_sum)[..., 0]
        )
        return H


class AugmentedRosenbrock(SyntheticTestFunction):
    r"""Augmented Rosenbrock synthetic test function for multi-fidelity optimization.

    d-dimensional function (usually evaluated on `[-5, 10]^(d-2) * [0, 1]^2`),
    where the last two dimensions are the fidelity parameters:

        f(x) = sum_{i=1}^{d-1} (100 (x_{i+1} - x_i^2 + 0.1 * (1-x_{d-1}))^2 +
            (x_i - 1 + 0.1 * (1 - x_d)^2)^2)

    f has one minimizer for its global minimum at `z_1 = (1, 1, ..., 1)` with
    `f(z_i) = 0.0`.
    """

    _optimal_value = 0.0

    def __init__(
        self,
        dim: int = 3,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension. Must be at least 3.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        if dim < 3:
            raise ValueError(
                "AugmentedRosenbrock must be defined it at least 3 dimensions"
            )
        self.dim = dim
        self.continuous_inds = list(range(dim))
        self._bounds = [(-5.0, 10.0) for _ in range(self.dim - 2)] + [
            (0.0, 1.0) for _ in range(2)
        ]
        self._optimizers = [tuple(1.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        X_curr = X[..., :-3]
        X_next = X[..., 1:-2]
        t1 = 100 * (X_next - X_curr.pow(2) + 0.1 * (1 - X[..., -2:-1])).pow(2)
        t2 = (X_curr - 1 + 0.1 * (1 - X[..., -1:]).pow(2)).pow(2)
        return (t1 + t2).sum(dim=-1)


class WingWeightMultiFidelity(SyntheticTestFunction):
    """Wing Weight Design Problem from [Chen2024]_.

    Design variables (physical units):
       1. s_w        in [150,    200]  (wing area)
       2. w_fw       in [220,    300]  (fuel weight)
       3. A          in [6,       10]  (aspect ratio)
       4. Lambda_deg in [-10,     10]  (sweep angle, degrees)
       5. q          in [16,      45]  (dynamic pressure)
       6. lam        in [0.5,    1.0]  (taper ratio)
       7. t_c        in [0.08,  0.18]   (thickness-to-chord)
       8. N_z        in [2.5,    6.0]  (ultimate load factor)
       9. w_dg       in [1700,  2500]  (design gross weight)
       10. w_pp      in [0.025, 0.08]  (weight per unit area)

    Fidelity parameter (stored as the 11th input):
      0: High fidelity (HF)
      1: Low fidelity 1 (LF1)
      2: Low fidelity 2 (LF2)
      3: Low fidelity 3 (LF3)

    LF models use slightly altered exponents and additive biases.
    """

    dim = 11
    continuous_inds = list(range(dim))
    _num_fidelities = 1
    _bounds = [
        (150.0, 200.0),  # s_w
        (220.0, 300.0),  # w_fw
        (6.0, 10.0),  # A
        (-10.0, 10.0),  # Lambda_deg
        (16.0, 45.0),  # q
        (0.5, 1.0),  # lam
        (0.08, 0.18),  # t_c
        (2.5, 6.0),  # N_z
        (1700.0, 2500.0),  # w_dg
        (0.025, 0.08),  # w_pp
        (0, 3),  # fidelity
    ]
    fidelities = [0, 1, 2, 3]
    _optimal_value = 123.25

    def _evaluate_true(self, X: torch.Tensor) -> Tensor:
        s_w, w_fw, A, Lambda_deg, q, lam, t_c, N_z, w_dg, w_pp, fidelity = X.unbind(
            dim=-1
        )
        Lambda_rad = Lambda_deg * math.pi / 180.0
        cos_val = torch.cos(Lambda_rad)
        y = torch.zeros_like(s_w)
        shared_multiplier = (
            0.036
            * q**0.006
            * lam**0.04
            * (A / (cos_val**2)) ** 0.6
            * (100.0 * t_c / cos_val) ** (-0.3)
            * (N_z * w_dg) ** 0.49
            * (w_fw**0.0035)
        )
        # High fidelity (fidelity == 0)
        mask = fidelity == 0
        if mask.any():
            hf = s_w**0.758 * shared_multiplier + s_w * w_pp
            y[mask] = hf[mask]
        # Low fidelity 1 (fidelity == 1)
        mask = fidelity == 1
        if mask.any():
            lf1 = s_w**0.758 * shared_multiplier + w_pp
            y[mask] = lf1[mask]
        # Low fidelity 2 (fidelity == 2)
        mask = fidelity == 2
        if mask.any():
            lf2 = s_w**0.8 * shared_multiplier + w_pp
            y[mask] = lf2[mask]
        # Low fidelity 3 (fidelity == 3)
        mask = fidelity == 3
        if mask.any():
            lf3 = s_w**0.9 * shared_multiplier
            y[mask] = lf3[mask]
        return y

    def cost(self, X: torch.Tensor) -> Tensor:
        fidelity = X[..., 10]
        c = torch.zeros_like(fidelity)
        c[fidelity == 0] = 1000.0
        c[fidelity == 1] = 100.0
        c[fidelity == 2] = 10.0
        c[fidelity == 3] = 1.0
        return c


class BoreholeMultiFidelity(SyntheticTestFunction):
    """Borehole Problem from [Chen2024]_.

    This problem models water flow through a borehole with 8 design variables:
      1. r_w   in [0.05,   0.15]   (borehole radius)
      2. r     in [100,    50000]  (radius of influence)
      3. T_u   in [63070,  115600] (transmissivity of upper aquifer)
      4. T_l   in [63.1,   116]    (transmissivity of lower aquifer)
      5. H_u   in [990,    1110]   (potentiometric head of upper aquifer)
      6. H_l   in [700,    820]    (potentiometric head of lower aquifer)
      7. L     in [1120,   1680]   (length of borehole)
      8. K_w   in [9855,   12045]  (hydraulic conductivity)

    The fidelity index (9th input) is categorical:
      0: High fidelity (HF)
      1: Low fidelity 1 (LF1)
      2: Low fidelity 2 (LF2)
      3: Low fidelity 3 (LF3)
      4: Low fidelity 4 (LF4)


    The low-fidelity models modify exponents and add a bias.
    """

    dim = 9
    continuous_inds = list(range(dim))
    _num_fidelities = 1
    _bounds = [
        (0.05, 0.15),  # r_w
        (100.0, 10000.0),  # r
        (100.0, 1000.0),  # T_u
        (10.0, 500.0),  # T_l
        (990.0, 1110.0),  # H_u
        (700.0, 820.0),  # H_l
        (1000.0, 2000.0),  # L
        (6000.0, 12000.0),  # K_w
        (0, 4),  # fidelity
    ]
    fidelities = [0, 1, 2, 3, 4]
    _optimal_value = 3.98

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r_w, r, T_u, T_l, H_u, H_l, L, K_w, fidelity = X.unbind(dim=-1)
        LTu = L * T_u
        two_pi_T_u = 2.0 * math.pi * T_u
        log_term = torch.log(r / r_w)
        denom = log_term * (r_w**2) * K_w
        numer = two_pi_T_u * (H_u - H_l)
        T_u_over_T_l = T_u / T_l
        y = torch.zeros_like(r_w)
        # HF (fidelity 0)
        mask = fidelity == 0
        if mask.any():
            hf_denom = log_term * (1.0 + (2.0 * LTu) / denom + T_u_over_T_l)
            hf = numer / hf_denom
            y[mask] = hf[mask]

        # LF1 (fidelity 1): add bias.
        mask = fidelity == 1
        if mask.any():
            lf1_numer = two_pi_T_u * (H_u - 0.8 * H_l)
            lf1_denom = log_term * (1.0 + LTu / denom + T_u_over_T_l)
            lf1 = lf1_numer / lf1_denom
            y[mask] = lf1[mask]

        # LF2 (fidelity 2): modify the exponent on log_term and add bias.
        mask = fidelity == 2
        if mask.any():
            lf2_denom = log_term * (1.0 + (8 * LTu) / denom + 0.75 * T_u_over_T_l)
            lf2 = numer / lf2_denom
            y[mask] = lf2[mask]

        # LF3 (fidelity 3): modify r_w exponent slightly.
        mask = fidelity == 3
        if mask.any():
            lf3_log_term = torch.log(4 * r / r_w)
            lf3_numer = two_pi_T_u * (1.09 * H_u - H_l)
            lf3_denom = lf3_log_term * (1.0 + (3 * LTu) / denom + T_u_over_T_l)
            lf3 = lf3_numer / lf3_denom
            y[mask] = lf3[mask]
        # LF4 (fidelity 4): further bias.
        mask = fidelity == 4
        if mask.any():
            lf4_log_term = torch.log(2 * r / r_w)
            lf4_numer = two_pi_T_u * (1.05 * H_u - H_l)
            lf4_denom = lf4_log_term * (1.0 + (3 * LTu) / denom + T_u_over_T_l)
            lf4 = lf4_numer / lf4_denom
            y[mask] = lf4[mask]

        return y

    def cost(self, X: torch.Tensor) -> torch.Tensor:
        fidelity = X[..., 8]
        c = torch.zeros_like(fidelity)
        c[fidelity == 0] = 1000.0
        c[fidelity == 1] = 100.0
        c[fidelity == 2] = 10.0
        c[fidelity == 3] = 100.0
        c[fidelity == 4] = 10.0
        return c
