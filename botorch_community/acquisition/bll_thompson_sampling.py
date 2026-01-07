#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
import scipy

import torch

from botorch.logging import logger

from botorch_community.models.blls import AbstractBLLModel
from torch.func import grad


class BLLMaxPosteriorSampling:
    """TODO: Integrate into `SamplingStrategy` (note, here we dont require
    X_cand to be passed but optimize the sample paths numerically as a default)."""

    def __init__(
        self,
        model: AbstractBLLModel,
        num_restarts: int = 10,
        bounds: torch.Tensor | None = None,
        discrete_inputs: bool = False,
    ):
        """
        Implements Maximum Posterior Sampling for Bayesian Linear Last (VBLL) models.

        This class provides functionality to sample from the posterior distribution of a
        BLL model, with optional optimization to refine the sampling process.

        Args:
            model: The VBLL model from which posterior samples are drawn. Must be an
                instance of `AbstractBLLModel`.
            num_restarts: Number of restarts for optimization-based sampling.
                Defaults to 10.
            bounds: Tensor of shape `2 x d` specifying the lower and upper
                bounds for sampling. If None, defaults to [(0, 1)] for each input
                dimension.
            discrete_inputs: If True, assumes the input space is discrete and will be
                provided in __call__. Defaults to False.

        Raises:
            ValueError:
                If the provided `model` is not an instance of `AbstractBLLModel`.

        Notes:
            - If `bounds` is not provided, the default range [0,1] is assumed for each
            input dimension.
        """
        if not isinstance(model, AbstractBLLModel):
            raise ValueError(
                f"Model must be an instance of AbstractBLLModel, is {type(model)}"
            )

        self.model = model
        self.device = model.device
        self.discrete_inputs = discrete_inputs
        self.num_restarts = num_restarts

        if bounds is None:
            # Default bounds [0,1] for each input dimension
            self.bounds = [(0, 1)] * self.model.num_inputs
            self.lb = torch.zeros(
                self.model.num_inputs, dtype=torch.float64, device=torch.device("cpu")
            )
            self.ub = torch.ones(
                self.model.num_inputs, dtype=torch.float64, device=torch.device("cpu")
            )
        else:
            # Ensure bounds are on CPU for compatibility with scipy.optimize.minimize
            self.lb = bounds[0, :].cpu()
            self.ub = bounds[1, :].cpu()
            self.bounds = [tuple(bound) for bound in bounds.T.cpu().tolist()]

    def __call__(
        self, X_cand: torch.Tensor = None, num_samples: int = 1
    ) -> torch.Tensor:
        r"""Sample from a Bayesian last layer model posterior.

        Args:
            X_cand: A `N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
                NOTE: X_cand is only accepted if `discrete_inputs` is `True`!
            num_samples: The number of samples to draw.

        Returns:
            A `num_samples x d`-dim Tensor of maximum posterior values from the model,
            where `X[i, :]` is the `i`-th sample.
        """
        if self.discrete_inputs and X_cand is None:
            raise ValueError("X_cand must be provided if `discrete_inputs` is True.")

        if X_cand is not None and not self.discrete_inputs:
            raise ValueError("X_cand is provided but `discrete_inputs` is False.")

        X_next = torch.empty(
            num_samples, self.model.num_inputs, dtype=torch.float64, device=self.device
        )

        # get max of sampled functions at candidate points for each function
        for i in range(num_samples):
            f = self.model.sample()

            if self.discrete_inputs:
                # evaluate sample path at candidate points and select best
                Y_cand = f(X_cand)
            else:
                # optimize sample path
                X_cand, Y_cand = _optimize_sample_path(
                    f=f,
                    num_restarts=self.num_restarts,
                    bounds=self.bounds,
                    lb=self.lb,
                    ub=self.ub,
                    device=self.device,
                )

            # select the best candidate
            X_next[i, :] = X_cand[Y_cand.argmax()]

        # ensure that the next point is within the bounds,
        # scipy minimize can sometimes return points outside the bounds
        X_next = torch.clamp(X_next, self.lb.to(self.device), self.ub.to(self.device))
        return X_next


def _optimize_sample_path(
    f: torch.nn.Module,
    num_restarts: int,
    bounds: list[tuple[float, float]],
    lb: torch.Tensor,
    ub: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper function to optimize the sample path of a BLL network.

    Args:
        f: The sample to optimize.
        num_restarts: Number of restarts for optimization-based sampling.
        bounds: List of tuples specifying the lower and upper bounds for each input
            dimension.
        lb: Lower bounds for each input dimension.
        ub: Upper bounds for each input dimension.
        device: Device on which to store the candidate points.

    Returns:
        Candidate points and corresponding function values as a tuple with a
        `num_restarts x d`-dim Tensor and `num_restarts x num_outputs`-dim Tensor,
        respectively.
    """
    X_cand = torch.empty(num_restarts, f.num_inputs, dtype=torch.float64, device=device)
    Y_cand = torch.empty(
        num_restarts, f.num_outputs, dtype=torch.float64, device=device
    )

    # create numpy wrapper around the sampled function, note we aim to maximize
    def func(x):
        return -f(torch.from_numpy(x).to(device)).detach().cpu().numpy()

    # get gradient and create wrapper
    grad_f = grad(lambda x: f(x).mean())

    def grad_func(x):
        return -grad_f(torch.from_numpy(x).to(device)).detach().cpu().numpy()

    # generate random initial conditions
    x0s = np.random.rand(num_restarts, f.num_inputs)

    lb = lb.numpy()
    ub = ub.numpy()

    optimization_successful = False
    for j in range(num_restarts):
        # map to bounds
        x0 = lb + (ub - lb) * x0s[j]

        # optimize sample path
        res = scipy.optimize.minimize(
            func, x0, jac=grad_func, bounds=bounds, method="L-BFGS-B"
        )

        # check if optimization was successful
        if res.success:
            optimization_successful = True
        if not res.success:
            logger.warning(f"Optimization failed with message: {res.message}")

        # store the candidate
        X_cand[j, :] = torch.from_numpy(res.x).to(dtype=torch.float64)
        Y_cand[j] = torch.tensor([-res.fun], dtype=torch.float64)

    if not optimization_successful:
        raise RuntimeError("All optimization attempts on the sample path failed.")

    return X_cand, Y_cand
