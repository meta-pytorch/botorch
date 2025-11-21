# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement, TAU_MAX, TAU_RELU
from botorch.acquisition.multi_objective.base import MultiObjectiveMCAcquisitionFunction
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.model import Model
from botorch.posteriors.fully_bayesian import MCMC_DIM
from botorch.sampling.base import MCSampler
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import is_ensemble
from torch import Tensor


class qLogNParEGO(qLogNoisyExpectedImprovement, MultiObjectiveMCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        scalarization_weights: Tensor | None = None,
        sampler: MCSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        constraints: list[Callable[[Tensor], Tensor]] | None = None,
        X_pending: Tensor | None = None,
        eta: Tensor | float = 1e-3,
        fat: bool = True,
        prune_baseline: bool = False,
        cache_root: bool | None = None,
        tau_relu: float = TAU_RELU,
        tau_max: float = TAU_MAX,
        incremental: bool = True,
    ) -> None:
        r"""q-LogNParEGO supporting m >= 2 outcomes. This acquisition function
        utilizes qLogNEI to compute the expected improvement over Chebyshev
        scalarization of the objectives.

        This is adapted from qNParEGO proposed in [Daulton2020qehvi]_ to utilize
        log-improvement acquisition functions of [Ament2023logei]_. See [Knowles2005]_
        for the original ParEGO algorithm.

        This implementation assumes maximization of all objectives. If any of the model
        outputs are to be minimized, either an `objective` should be used to negate the
        model outputs or the `scalarization_weights` should be provided with negative
        weights for the outputs to be minimized.

         Args:
            model: A fitted multi-output model, producing outputs for `m` objectives
                and any number of outcome constraints.
                NOTE: The model posterior must have a `mean` attribute.
            X_baseline: A `batch_shape x r x d`-dim Tensor of `r` design points
                that have already been observed. These points are considered as
                the potential best design point.
            scalarization_weights: A `m`-dim Tensor of weights to be used in the
                Chebyshev scalarization. If omitted, samples from the unit simplex.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MultiOutputMCAcquisitionObjective under which the samples are
                evaluated before applying Chebyshev scalarization.
                Defaults to `IdentityMultiOutputObjective()`.
            constraints: A list of constraint callables which map a Tensor of posterior
                samples of dimension `sample_shape x batch-shape x q x m'`-dim to a
                `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
                are satisfied if `constraint(samples) < 0`.
            X_pending: A `batch_shape x q' x d`-dim Tensor of `q'` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated. Concatenated into `X` upon
                forward call. Copied and set to have no gradient.
            eta: Temperature parameter(s) governing the smoothness of the sigmoid
                approximation to the constraint indicators. See the docs of
                `compute_(log_)smoothed_constraint_indicator` for details.
            fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
                approximation to the ReLU.
            prune_baseline: If True, remove points in `X_baseline` that are
                highly unlikely to be the best point. This can significantly
                improve performance and is generally recommended. In order to
                customize pruning parameters, instead manually call
                `botorch.acquisition.utils.prune_inferior_points` on `X_baseline`
                before instantiating the acquisition function.
            cache_root: A boolean indicating whether to cache the root
                decomposition over `X_baseline` and use low-rank updates.
            tau_max: Temperature parameter controlling the sharpness of the smooth
                approximations to max.
            tau_relu: Temperature parameter controlling the sharpness of the smooth
                approximations to ReLU.
            incremental: Whether to compute incremental EI over the pending points
                or compute EI of the joint batch improvement (including pending
                points).
        """
        MultiObjectiveMCAcquisitionFunction.__init__(
            self,
            model=model,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
        )
        org_objective = self.objective
        # Create the composite objective.
        with torch.no_grad():
            Y_baseline = org_objective(model.posterior(X_baseline).mean)
        if is_ensemble(model):
            Y_baseline = torch.mean(Y_baseline, dim=MCMC_DIM)
        scalarization_weights = (
            scalarization_weights
            if scalarization_weights is not None
            else sample_simplex(
                d=Y_baseline.shape[-1], device=X_baseline.device, dtype=X_baseline.dtype
            ).view(-1)
        )
        chebyshev_scalarization = get_chebyshev_scalarization(
            weights=scalarization_weights,
            Y=Y_baseline,
        )
        composite_objective = GenericMCObjective(
            objective=lambda samples, X=None: chebyshev_scalarization(
                org_objective(samples=samples, X=X), X=X
            ),
        )
        qLogNoisyExpectedImprovement.__init__(
            self,
            model=model,
            X_baseline=X_baseline,
            sampler=sampler,
            # This overwrites self.objective with the composite objective.
            objective=composite_objective,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
            fat=fat,
            prune_baseline=prune_baseline,
            cache_root=cache_root,
            tau_max=tau_max,
            tau_relu=tau_relu,
            incremental=incremental,
        )
        # Set these after __init__ calls so that they're not overwritten / deleted.
        # These are intended mainly for easier debugging & transparency.
        self._org_objective: MCMultiOutputObjective = org_objective
        self.chebyshev_scalarization: Callable[[Tensor, Tensor | None], Tensor] = (
            chebyshev_scalarization
        )
        self.scalarization_weights: Tensor = scalarization_weights
        self.Y_baseline: Tensor = Y_baseline
