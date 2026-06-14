# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""AddTree GP regression model.

A :class:`~botorch.models.gp_regression.SingleTaskGP` whose covariance
is the AddTree kernel built from an
:class:`~botorch_community.models.addtree.space.AddTreeSpace`.

Contributor: maxc01
"""

from __future__ import annotations

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform

from botorch_community.models.addtree.kernels import build_addtree_kernel
from botorch_community.models.addtree.space import AddTreeSpace
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from torch import Tensor


__all__ = ["AddTreeGP"]


class AddTreeGP(SingleTaskGP):
    r"""SingleTaskGP whose covariance is the AddTree kernel for a given space.

    Inputs are BFS-encoded vectors produced by
    :func:`~botorch_community.models.addtree.encoding.encode` (one
    function call per training point); see also
    :func:`~botorch_community.acquisition.addtree.optimize_addtree_acqf`
    which produces them automatically during optimisation.

    Sign convention:
        BoTorch maximises acquisition functions; the original AddTree
        algorithm minimises the objective. Negate the observed objective
        values before passing them as ``train_Y`` (i.e. ``train_Y =
        -y_obs``) and use a maximisation acquisition such as
        :class:`~botorch.acquisition.UpperConfidenceBound`. The helper
        :func:`~botorch_community.acquisition.addtree.optimize_addtree_acqf`
        handles this convention by default.
    """

    def __init__(
        self,
        space: AddTreeSpace,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        r"""Initialise an AddTreeGP.

        Args:
            space: The :class:`AddTreeSpace` describing the conditional
                parameter space.
            train_X: ``batch_shape x n x space.dim`` BFS-encoded inputs.
            train_Y: ``batch_shape x n x m`` observations (negated if
                minimising; see the class docstring).
            train_Yvar: Optional observation noise variances.
            likelihood: Optional GPyTorch likelihood.
            mean_module: Optional mean module (default ``ConstantMean``).
            outcome_transform: Optional outcome transform. ``None`` keeps
                the BoTorch default (``Standardize``).
            input_transform: Optional input transform. Usually ``None``
                because the AddTree encoding is already structured.
        """
        if not isinstance(space, AddTreeSpace):
            raise TypeError(
                f"AddTreeGP expects an AddTreeSpace; got {type(space).__name__}."
            )
        if train_X.shape[-1] != space.dim:
            raise ValueError(
                f"train_X has last dim {train_X.shape[-1]} but "
                f"space.dim={space.dim}."
            )

        kwargs: dict = {}
        if likelihood is not None:
            kwargs["likelihood"] = likelihood
        if mean_module is not None:
            kwargs["mean_module"] = mean_module
        if outcome_transform is not None:
            kwargs["outcome_transform"] = outcome_transform
        if input_transform is not None:
            kwargs["input_transform"] = input_transform

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=build_addtree_kernel(space),
            **kwargs,
        )
        self._addtree_space = space

    @property
    def addtree_space(self) -> AddTreeSpace:
        """The :class:`AddTreeSpace` used to build this model."""
        return self._addtree_space
