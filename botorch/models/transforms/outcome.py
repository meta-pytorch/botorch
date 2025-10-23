#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Outcome transformations for automatically transforming and un-transforming
model outputs. Outcome transformations are typically part of a Model and
applied (i) within the model constructor to transform the train observations
to the model space, and (ii) in the `Model.posterior` call to untransform
the model posterior back to the original space.

References

.. [eriksson2021scalable]
    D. Eriksson, M. Poloczek. Scalable Constrained Bayesian Optimization.
    International Conference on Artificial Intelligence and Statistics. PMLR, 2021,
    http://proceedings.mlr.press/v130/eriksson21a.html

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from botorch.models.transforms.utils import (
    nanstd,
    norm_to_lognorm_mean,
    norm_to_lognorm_variance,
)
from botorch.models.utils.assorted import get_task_value_remapping
from botorch.posteriors import GPyTorchPosterior, Posterior, TransformedPosterior
from botorch.utils.transforms import normalize_indices
from linear_operator.operators import CholLinearOperator, DiagLinearOperator
from torch import Tensor
from torch.nn import Module, ModuleDict


class OutcomeTransform(Module, ABC):
    """Abstract base class for outcome transforms."""

    @abstractmethod
    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        r"""Transform the outcomes in a model's training targets

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).
            X: A `batch_shape x n x d`-dim tensor of training inputs (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        pass  # pragma: no cover

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        This functionality is used to properly treat outcome transformations
        in the `subset_model` functionality.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the "
            "`subset_output` method"
        )

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        r"""Un-transform previously transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of transfomred training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of transformed observation
                noises associated with the training targets (if applicable).
            X: A `batch_shape x n x d`-dim tensor of training inputs (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the `untransform` method"
        )

    @property
    def _is_linear(self) -> bool:
        """
        True for transformations such as `Standardize`; these should be able to apply
        `untransform_posterior` to a GPyTorchPosterior and return a GPyTorchPosterior,
        because a multivariate normal distribution should remain multivariate normal
        after applying the transform.
        """
        return False

    def untransform_posterior(
        self, posterior: Posterior, X: Tensor | None = None
    ) -> Posterior:
        r"""Un-transform a posterior.

        Posteriors with `_is_linear=True` should return a `GPyTorchPosterior` when
        `posterior` is a `GPyTorchPosterior`. Posteriors with `_is_linear=False`
        likely return a `TransformedPosterior` instead.

        Args:
            posterior: A posterior in the transformed space.
            X: A `batch_shape x n x d`-dim tensor of training inputs (if applicable).

        Returns:
            The un-transformed posterior.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the "
            "`untransform_posterior` method"
        )


class ChainedOutcomeTransform(OutcomeTransform, ModuleDict):
    r"""An outcome transform representing the chaining of individual transforms"""

    def __init__(self, **transforms: OutcomeTransform) -> None:
        r"""Chaining of outcome transforms.

        Args:
            transforms: The transforms to chain. Internally, the names of the
                kwargs are used as the keys for accessing the individual
                transforms on the module.
        """
        super().__init__(OrderedDict(transforms))

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        r"""Transform the outcomes in a model's training targets

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).
            X: A `batch_shape x n x d`-dim tensor of training inputs (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        for tf in self.values():
            Y, Yvar = tf.forward(Y=Y, Yvar=Yvar, X=X)
        return Y, Yvar

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        return self.__class__(
            **{name: tf.subset_output(idcs=idcs) for name, tf in self.items()}
        )

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        r"""Un-transform previously transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of transfomred training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of transformed observation
                noises associated with the training targets (if applicable).
            X: A `batch_shape x n x d`-dim tensor of training inputs (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        for tf in reversed(self.values()):
            Y, Yvar = tf.untransform(Y=Y, Yvar=Yvar, X=X)
        return Y, Yvar

    @property
    def _is_linear(self) -> bool:
        """
        A `ChainedOutcomeTransform` is linear only if all of the component transforms
        are linear.
        """
        return all(octf._is_linear for octf in self.values())

    def untransform_posterior(
        self, posterior: Posterior, X: Tensor | None = None
    ) -> Posterior:
        r"""Un-transform a posterior

        Args:
            posterior: A posterior in the transformed space.
            X: A `batch_shape x n x d`-dim tensor of training inputs (if applicable).

        Returns:
            The un-transformed posterior.
        """
        for tf in reversed(self.values()):
            posterior = tf.untransform_posterior(posterior, X=X)
        return posterior


class Standardize(OutcomeTransform):
    r"""Standardize outcomes (zero mean, unit variance).

    This module is stateful: If in train mode, calling forward updates the
    module state (i.e. the mean/std normalizing constants). If in eval mode,
    calling forward simply applies the standardization using the current module
    state.
    """

    def __init__(
        self,
        m: int,
        outputs: list[int] | None = None,
        batch_shape: torch.Size = torch.Size(),  # noqa: B008
        min_stdv: float = 1e-8,
    ) -> None:
        r"""Standardize outcomes (zero mean, unit variance).

        Args:
            m: The output dimension.
            outputs: Which of the outputs to standardize. If omitted, all
                outputs will be standardized.
            batch_shape: The batch_shape of the training targets.
            min_stddv: The minimum standard deviation for which to perform
                standardization (if lower, only de-mean the data).
        """
        super().__init__()
        self.register_buffer("means", torch.zeros(*batch_shape, 1, m))
        self.register_buffer("stdvs", torch.ones(*batch_shape, 1, m))
        self.register_buffer("_stdvs_sq", torch.ones(*batch_shape, 1, m))
        self.register_buffer("_is_trained", torch.tensor(False))
        self._outputs = normalize_indices(outputs, d=m)
        self._m = m
        self._batch_shape = batch_shape
        self._min_stdv = min_stdv

    def _get_per_input_means_stdvs(
        self, X: Tensor, include_stdvs_sq: bool
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        r"""Get per-input means and stdvs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of input parameters.
            include_stdvs_sq: Whether to include the stdvs squared.
                This parameter is not used by this method

        Returns:
            A three-tuple with the  means and stdvs:

            - The per-input means.
            - The per-input stdvs.
            - The per-input stdvs squared.
        """
        return self.means, self.stdvs, self._stdvs_sq

    def _validate_training_inputs(self, Y: Tensor, Yvar: Tensor | None = None) -> None:
        """Validate training inputs.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises.
        """
        if Y.shape[:-2] != self._batch_shape:
            raise RuntimeError(
                f"Expected Y.shape[:-2] to be {self._batch_shape}, matching "
                f"the `batch_shape` argument to `{self.__class__.__name__}`, but got "
                f"Y.shape[:-2]={Y.shape[:-2]}."
            )
        elif Y.shape[-2] < 1:
            raise ValueError(f"Can't standardize with no observations. {Y.shape=}.")
        elif Y.size(-1) != self._m:
            raise RuntimeError(
                f"Wrong output dimension. Y.size(-1) is {Y.size(-1)}; expected "
                f"{self._m}."
            )

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        r"""Standardize outcomes.

        If the module is in train mode, this updates the module state (i.e. the
        mean/std normalizing constants). If the module is in eval mode, simply
        applies the normalization using the module state.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).
            X: A `batch_shape x n x d`-dim tensor of training inputs (if applicable).
                This argument is not used by this transform, but it is used by
                its subclass, `StratifiedStandardize`.

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        if self.training:
            self._validate_training_inputs(Y=Y, Yvar=Yvar)
            if Y.shape[-2] == 1:
                stdvs = torch.ones(
                    (*Y.shape[:-2], 1, Y.shape[-1]), dtype=Y.dtype, device=Y.device
                )
            else:
                stdvs = Y.std(dim=-2, keepdim=True)
            stdvs = stdvs.where(stdvs >= self._min_stdv, torch.full_like(stdvs, 1.0))
            means = Y.mean(dim=-2, keepdim=True)
            if self._outputs is not None:
                unused = [i for i in range(self._m) if i not in self._outputs]
                means[..., unused] = 0.0
                stdvs[..., unused] = 1.0
            self.means = means
            self.stdvs = stdvs
            self._stdvs_sq = stdvs.pow(2)
            self._is_trained = torch.tensor(True)
        include_stdvs_sq = Yvar is not None
        means, stdvs, stdvs_sq = self._get_per_input_means_stdvs(
            X=X, include_stdvs_sq=include_stdvs_sq
        )
        Y_tf = (Y - means) / stdvs
        Yvar_tf = Yvar / stdvs_sq if include_stdvs_sq else None
        return Y_tf, Yvar_tf

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        new_m = len(idcs)
        if new_m > self._m:
            raise RuntimeError(
                "Trying to subset a transform have more outputs than "
                " the original transform."
            )
        nlzd_idcs = normalize_indices(idcs, d=self._m)
        new_outputs = None
        if self._outputs is not None:
            new_outputs = [i for i in self._outputs if i in nlzd_idcs]
        new_tf = self.__class__(
            m=new_m,
            outputs=new_outputs,
            batch_shape=self._batch_shape,
            min_stdv=self._min_stdv,
        )
        new_tf.means = self.means[..., nlzd_idcs]
        new_tf.stdvs = self.stdvs[..., nlzd_idcs]
        new_tf._stdvs_sq = self._stdvs_sq[..., nlzd_idcs]
        new_tf._is_trained = self._is_trained
        if not self.training:
            new_tf.eval()
        return new_tf

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        r"""Un-standardize outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of standardized targets.
            Yvar: A `batch_shape x n x m`-dim tensor of standardized observation
                noises associated with the targets (if applicable).
            X: A `batch_shape x n x d`-dim tensor of inputs (if applicable).
                This argument is not used by this transform, but it is used by
                its subclass, `StratifiedStandardize`.

        Returns:
            A two-tuple with the un-standardized outcomes:

            - The un-standardized outcome observations.
            - The un-standardized observation noise (if applicable).
        """
        if not self._is_trained:
            raise RuntimeError(
                "`Standardize` transforms must be called on outcome data "
                "(e.g. `transform(Y)`) before calling `untransform`, since "
                "means and standard deviations need to be computed."
            )
        include_stdvs_sq = Yvar is not None
        means, stdvs, stdvs_sq = self._get_per_input_means_stdvs(
            X=X, include_stdvs_sq=include_stdvs_sq
        )
        Y_utf = means + stdvs * Y
        Yvar_utf = stdvs_sq * Yvar if include_stdvs_sq else None
        return Y_utf, Yvar_utf

    @property
    def _is_linear(self) -> bool:
        return True

    def untransform_posterior(
        self, posterior: Posterior, X: Tensor | None = None
    ) -> GPyTorchPosterior | TransformedPosterior:
        r"""Un-standardize the posterior.

        Args:
            posterior: A posterior in the standardized space.
            X: A `batch_shape x n x d`-dim tensor of inputs (if applicable).
                This argument is not used by this transform, but it is used by
                its subclass, `StratifiedStandardize`.

        Returns:
            The un-standardized posterior. If the input posterior is a
            `GPyTorchPosterior`, return a `GPyTorchPosterior`. Otherwise, return a
            `TransformedPosterior`.
        """
        if self._outputs is not None:
            raise NotImplementedError(
                "Standardize does not yet support output selection for "
                "untransform_posterior"
            )
        if not self._is_trained:
            raise RuntimeError(
                "`Standardize` transforms must be called on outcome data "
                "(e.g. `transform(Y)`) before calling `untransform_posterior`, since "
                "means and standard deviations need to be computed."
            )
        is_mtgp_posterior = False
        if type(posterior) is GPyTorchPosterior:
            is_mtgp_posterior = posterior._is_mt
        if not self._m == posterior._extended_shape()[-1] and not is_mtgp_posterior:
            raise RuntimeError(
                "Incompatible output dimensions encountered. Transform has output "
                f"dimension {self._m} and posterior has "
                f"{posterior._extended_shape()[-1]}."
            )

        if type(posterior) is not GPyTorchPosterior:
            # fall back to TransformedPosterior
            # this applies to subclasses of GPyTorchPosterior like MultitaskGPPosterior
            return TransformedPosterior(
                posterior=posterior,
                sample_transform=lambda s: self.means + self.stdvs * s,
                mean_transform=lambda m, v: self.means + self.stdvs * m,
                variance_transform=lambda m, v: self._stdvs_sq * v,
            )
        # GPyTorchPosterior (TODO: Should we Lazy-evaluate the mean here as well?)
        mvn = posterior.distribution
        offset, scale_fac, _ = self._get_per_input_means_stdvs(
            X=X, include_stdvs_sq=False
        )
        if not posterior._is_mt:
            mean_tf = offset.squeeze(-1) + scale_fac.squeeze(-1) * mvn.mean
            scale_fac = scale_fac.squeeze(-1).expand_as(mean_tf)
        else:
            mean_tf = offset + scale_fac * mvn.mean
            reps = mean_tf.shape[-2:].numel() // scale_fac.size(-1)
            scale_fac = scale_fac.squeeze(-2)
            if mvn._interleaved:
                scale_fac = scale_fac.repeat(*[1 for _ in scale_fac.shape[:-1]], reps)
            else:
                scale_fac = torch.repeat_interleave(scale_fac, reps, dim=-1)

        if (
            not mvn.islazy
            or mvn._MultivariateNormal__unbroadcasted_scale_tril is not None
        ):
            # if already computed, we can save a lot of time using scale_tril
            covar_tf = CholLinearOperator(mvn.scale_tril * scale_fac.unsqueeze(-1))
        else:
            lcv = mvn.lazy_covariance_matrix
            scale_fac = scale_fac.expand(lcv.shape[:-1])
            scale_mat = DiagLinearOperator(scale_fac)
            covar_tf = scale_mat @ lcv @ scale_mat

        kwargs = {"interleaved": mvn._interleaved} if posterior._is_mt else {}
        mvn_tf = mvn.__class__(mean=mean_tf, covariance_matrix=covar_tf, **kwargs)
        return GPyTorchPosterior(mvn_tf)


class StratifiedStandardize(Standardize):
    r"""Standardize outcomes (zero mean, unit variance) along stratification dimension.

    This module is stateful: If in train mode, calling forward updates the
    module state (i.e. the mean/std normalizing constants). If in eval mode,
    calling forward simply applies the standardization using the current module
    state.
    """

    def __init__(
        self,
        stratification_idx: int,
        observed_task_values: Tensor,
        all_task_values: Tensor,
        batch_shape: torch.Size = torch.Size(),  # noqa: B008
        min_stdv: float = 1e-8,
        dtype: torch.dtype = torch.double,
        default_task_value: int | None = None,
    ) -> None:
        r"""Standardize outcomes (zero mean, unit variance) along stratification dim.

        Note: This currenlty only supports single output models
        (including multi-task models that have a single output).

        Args:
            stratification_idx: The index of the stratification dimension in the
                input tensor X.
            observed_task_values: `t`-dim tensor of task values that were actually
                observed in the training data.
            all_task_values: `t`-dim tensor of all possible task values that could
                appear in the dataset.
            batch_shape: The batch_shape of the training targets.
            min_stdv: The minimum standard deviation for which to perform
                standardization (if lower, only de-mean the data).
            dtype: The data type for internal computations.
            default_task_value: The default task value that unexpected tasks are
                mapped to. This is used in `get_task_value_remapping`.
        """
        OutcomeTransform.__init__(self)
        self._stratification_idx = stratification_idx
        observed_task_values = observed_task_values.unique(sorted=True)
        self.strata_mapping = get_task_value_remapping(
            observed_task_values=observed_task_values,
            all_task_values=all_task_values.unique(sorted=True),
            dtype=dtype,
            default_task_value=default_task_value,
        )
        if self.strata_mapping is None:
            self.strata_mapping = observed_task_values
        n_strata = self.strata_mapping.shape[0]
        self._min_stdv = min_stdv
        self.register_buffer("means", torch.zeros(*batch_shape, n_strata, 1))
        self.register_buffer("stdvs", torch.ones(*batch_shape, n_strata, 1))
        self.register_buffer("_stdvs_sq", torch.ones(*batch_shape, n_strata, 1))
        self.register_buffer("_is_trained", torch.tensor(False))
        self._batch_shape = batch_shape
        self._m = 1  # TODO: support multiple outputs
        self._outputs = None

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        r"""Standardize outcomes.

        If the module is in train mode, this updates the module state (i.e. the
        mean/std normalizing constants). If the module is in eval mode, simply
        applies the normalization using the module state.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).
            X: A `batch_shape x n x d`-dim tensor of input parameters.

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        if X is None:
            raise ValueError("X is required for StratifiedStandardize.")
        if self.training:
            self._validate_training_inputs(Y=Y, Yvar=Yvar)
            self.means = self.means.to(dtype=X.dtype, device=X.device)
            self.stdvs = self.stdvs.to(dtype=X.dtype, device=X.device)
            self._stdvs_sq = self._stdvs_sq.to(dtype=X.dtype, device=X.device)
            strata = X[..., self._stratification_idx].long()
            unique_strata = strata.unique()
            for s in unique_strata:
                mapped_strata = self.strata_mapping[s].long()
                mask = strata != s
                Y_strata = Y.clone()
                Y_strata[..., mask, :] = float("nan")
                stdvs = (
                    torch.ones_like(Y_strata)
                    if Y.shape[-2] == 1
                    else nanstd(X=Y_strata, dim=-2)
                )
                stdvs = stdvs.where(
                    stdvs >= self._min_stdv, torch.full_like(stdvs, 1.0)
                )
                means = Y_strata.nanmean(dim=-2)
                self.means[..., mapped_strata, :] = means
                self.stdvs[..., mapped_strata, :] = stdvs
                self._stdvs_sq[..., mapped_strata, :] = stdvs.pow(2)
            self._is_trained = torch.tensor(True)
        training = self.training
        self.training = False
        tf_Y, tf_Yvar = super().forward(Y=Y, Yvar=Yvar, X=X)
        self.training = training
        return tf_Y, tf_Yvar

    def _get_per_input_means_stdvs(
        self, X: Tensor, include_stdvs_sq: bool
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        r"""Get per-input means and stdvs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of input parameters.
            include_stdvs_sq: Whether to include the stdvs squared.

        Returns:
            A three-tuple with the per-input means and stdvs:

            - The per-input means.
            - The per-input stdvs.
            - The per-input stdvs squared.
        """
        strata = X[..., self._stratification_idx].long()
        mapped_strata = self.strata_mapping[strata].unsqueeze(-1).long()
        # get means and stdvs for each strata
        n_extra_batch_dims = mapped_strata.ndim - 2 - len(self._batch_shape)
        expand_shape = mapped_strata.shape[:n_extra_batch_dims] + self.means.shape
        means = torch.gather(
            input=self.means.expand(expand_shape),
            dim=-2,
            index=mapped_strata,
        )
        stdvs = torch.gather(
            input=self.stdvs.expand(expand_shape),
            dim=-2,
            index=mapped_strata,
        )
        if include_stdvs_sq:
            stdvs_sq = torch.gather(
                input=self._stdvs_sq.expand(expand_shape),
                dim=-2,
                index=mapped_strata,
            )
        else:
            stdvs_sq = None
        return means, stdvs, stdvs_sq

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        raise NotImplementedError

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        r"""Un-standardize outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of standardized targets.
            Yvar: A `batch_shape x n x m`-dim tensor of standardized observation
                noises associated with the targets (if applicable).
            X: A `batch_shape x n x d`-dim tensor of input parameters.

        Returns:
            A two-tuple with the un-standardized outcomes:

            - The un-standardized outcome observations.
            - The un-standardized observation noise (if applicable).
        """
        if X is None:
            raise ValueError("X is required for StratifiedStandardize.")
        return super().untransform(Y=Y, Yvar=Yvar, X=X)

    def untransform_posterior(
        self, posterior: Posterior, X: Tensor | None = None
    ) -> GPyTorchPosterior | TransformedPosterior:
        r"""Un-standardize the posterior.

        Args:
            posterior: A posterior in the standardized space.
            X: A `batch_shape x n x d`-dim tensor of training inputs (if applicable).

        Returns:
            The un-standardized posterior. If the input posterior is a
            `GPyTorchPosterior`, return a `GPyTorchPosterior`. Otherwise, return a
            `TransformedPosterior`.
        """
        if X is None:
            raise ValueError("X is required for StratifiedStandardize.")
        return super().untransform_posterior(posterior=posterior, X=X)


class Log(OutcomeTransform):
    r"""Log-transform outcomes.

    Useful if the targets are modeled using a (multivariate) log-Normal
    distribution. This means that we can use a standard GP model on the
    log-transformed outcomes and un-transform the model posterior of that GP.
    """

    def __init__(self, outputs: list[int] | None = None) -> None:
        r"""Log-transform outcomes.

        Args:
            outputs: Which of the outputs to log-transform. If omitted, all
                outputs will be standardized.
        """
        super().__init__()
        self._outputs = outputs

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        new_outputs = None
        if self._outputs is not None:
            if min(self._outputs + idcs) < 0:
                raise NotImplementedError(
                    f"Negative indexing not supported for {self.__class__.__name__} "
                    "when subsetting outputs and only transforming some outputs."
                )
            new_outputs = [i for i in self._outputs if i in idcs]
        new_tf = self.__class__(outputs=new_outputs)
        if not self.training:
            new_tf.eval()
        return new_tf

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        r"""Log-transform outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).
            X: A `batch_shape x n x d`-dim tensor of training inputs (if applicable).
                This argument is not used by this transform.

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        Y_tf = torch.log(Y)
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_tf = torch.stack(
                [
                    Y_tf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        if Yvar is not None:
            # TODO: Delta method, possibly issue warning
            raise NotImplementedError(
                "Log does not yet support transforming observation noise"
            )
        return Y_tf, Yvar

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        r"""Un-transform log-transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of log-transfomred targets.
            Yvar: A `batch_shape x n x m`-dim tensor of log- transformed
                observation noises associated with the training targets
                (if applicable).
            X: A `batch_shape x n x d`-dim tensor of inputs (if applicable).
                This argument is not used by this transform.

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The exponentiated outcome observations.
            - The exponentiated observation noise (if applicable).
        """
        Y_utf = torch.exp(Y)
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_utf = torch.stack(
                [
                    Y_utf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        if Yvar is not None:
            # TODO: Delta method, possibly issue warning
            raise NotImplementedError(
                "Log does not yet support transforming observation noise"
            )
        return Y_utf, Yvar

    def untransform_posterior(
        self, posterior: Posterior, X: Tensor | None = None
    ) -> TransformedPosterior:
        r"""Un-transform the log-transformed posterior.

        Args:
            posterior: A posterior in the log-transformed space.
            X: A `batch_shape x n x d`-dim tensor of inputs (if applicable).
                This argument is not used by this transform.

        Returns:
            The un-transformed posterior.
        """
        if self._outputs is not None:
            raise NotImplementedError(
                "Log does not yet support output selection for untransform_posterior"
            )
        return TransformedPosterior(
            posterior=posterior,
            sample_transform=torch.exp,
            mean_transform=norm_to_lognorm_mean,
            variance_transform=norm_to_lognorm_variance,
        )


class Power(OutcomeTransform):
    r"""Power-transform outcomes.

    Useful if the targets are modeled using a (multivariate) power transform of
    a Normal distribution. This means that we can use a standard GP model on the
    power-transformed outcomes and un-transform the model posterior of that GP.
    """

    def __init__(self, power: float, outputs: list[int] | None = None) -> None:
        r"""Power-transform outcomes.

        Args:
            outputs: Which of the outputs to power-transform. If omitted, all
                outputs will be standardized.
        """
        super().__init__()
        self._outputs = outputs
        self.power = power

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        new_outputs = None
        if self._outputs is not None:
            if min(self._outputs + idcs) < 0:
                raise NotImplementedError(
                    f"Negative indexing not supported for {self.__class__.__name__} "
                    "when subsetting outputs and only transforming some outputs."
                )
            new_outputs = [i for i in self._outputs if i in idcs]
        new_tf = self.__class__(power=self.power, outputs=new_outputs)
        if not self.training:
            new_tf.eval()
        return new_tf

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        r"""Power-transform outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).
            X: A `batch_shape x n x d`-dim tensor of training inputs (if applicable).
                This argument is not used by this transform.

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        Y_tf = Y.pow(self.power)
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_tf = torch.stack(
                [
                    Y_tf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        if Yvar is not None:
            # TODO: Delta method, possibly issue warning
            raise NotImplementedError(
                "Power does not yet support transforming observation noise"
            )
        return Y_tf, Yvar

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        r"""Un-transform power-transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of power-transfomred targets.
            Yvar: A `batch_shape x n x m`-dim tensor of power-transformed
                observation noises associated with the training targets
                (if applicable).
            X: A `batch_shape x n x d`-dim tensor of inputs (if applicable).
                This argument is not used by this transform.

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The un-power transformed outcome observations.
            - The un-power transformed observation noise (if applicable).
        """
        Y_utf = Y.pow(1.0 / self.power)
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_utf = torch.stack(
                [
                    Y_utf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        if Yvar is not None:
            # TODO: Delta method, possibly issue warning
            raise NotImplementedError(
                "Power does not yet support transforming observation noise"
            )
        return Y_utf, Yvar

    def untransform_posterior(
        self, posterior: Posterior, X: Tensor | None = None
    ) -> TransformedPosterior:
        r"""Un-transform the power-transformed posterior.

        Args:
            posterior: A posterior in the power-transformed space.
            X: A `batch_shape x n x d`-dim tensor of inputs (if applicable).
                This argument is not used by this transform.

        Returns:
            The un-transformed posterior.
        """
        if self._outputs is not None:
            raise NotImplementedError(
                "Power does not yet support output selection for untransform_posterior"
            )
        return TransformedPosterior(
            posterior=posterior,
            sample_transform=lambda x: x.pow(1.0 / self.power),
        )


class Bilog(OutcomeTransform):
    r"""Bilog-transform outcomes.

    The Bilog transform [eriksson2021scalable]_ is useful for modeling outcome
    constraints as it magnifies values near zero and flattens extreme values.
    """

    def __init__(self, outputs: list[int] | None = None) -> None:
        r"""Bilog-transform outcomes.

        Args:
            outputs: Which of the outputs to Bilog-transform. If omitted, all
                outputs will be transformed.
        """
        super().__init__()
        self._outputs = outputs

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        new_outputs = None
        if self._outputs is not None:
            if min(self._outputs + idcs) < 0:
                raise NotImplementedError(
                    f"Negative indexing not supported for {self.__class__.__name__} "
                    "when subsetting outputs and only transforming some outputs."
                )
            new_outputs = [i for i in self._outputs if i in idcs]
        new_tf = self.__class__(outputs=new_outputs)
        if not self.training:
            new_tf.eval()
        return new_tf

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        r"""Bilog-transform outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).
            X: A `batch_shape x n x d`-dim tensor of training inputs (if applicable).
                This argument is not used by this transform.

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        Y_tf = Y.sign() * (Y.abs() + 1.0).log()
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_tf = torch.stack(
                [
                    Y_tf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        if Yvar is not None:
            raise NotImplementedError(
                "Bilog does not yet support transforming observation noise"
            )
        return Y_tf, Yvar

    def untransform(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        r"""Un-transform bilog-transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of bilog-transfomred targets.
            Yvar: A `batch_shape x n x m`-dim tensor of bilog-transformed
                observation noises associated with the training targets
                (if applicable).
            X: A `batch_shape x n x d`-dim tensor of inputs (if applicable).
                This argument is not used by this transform.

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        Y_utf = Y.sign() * Y.abs().expm1()
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_utf = torch.stack(
                [
                    Y_utf[..., i] if i in outputs else Y[..., i]
                    for i in range(Y.size(-1))
                ],
                dim=-1,
            )
        if Yvar is not None:
            # TODO: Delta method, possibly issue warning
            raise NotImplementedError(
                "Bilog does not yet support transforming observation noise"
            )
        return Y_utf, Yvar

    def untransform_posterior(
        self, posterior: Posterior, X: Tensor | None = None
    ) -> TransformedPosterior:
        r"""Un-transform the bilog-transformed posterior.

        Args:
            posterior: A posterior in the bilog-transformed space.
            X: A `batch_shape x n x d`-dim tensor of inputs (if applicable).
                This argument is not used by this transform.

        Returns:
            The un-transformed posterior.
        """
        if self._outputs is not None:
            raise NotImplementedError(
                "Bilog does not yet support output selection for untransform_posterior"
            )
        return TransformedPosterior(
            posterior=posterior,
            sample_transform=lambda x: x.sign() * x.abs().expm1(),
        )
