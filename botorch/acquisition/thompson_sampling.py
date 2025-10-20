# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.acquisition.analytic import AcquisitionFunction
from botorch.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from botorch.sampling.pathwise.posterior_samplers import get_matheron_path_model
from botorch.utils.transforms import is_ensemble, t_batch_mode_transform
from torch import Tensor


BATCH_SIZE_CHANGE_ERROR = """The batch size of PathwiseThompsonSampling should \
not change during a forward pass - was {}, now {}. Please re-initialize the \
acquisition if you want to change the batch size."""


class PathwiseThompsonSampling(AcquisitionFunction):
    r"""Single-outcome Thompson Sampling packaged as an (analytic)
    acquisition function. Querying the acquisition function gives the summed
    values of one or more draws from a pathwise drawn posterior sample, and thus
    it maximization yields one (or multiple) Thompson sample(s).

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> TS = PathwiseThompsonSampling(model)
    """

    def __init__(
        self,
        model: Model,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransform | None = None,
    ) -> None:
        r"""Single-outcome TS.

        If using a multi-output `model`, the acquisition function requires either an
        `objective` or a `posterior_transform` that transforms the multi-output
        posterior samples to single-output posterior samples.

        Args:
            model: A fitted GP model.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform.
        """

        super().__init__(model=model)
        self.batch_size: int | None = None
        self.samples: GenericDeterministicModel | None = None
        self.ensemble_indices: Tensor | None = None

        # NOTE: This conditional block is copied from MCAcquisitionFunction, we should
        # consider inherting from it and e.g. getting the X_pending logic as well.
        if objective is None and model.num_outputs != 1:
            if posterior_transform is None:
                raise UnsupportedError(
                    "Must specify an objective or a posterior transform when using "
                    "a multi-output model."
                )
            elif not posterior_transform.scalarize:
                raise UnsupportedError(
                    "If using a multi-output model without an objective, "
                    "posterior_transform must scalarize the output."
                )
        if objective is None:
            objective = IdentityMCObjective()
        self.objective = objective
        self.posterior_transform = posterior_transform

    def redraw(self, batch_size: int) -> None:
        sample_shape = (batch_size,)
        self.samples = get_matheron_path_model(
            model=self.model, sample_shape=torch.Size(sample_shape)
        )
        if is_ensemble(self.model):
            # the ensembling dimension is assumed to be part of the batch shape
            model_batch_shape = self.model.batch_shape
            if len(model_batch_shape) > 1:
                raise NotImplementedError(
                    "Ensemble models with more than one ensemble dimension are not "
                    "yet supported."
                )
            num_ensemble = model_batch_shape[0]
            # ensemble_indices is cached here to ensure that the acquisition function
            # becomes deterministic for the same input and can be optimized with LBFGS.
            # ensemble_indices is used in select_from_ensemble_models.
            self.ensemble_indices = torch.randint(
                0,
                num_ensemble,
                (*sample_shape, 1, self.model.num_outputs),
            )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the pathwise posterior sample draws on the candidate set X.

        Args:
            X: A `batch_shape x q x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `batch_shape`-dim tensor of evaluations on the posterior sample draws,
            where the samples are summed over the q-batch dimension.
        """
        objective_values = self._pathwise_forward(X)  # batch_shape x q
        # NOTE: The current implementation sums over the q-batch dimension, which means
        # that we are optimizing the sum of independent Thompson samples. In the future,
        # we can leverage *batched* L-BFGS optimization, rather than summing over the q
        # dimension, which will guarantee descent steps for all members of the batch
        # through batch-member-specific learning rate selection.
        return objective_values.sum(-1)  # batch_shape

    def _pathwise_forward(self, X: Tensor) -> Tensor:
        """Evaluate the pathwise posterior sample draws on the candidate set X.

        Args:
            X: A `batch_shape x q x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `batch_shape x q`-dim tensor of evaluations on the posterior sample draws.
        """
        batch_size = X.shape[-2]
        # batch_shape x q x 1 x d
        X = X.unsqueeze(-2)
        if self.samples is None:
            self.batch_size = batch_size
            self.redraw(batch_size=batch_size)

        if self.batch_size != batch_size:
            raise ValueError(
                BATCH_SIZE_CHANGE_ERROR.format(self.batch_size, batch_size)
            )
        # batch_shape x q [x num_ensembles] x 1 x m
        posterior_values = self.samples(X)
        # batch_shape x q [x num_ensembles] x m
        posterior_values = posterior_values.squeeze(-2)

        # batch_shape x q x m
        posterior_values = self.select_from_ensemble_models(values=posterior_values)

        if self.posterior_transform:
            posterior_values = self.posterior_transform.evaluate(posterior_values)
        # objective removes the `m` dimension
        objective_values = self.objective(posterior_values)  # batch_shape x q
        return objective_values

    def select_from_ensemble_models(self, values: Tensor):
        """Subselecting a value associated with a single sample in the ensemble for each
        element of samples that is not associated with an ensemble dimension.

        NOTE: 1) uses `self.model` and `is_ensemble` to determine whether or not an
        ensembling dimension is present. 2) uses `self.ensemble_indices` to select the
        value associated with a single sample in the ensemble. `ensemble_indices`
        contains uniformly randomly sample indices for each element of the ensemble, but
        is cached to make the evaluation of the acquisition function deterministic.

        Args:
            values: A `batch_shape x num_draws x q [x num_ensemble] x m`-dim Tensor.

        Returns:
            A`batch_shape x num_draws x q x m`-dim where each element is contains a
            single sample from the ensemble, selected with `self.ensemble_indices`.
        """
        if not is_ensemble(self.model):
            return values

        ensemble_dim = -2
        # `ensemble_indices` are fixed so that the acquisition function becomes
        # deterministic for the same input and can be optimized with LBFGS.
        # ensemble indices have shape num_paths x 1 x m
        self.ensemble_indices = self.ensemble_indices.to(device=values.device)
        index = self.ensemble_indices
        input_batch_shape = values.shape[:-3]
        index = index.expand(*input_batch_shape, *index.shape)
        # samples is batch_shape x q x num_ensemble x m
        values_wo_ensemble = torch.gather(values, dim=ensemble_dim, index=index)
        return values_wo_ensemble.squeeze(
            ensemble_dim
        )  # removing the ensemble dimension
