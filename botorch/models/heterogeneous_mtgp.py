#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

r"""
Multi-Task GP model designed to operate on tasks from different search spaces.

References:

.. [Deshwal2024Heterogeneous]
    A. Deshwal, S. Cakmak., Y. Xia, and D. Eriksson.
    Sample-Efficient Bayesian Optimization with Transfer Learning for
    Heterogeneous Search Spaces. AutoML Conference, 2024.
"""

from itertools import chain
from typing import Any

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.kernels.heterogeneous_multitask import MultiTaskConditionalKernel
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils.gpytorch_modules import (
    get_gaussian_likelihood_with_gamma_prior,
)
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.transformed import TransformedPosterior
from botorch.utils.datasets import MultiTaskDataset
from torch import Tensor


class HeterogeneousMTGP(MultiTaskGP):
    """A multi-task GP model designed to operate on tasks from
    different search spaces. This model uses ``MultiTaskConditionalKernel``.

    This model was introduced in [Deshwal2024Heterogeneous]_.

    * The model is designed to work with a ``MultiTaskDataset`` that contains
        datasets with different features.
    * It uses a helper to embed the ``X`` coming from the sub-spaces into the
        full-feature space (+ task feature) before passing them down to the
        base ``MultiTaskGP``.
    * The same helper is used in the ``posterior`` method to embed the ``X`` from
        the target task into the full dimensional space before evaluating the
        ``posterior`` method of the base class.
    * This model also overwrites the ``_split_inputs`` method. Instead of
        ``x_basic``, we return the ``X`` with task feature included since this is
        used by the  ``MultiTaskConditionalKernel`` to identify the active
        dimensions of / the kernels to evaluate for the given input.
    """

    def __init__(
        self,
        train_Xs: list[Tensor],
        train_Ys: list[Tensor],
        train_Yvars: list[Tensor] | None,
        feature_indices: list[list[int]],
        full_feature_dim: int,
        rank: int | None = None,
        use_saas_prior: bool = True,
        use_combinatorial_kernel: bool = True,
        all_tasks: list[int] | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | None = None,
        validate_task_values: bool = True,
    ) -> None:
        """Construct a heterogeneous multi-task GP model from lists of inputs
        corresponding to each task.

        NOTE: This model assumes that the task 0 is the output / target task.
        It will only produce predictions for task 0.

        Args:
            train_Xs: A list of tensors of shape ``(n_i x d_i)`` where ``d_i`` is the
                dimensionality of the input features for task i.
                NOTE: These should not include the task feature!
            train_Ys: A list of tensors of shape ``(n_i x 1)`` containing the
                observations for the corresponding task.
            train_Yvars: An optional list of tensors of shape ``(n_i x 1)`` containing
                the observation variances for the corresponding task.
            feature_indices: A list of lists of integers specifying the indices
                mapping the features from a given task to the full tensor of features.
                The ``i``th element of the list should contain ``d_i`` integers.
            full_feature_dim: The total number of features across all tasks. This
                does not include the task feature dimension.
            rank: The rank of the cross-task covariance matrix.
            use_saas_prior: Whether to use the SAAS prior for base kernels of the
                ``MultiTaskConditionalKernel``.
            use_combinatorial_kernel: Whether to use a combinatorial kernel over the
                binary embedding of task features in ``MultiTaskConditionalKernel``.
            all_tasks: By default, multi-task GPs infer the list of all tasks from
                the task features in ``train_X``. This is an experimental feature that
                enables creation of multi-task GPs with tasks that don't appear in the
                training data. Note that when a task is not observed, the corresponding
                task covariance will heavily depend on random initialization and may
                behave unexpectedly.
            input_transform: An input transform that is applied in the model's
                forward pass. The transform should be compatible with the inputs
                from the full feature space with the task feature appended.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the ``Posterior`` obtained by calling
                ``.posterior`` on the model will be on the original scale).
            validate_task_values: If True, validate that the task values supplied in the
                input are expected tasks values. If false, unexpected task values
                will be mapped to the first output_task if supplied.
        """
        self.full_feature_dim = full_feature_dim
        self.feature_indices = feature_indices
        full_X = torch.cat(
            [self.map_to_full_tensor(X=X, task_index=i) for i, X in enumerate(train_Xs)]
        )
        full_Y = torch.cat(train_Ys)
        full_Yvar = None if train_Yvars is None else torch.cat(train_Yvars)
        covar_module = MultiTaskConditionalKernel(
            feature_indices=feature_indices,
            use_saas_prior=use_saas_prior,
            use_combinatorial_kernel=use_combinatorial_kernel,
        )
        # The features that are forward passed through the kernel should include
        # the task dim
        covar_module.active_dims = torch.arange(full_feature_dim + 1)
        likelihood = (
            None  # Constructed in MultiTaskGP.
            if full_Yvar is not None
            else get_gaussian_likelihood_with_gamma_prior()
        )
        super().__init__(
            train_X=full_X,
            train_Y=full_Y,
            task_feature=-1,
            train_Yvar=full_Yvar,
            mean_module=None,
            covar_module=covar_module,
            likelihood=likelihood,
            output_tasks=[0],
            rank=rank,
            all_tasks=all_tasks,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            validate_task_values=validate_task_values,
        )

    @classmethod
    def get_all_tasks(
        cls,
        train_X: Tensor,
        task_feature: int,
        output_tasks: list[int] | None = None,
    ) -> tuple[list[int], int, int]:
        (
            all_tasks_inferred,
            task_feature,
            num_non_task_features,
        ) = super().get_all_tasks(
            train_X=train_X, task_feature=task_feature, output_tasks=output_tasks
        )
        if 0 not in all_tasks_inferred:
            all_tasks_inferred = [0] + all_tasks_inferred
        return all_tasks_inferred, task_feature, num_non_task_features

    def map_to_full_tensor(self, X: Tensor, task_index: int) -> Tensor:
        """Map a tensor of task-specific features to the full tensor of features,
        utilizing the feature indices to map each feature to its corresponding
        position in the full tensor. Also append the task index as the last column.
        The columns of the full tensor that are not used by the given task will be
        filled with zeros.

        Args:
            X: A tensor of shape ``(n x d_i)`` where ``d_i`` is the number of features
                in the original task dataset.
            task_index: The index of the task whose features are being mapped.

        Returns:
            A tensor of shape ``(n x (self.full_feature_dim + 1))`` containing the
            mapped features.

        Example:
            >>> # Suppose full feature dim is 3 and the feature indices for
            >>> # task 5 are [2, 0].
            >>> X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            >>> X_full = self.map_to_full_tensor(X=X, task_index=5)
            >>> # X_full = torch.tensor([[2.0, 0.0, 1.0, 5.0], [4.0, 0.0, 3.0, 5.0]])
        """
        X_full = torch.zeros(
            *X.shape[:-1], self.full_feature_dim + 1, dtype=X.dtype, device=X.device
        )
        X_full[..., self.feature_indices[task_index]] = X
        X_full[..., -1] = task_index
        return X_full

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior | TransformedPosterior:
        r"""Computes the posterior for the target task at the provided points.

        Args:
            X: A tensor of shape ``batch_shape x q x d_0(+1)``, where ``d_0`` is the
                dimension of the feature space for task 0 (not including task indices)
                and ``q`` is the number of points considered jointly.
            output_indices: Not supported. Must be ``None`` or ``[0]``. The model will
                only produce predictions for the target task regardless of
                the value of ``output_indices``.
            observation_noise: If True, add observation noise from the respective
                likelihoods. If a Tensor, specifies the observation noise levels
                to add.
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A ``GPyTorchPosterior`` object, representing ``batch_shape`` joint
            distributions over ``q`` points.
        """
        if output_indices is not None and output_indices != [0]:
            raise UnsupportedError(
                "Heterogeneous MTGP does not support `output_indices`. "
            )
        if X.shape[-1] == len(self.feature_indices[0]) + 1:
            # X contains task feature
            if (X[..., -1] != 0).any():
                raise UnsupportedError(
                    "Posterior can only be called for the target task."
                )
            X = X[..., :-1]
        X_full = self.map_to_full_tensor(X=X, task_index=0)
        return super().posterior(
            X=X_full,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def _split_inputs(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"""Returns x itself along with a tensor containing the task indices only.

        NOTE: This differs from the base class implementation because it returns
        the full tensor in place of ``x_basic``. This is because the multi-task
        conditional kernel utilized the task feature for conditioning.

        Args:
            x: The full input tensor with trailing dimension of size
                ``self.full_feature_dim + 1 + 1``.

        Returns:
            3-element tuple containing
            - The original tensor ``x``.
            - A tensor of long data type containing the task indices.
            - A tensor with d=0. split_inputs by default returns X_before_index,
                task_indices, X_after_index, and so thus has to return a 3-tuple.
        """
        task_idcs = x[..., self._task_feature : self._task_feature + 1].to(
            dtype=torch.long
        )
        return x, task_idcs, torch.zeros(x.shape[:-1] + (0,)).to(x)

    @classmethod
    # pyre-ignore [14] Inconsistent override is expected.
    def construct_inputs(
        cls,
        training_data: MultiTaskDataset,
        task_feature: int = -1,
        output_tasks: list[int] | None = None,
        rank: int | None = None,
        use_saas_prior: bool = True,
        use_combinatorial_kernel: bool = True,
    ) -> dict[str, Any]:
        r"""Construct ``Model`` keyword arguments from a given ``MultiTaskDataset``.

        Args:
            training_data: A ``MultiTaskDataset``.
            task_feature: Column index of embedded task indicator features.
                Only supported value is ``-1``.
            output_tasks: A list of task indices for which to compute model
                outputs for. Only supported value is ``[0]``.
            rank: The rank of the cross-task covariance matrix.
            use_saas_prior: Whether to use the SAAS prior for base kernels of the
                ``MultiTaskConditionalKernel``.
            use_combinatorial_kernel: Whether to use a combinatorial kernel over the
                binary embedding of task features in ``MultiTaskConditionalKernel``.
        """
        if training_data.task_feature_index != -1:
            raise NotImplementedError(
                "Heterogeneous MTGP requires `task_feature_index` to be -1."
            )
        if task_feature != -1:
            raise NotImplementedError("Heterogeneous MTGP requires `task_feature=-1`.")
        if output_tasks is not None and output_tasks != [0]:
            raise NotImplementedError(
                "Heterogeneous MTGP currently only supports output_tasks=[0]. "
                "The target task will be given the task value of 0."
            )
        child_datasets = training_data.datasets.copy()
        target_dataset = child_datasets.pop(training_data.target_outcome_name)
        all_datasets = [target_dataset] + list(child_datasets.values())
        # We want all parameters to be in the same order, and include the full X.
        # remove task feature
        all_features = sorted(
            set(chain(*(ds.feature_names[:-1] for ds in all_datasets)))
        )
        # Get indices mapping the features from a given dataset to all features.
        feature_indices = [
            [all_features.index(fn) for fn in ds.feature_names[:-1]]
            for ds in all_datasets
        ]
        Xs = [ds.X[..., :-1] for ds in all_datasets]
        Ys = [ds.Y for ds in all_datasets]
        Yvars = (
            None if target_dataset.Yvar is None else [ds.Yvar for ds in all_datasets]
        )
        all_tasks = list(range(len(all_datasets)))
        return {
            "train_Xs": Xs,
            "train_Ys": Ys,
            "train_Yvars": Yvars,
            "feature_indices": feature_indices,
            "full_feature_dim": len(all_features),
            "rank": rank,
            "use_saas_prior": use_saas_prior,
            "use_combinatorial_kernel": use_combinatorial_kernel,
            "all_tasks": all_tasks,
        }
