#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from botorch.models import SingleTaskGP
from botorch.models.hierarchical.utils import get_blocks_with_paths
from botorch.models.map_saas import add_saas_prior
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.utils.constraints import LogTransformedInterval
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.priors.prior import Prior
from torch import Tensor
from torch.nn import ModuleList


LOG_OUTPUTSCALE_CONSTRAINT = LogTransformedInterval(1e-2, 1e4, initial_value=10)

RTOL = 1e-5
ATOL = 1e-8


def _row_equal(
    tensor: Tensor,
    target: Tensor,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    """
    Check if each row of ``tensor`` is approximately equal to ``target``.

    Uses ``torch.isclose`` with configurable tolerances to handle both integer
    and float comparisons. For integer hierarchical parameters, the small
    tolerance ensures near-exact matching while handling minor numerical
    precision issues. For float hierarchical parameters, the tolerance defines
    the minimum separation required.

    Args:
        tensor: A tensor whose shape is ``(..., n, d)``. It's typically in
            double precision.
        target: A tensor whose shape is ``(d,)``. Can be integer or float dtype.
        rtol: Relative tolerance for comparisons.
        atol: Absolute tolerance for comparisons.

    Returns:
        A tensor whose shape is ``(..., n)`` indicating if each row of the
        input ``tensor`` is approximately equal to ``target``.
    """
    target = target.to(tensor)
    return torch.isclose(tensor, target, rtol=rtol, atol=atol).all(dim=-1)


class HierarchicalConditionalKernel(Kernel):
    """
    A conditional kernel that exploits correlations in hierarchical search spaces.

    It is best to describe its behavior by walking through an example. Let's say the
    hierarchical search space tree is as follows::

        ROOT
        ├── C1
        ├── C2
        └── P1
            ├── (0) C3
            └── (1) P2
                    ├── (0) C4
                    └── (1) C5

    Let's say the input X is a vector of the form ``(C1, C2, C3, C4, C5, P1, P2)``.
    The entries do not necessarily need to follow this particular order, though.

    As a concrete example, this kernel computes::

        # Ex1:
        k([C1, C2, C3, C4, C5, P1=0, P2=0], [C1', C2', C3', C4', C5', P1'=1, P2'=1]) =
            k([C1, C2], [C1', C2']) + k(P1, P1').
        # Ex2:
        k([C1, C2, C3, C4, C5, P1=1, P2=0], [C1', C2', C3', C4', C5', P1'=1, P2'=1]) =
            k([C1, C2], [C1', C2']) + k(P1, P1') + k(P2, P2').
        # Ex3:
        k([C1, C2, C3, C4, C5, P1=1, P2=1], [C1', C2', C3', C4', C5', P1'=1, P2'=1]) =
            k([C1, C2], [C1', C2']) + k(C5, C5') + k(P1, P1') + k(P2, P2').

    More generally, the kernel finds all common active features and sums the kernel
    values over them.

    1. This kernel supports arbitrary tree depths.
    2. Each parent node is allowed to have multiple child nodes.
    3. In particular, single-child parents are allowed. In this case, the parent is
       a boolean flag.
    4. Dimensions that correspond to parent nodes must be discrete or categorical.
       Approximate equality is used to compare them for branching logic, which
       requires the values to be separated by at least a tolerance of
       ``rtol=RTOL``, ``atol=ATOL``.

    Internally, this kernel determines if a child node is active by checking the
    values of its ancestors (not just its parent). Examples:

    1. C3 is active if and only if ``P1 == 0``;
    2. C4 is active if and only if ``P1 == 1`` and ``P2 == 0``.
    """

    def __init__(
        self,
        dim: int,
        hierarchical_dependencies: dict[int, dict[int | float, list[int]]],
        eval_hierarchical_features: bool = True,
        separate_hierarchical_features: bool = True,
        use_saas_prior: bool = True,
        use_outputscale: bool = True,
    ):
        """
        Args:
            dim: The dimension of the feature vector.
            hierarchical_dependencies: A dictionary of the form
                ``{parent_index: {parent_value: children_indices}}``.
            eval_hierarchical_features: Whether to evaluate correlations over
                hierarchical features or not. If false, the hierarchical features
                are merely used as flags to determine the active features, but
                they do not directly contribute to the kernel values.
            separate_hierarchical_features: This is relevant only if
                ``eval_hierarchical_features=True``. If true, the correlations of
                hierarchical features will be captured by a separate additive
                kernel. Otherwise, they are treated together with
                non-hierarchical features.
            use_saas_prior: Whether to use the SAAS prior. If false, use the log-normal
                prior instead.
            use_outputscale: Whether to use the outputscale parameter. If false, the
                outputscales of each sub-kernels are fixed to 1.
        """
        super().__init__()

        # Check that parent values are not too close to each other to avoid issues
        # with _row_equal checks used in branching.
        all_parent_values = sorted(
            {
                p
                for dependencies in hierarchical_dependencies.values()
                for p in dependencies.keys()
            }
        )
        for i in range(len(all_parent_values) - 1):
            val1, val2 = all_parent_values[i : i + 2]
            # This is the same check as torch.isclose.
            if val2 - val1 <= ATOL + RTOL * max(abs(val1), abs(val2)):
                raise ValueError(
                    f"Float parent values {val1} and {val2} are closer than the "
                    f"default tolerance (rtol={RTOL}, atol={ATOL}). This will cause "
                    "ambiguity in hierarchical branching. "
                )

        self.hierarchical_dependencies = hierarchical_dependencies
        self.use_saas_prior = use_saas_prior
        self.use_outputscale = use_outputscale
        self.partition, self.paths = get_blocks_with_paths(
            dim=dim,
            hierarchical_dependencies=hierarchical_dependencies,
            keep_hierarchical_features=eval_hierarchical_features,
            separate_hierarchical_features=separate_hierarchical_features,
        )

        # Extract two data structures from `self.path`:
        # 1. The ancestors of each block in `self.partition`;
        # 2. The target ancestor values for each block.
        # A block is active if and only if all of its ancestors have values equal to the
        # target values specified in `ancester_values`.
        self.ancestor_indices = [[index for index, _ in path] for path in self.paths]
        self.ancestor_values = [
            torch.tensor([value for _, value in path], dtype=torch.float64)
            for path in self.paths
        ]
        self.kernels = self.construct_individual_kernels()

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        if diag:
            # Not doing any validation here. x1 & x2 will be validated in the
            # sub-kernel calls anyway.
            res = torch.zeros(*x1.shape[:-1], dtype=x1.dtype, device=x1.device)
        else:
            res = torch.zeros(
                *x1.shape[:-1], x2.shape[-2], dtype=x1.dtype, device=x1.device
            )

        for kernel, indices, values in zip(
            self.kernels,
            self.ancestor_indices,
            self.ancestor_values,
        ):
            # No need for indexing here. It has already been taken care of by
            # `active_dims` of `kernel`.
            mat = kernel(x1, x2, diag=diag, **params)

            is_active_x1 = _row_equal(x1[..., indices], values)
            is_active_x2 = _row_equal(x2[..., indices], values)
            if diag:
                delta = is_active_x1.logical_and(is_active_x2)
            else:
                delta = is_active_x1.unsqueeze(-1) * is_active_x2.unsqueeze(-2)

            res = res + delta * mat

        return res

    def construct_individual_kernels(self) -> ModuleList:
        covar_modules = ModuleList()

        for indices in self.partition:
            if self.use_saas_prior:
                base_kernel = MaternKernel(
                    nu=2.5, ard_num_dims=len(indices), active_dims=indices
                )
                add_saas_prior(base_kernel)

            else:
                base_kernel = get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=len(indices), use_rbf_kernel=False, active_dims=indices
                )
                base_kernel.lengthscale = 1.0

            if self.use_outputscale:
                covar_modules.append(
                    ScaleKernel(
                        base_kernel,
                        outputscale_constraint=LOG_OUTPUTSCALE_CONSTRAINT,
                    )
                )
            else:
                covar_modules.append(base_kernel)

        return covar_modules


def _transform_hierarchical_dependencies(
    train_X: Tensor,
    hierarchical_dependencies: dict[int, dict[int | float, list[int]]],
    input_transform: InputTransform | None,
) -> dict[int, dict[int | float, list[int]]]:
    """Transform hierarchical_dependencies values according to the input transform.

    The hierarchical_dependencies dict contains parent indices and their values
    in the original input space. When an input transform is applied, we need to
    transform these values to match the transformed inputs that the kernel sees.

    Args:
        train_X: Training inputs of shape ``n x d``, used as a template for
            creating dummy X tensors.
        hierarchical_dependencies: A dictionary of the form
            ``{parent_index: {parent_value: children_indices}}``.
        input_transform: The input transform to apply.

    Returns:
        A new hierarchical_dependencies dict with transformed parent values.
    """
    if input_transform is None:
        return hierarchical_dependencies

    # Collect all (parent_index, parent_value) pairs.
    parent_value_pairs = [
        (parent_idx, parent_value)
        for parent_idx, value_to_children in hierarchical_dependencies.items()
        for parent_value in value_to_children.keys()
    ]

    if not parent_value_pairs:  # pragma: no cover
        raise ValueError(
            "Something went wrong. No parent values could be extracted from "
            f"{hierarchical_dependencies=}."
        )

    # Create dummy X tensors with the parent values in the appropriate columns.
    n_pairs = len(parent_value_pairs)
    dummy_X = train_X[0:1].repeat(n_pairs, 1)
    row_indices = torch.arange(n_pairs)
    col_indices = torch.tensor([p[0] for p in parent_value_pairs])
    values = torch.tensor([p[1] for p in parent_value_pairs], dtype=dummy_X.dtype)
    dummy_X[row_indices, col_indices] = values

    # Apply the input transform to get the transformed values.
    # Use preprocess_transform to ensure the transform is applied in the same way
    # as it is applied to model training data, without modifying buffers.
    transformed_X = input_transform.preprocess_transform(dummy_X)

    # Extract transformed parent values.
    transformed_values = transformed_X[row_indices, col_indices].tolist()

    # Build the new dict.
    result = {}
    for i, (parent_idx, original_value) in enumerate(parent_value_pairs):
        result.setdefault(parent_idx, {})
        # Replace the parent value with the transformed value, mapping to the
        # same dependent indices as before.
        result[parent_idx][transformed_values[i]] = hierarchical_dependencies[
            parent_idx
        ][original_value]
    return result


class HierarchicalConditionalKernelGP(SingleTaskGP):
    r"""
    A GP model with a conditional kernel that exploits correlations in hierarchical
    search spaces.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        hierarchical_dependencies: dict[int, dict[int | float, list[int]]],
        eval_hierarchical_features: bool = True,
        separate_hierarchical_features: bool = True,
        train_Yvar: Tensor | None = None,
        use_saas_prior: bool = True,
        use_outputscale: bool = True,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
    ) -> None:
        r"""Single-Task GP model using a hierarchical conditional kernel.

        Args:
            train_X: A ``batch_shape x n x d`` tensor of training features, where
                ``d`` is the number of features in the search space. Column
                indices in ``hierarchical_dependencies`` refer to columns of
                ``train_X``.
            train_Y: A ``batch_shape x n x m`` tensor of training observations.
            hierarchical_dependencies: A dictionary of the form
                ``{parent_index: {parent_value: children_indices}}`` that defines
                the hierarchical structure of the search space. All indices must
                be valid column indices into ``train_X`` (i.e., in ``[0, d)``).
            eval_hierarchical_features: Whether to evaluate correlations over
                hierarchical features or not. If false, the hierarchical features
                are merely used as flags to determine the active features, but
                they do not directly contribute to the kernel values.
            separate_hierarchical_features: This is relevant only if
                ``eval_hierarchical_features=True``. If true, the correlations of
                hierarchical features will be captured by a separate additive
                kernel.
            train_Yvar: An optional ``batch_shape x n x m`` tensor of observed
                measurement noise. If None, the noise is inferred.
            use_saas_prior: Whether to use the SAAS prior. If false, use the
                log-normal prior instead.
            use_outputscale: Whether to use the outputscale parameter. If false,
                the outputscales of each sub-kernel are fixed to 1.
            input_transform: An input transform that is applied in the model's
                forward pass.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference.
        """
        # Transform hierarchical_dependencies values to match the transformed inputs.
        transformed_dependencies = _transform_hierarchical_dependencies(
            train_X=train_X,
            hierarchical_dependencies=hierarchical_dependencies,
            input_transform=input_transform,
        )
        covar_module = HierarchicalConditionalKernel(
            dim=train_X.shape[-1],
            hierarchical_dependencies=transformed_dependencies,
            eval_hierarchical_features=eval_hierarchical_features,
            separate_hierarchical_features=separate_hierarchical_features,
            use_saas_prior=use_saas_prior,
            use_outputscale=use_outputscale,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        hierarchical_dependencies: dict[int, dict[int | float, list[int]]],
        eval_hierarchical_features: bool = True,
        separate_hierarchical_features: bool = True,
        use_saas_prior: bool = True,
        use_outputscale: bool = True,
    ) -> dict[str, Any]:
        return {
            **super().construct_inputs(training_data=training_data),
            "hierarchical_dependencies": hierarchical_dependencies,
            "eval_hierarchical_features": eval_hierarchical_features,
            "separate_hierarchical_features": separate_hierarchical_features,
            "use_saas_prior": use_saas_prior,
            "use_outputscale": use_outputscale,
        }


class HierarchicalConditionalKernelMultiTaskGP(MultiTaskGP):
    r"""Multi-Task GP with a conditional kernel that exploits correlations in
    hierarchical search spaces.

    This model extends ``MultiTaskGP`` by using a ``HierarchicalConditionalKernel``
    for the data covariance instead of the default kernel. Task correlations are
    still captured via a ``PositiveIndexKernel`` as in the parent class.

    The model can be single-output or multi-output, determined by ``output_tasks``.
    It supports dimension-scaled priors on the Kernel hyperparameters, which work best
    when covariates are normalized to the unit cube and outcomes are standardized
    (zero mean, unit variance). The standardization should be applied in a stratified
    fashion at the level of the tasks, rather than across all data points.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        hierarchical_dependencies: dict[int, dict[int | float, list[int]]],
        train_Yvar: Tensor | None = None,
        eval_hierarchical_features: bool = True,
        separate_hierarchical_features: bool = True,
        use_saas_prior: bool = True,
        use_outputscale: bool = True,
        likelihood: Likelihood | None = None,
        task_covar_prior: Prior | None = None,
        output_tasks: list[int] | None = None,
        rank: int | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        validate_task_values: bool = True,
    ) -> None:
        r"""Multi-Task GP model using a hierarchical conditional kernel.

        Args:
            train_X: A ``n x (d + 1)`` tensor of training data. One of the columns
                should contain the task features (see ``task_feature`` argument).
            train_Y: A ``n x 1`` tensor of training observations.
            task_feature: The index of the task feature
                (``-d <= task_feature <= d``).
            hierarchical_dependencies: A dictionary of the form
                ``{parent_index: {parent_value: children_indices}}`` that defines
                the hierarchical structure of the search space. Parent indices
                should refer to the feature indices AFTER removing the task feature.
            train_Yvar: An optional ``n`` tensor of observed measurement noise.
                If None, we infer the noise. Note that the inferred noise is
                common across all tasks.
            eval_hierarchical_features: Whether to evaluate correlations over
                hierarchical features or not. If false, the hierarchical features
                are merely used as flags to determine the active features, but
                they do not directly contribute to the kernel values.
            separate_hierarchical_features: This is relevant only if
                ``eval_hierarchical_features=True``. If true, the correlations of
                hierarchical features will be captured by a separate additive
                kernel.
            use_saas_prior: Whether to use the SAAS prior. If false, use the
                log-normal prior instead.
            use_outputscale: Whether to use the outputscale parameter. If false,
                the outputscales of each sub-kernel are fixed to 1.
            likelihood: A likelihood. The default is selected based on
                ``train_Yvar``. If ``train_Yvar`` is None, a
                ``HadamardGaussianLikelihood`` with inferred noise level is used.
                Otherwise, a ``FixedNoiseGaussianLikelihood`` is used.
            task_covar_prior: A Prior on the task covariance matrix. Must operate
                on p.s.d. matrices. A common prior for this is the ``LKJ`` prior.
            output_tasks: A list of task indices for which to compute model
                outputs for. If omitted, return outputs for all task indices.
            rank: The rank to be used for the index kernel. If omitted, use a
                full rank (i.e. number of tasks) kernel.
            all_tasks: By default, multi-task GPs infer the list of all tasks from
                the task features in ``train_X``. This is an experimental feature
                that enables creation of multi-task GPs with tasks that don't
                appear in the training data.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference. We use a ``Standardize`` transform if no
                ``outcome_transform`` is specified. Pass down ``None`` to use no
                outcome transform.
            input_transform: An input transform that is applied in the model's
                forward pass.
            validate_task_values: If True, validate that the task values supplied
                in the input are expected task values.
        """
        # Determine num_non_task_features for constructing the hierarchical kernel.
        # We need this before calling super().__init__().
        num_non_task_features = train_X.shape[-1] - 1

        # Transform hierarchical_dependencies values to match the transformed inputs.
        transformed_dependencies = _transform_hierarchical_dependencies(
            train_X=train_X,
            hierarchical_dependencies=hierarchical_dependencies,
            input_transform=input_transform,
        )

        # Construct the hierarchical conditional kernel for data covariance.
        covar_module = HierarchicalConditionalKernel(
            dim=num_non_task_features,
            hierarchical_dependencies=transformed_dependencies,
            eval_hierarchical_features=eval_hierarchical_features,
            separate_hierarchical_features=separate_hierarchical_features,
            use_saas_prior=use_saas_prior,
            use_outputscale=use_outputscale,
        )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
            likelihood=likelihood,
            task_covar_prior=task_covar_prior,
            output_tasks=output_tasks,
            rank=rank,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            validate_task_values=validate_task_values,
        )

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset | MultiTaskDataset,
        task_feature: int,
        hierarchical_dependencies: dict[int, dict[int | float, list[int]]],
        eval_hierarchical_features: bool = True,
        separate_hierarchical_features: bool = True,
        use_saas_prior: bool = True,
        use_outputscale: bool = True,
        output_tasks: list[int] | None = None,
        task_covar_prior: Prior | None = None,
        rank: int | None = None,
    ) -> dict[str, Any]:
        r"""Construct ``Model`` keyword arguments from a dataset and other args.

        Args:
            training_data: A ``SupervisedDataset`` or a ``MultiTaskDataset``.
            task_feature: Column index of embedded task indicator features.
            hierarchical_dependencies: A dictionary of the form
                ``{parent_index: {parent_value: children_indices}}`` that defines
                the hierarchical structure of the search space.
            eval_hierarchical_features: Whether to evaluate correlations over
                hierarchical features or not.
            separate_hierarchical_features: Whether to capture the correlations of
                hierarchical features with a separate additive kernel.
            use_saas_prior: Whether to use the SAAS prior.
            use_outputscale: Whether to use the outputscale parameter.
            output_tasks: A list of task indices for which to compute model
                outputs for. If omitted, return outputs for all task indices.
            task_covar_prior: A GPyTorch ``Prior`` object to use as prior on
                the cross-task covariance matrix.
            rank: The rank of the cross-task covariance matrix.
        """
        base_inputs = MultiTaskGP.construct_inputs(
            training_data=training_data,
            task_feature=task_feature,
            output_tasks=output_tasks,
            task_covar_prior=task_covar_prior,
            rank=rank,
        )
        base_inputs["hierarchical_dependencies"] = hierarchical_dependencies
        base_inputs["eval_hierarchical_features"] = eval_hierarchical_features
        base_inputs["separate_hierarchical_features"] = separate_hierarchical_features
        base_inputs["use_saas_prior"] = use_saas_prior
        base_inputs["use_outputscale"] = use_outputscale
        return base_inputs
