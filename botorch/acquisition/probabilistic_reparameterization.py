#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Probabilistic Reparameterization (with gradients) using Monte Carlo estimators.

See [Daulton2022bopr]_ for details.
"""

from abc import ABC
from collections import OrderedDict
from contextlib import ExitStack

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.wrapper import AbstractAcquisitionFunctionWrapper

from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputTransform,
    Normalize,
    OneHotToNumeric,
)

from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor
from torch.autograd import Function
from torch.nn.functional import one_hot


class _MCProbabilisticReparameterization(Function):
    r"""Evaluate the acquisition function via probabistic reparameterization.

    This uses a score function gradient estimator. See [Daulton2022bopr]_ for details.
    """

    @staticmethod
    def forward(
        ctx,
        X: Tensor,
        acq_function: AcquisitionFunction,
        input_tf: InputTransform,
        batch_limit: int | None,
        integer_indices: Tensor,
        cont_indices: Tensor,
        categorical_indices: Tensor,
        use_ma_baseline: bool,
        one_hot_to_numeric: OneHotToNumeric | None,
        ma_counter: Tensor | None,
        ma_hidden: Tensor | None,
        ma_decay: float | None,
    ):
        """Evaluate the expectation of the acquisition function under
        probabilistic reparameterization. Compute this in chunks of size
        batch_limit to enable scaling to large numbers of samples from the
        proposal distribution.
        """
        with ExitStack() as es:
            if ctx.needs_input_grad[0]:
                es.enter_context(torch.enable_grad())
            if cont_indices.shape[0] > 0:
                # only require gradient for continuous parameters
                ctx.cont_X = X[..., cont_indices].detach().requires_grad_(True)
                cont_idx = 0
                cols = []
                for col in range(X.shape[-1]):
                    # cont_indices is sorted in ascending order
                    if (
                        cont_idx < cont_indices.shape[0]
                        and col == cont_indices[cont_idx]
                    ):
                        cols.append(ctx.cont_X[..., cont_idx])
                        cont_idx += 1
                    else:
                        cols.append(X[..., col])
                X = torch.stack(cols, dim=-1)
            else:
                ctx.cont_X = None
            ctx.discrete_indices = input_tf["round"].discrete_indices
            ctx.cont_indices = cont_indices
            ctx.categorical_indices = categorical_indices
            ctx.ma_counter = ma_counter
            ctx.ma_hidden = ma_hidden
            ctx.X_shape = X.shape
            tilde_x_samples = input_tf(X.unsqueeze(-3))
            # save the rounding component

            rounding_component = tilde_x_samples.clone()
            if integer_indices.shape[0] > 0:
                X_integer_params = X[..., integer_indices].unsqueeze(-3)
                rounding_component[..., integer_indices] = (
                    (tilde_x_samples[..., integer_indices] - X_integer_params > 0)
                    | (X_integer_params == 1)
                ).to(tilde_x_samples)
            if categorical_indices.shape[0] > 0:
                rounding_component[..., categorical_indices] = tilde_x_samples[
                    ..., categorical_indices
                ]
            ctx.rounding_component = rounding_component[..., ctx.discrete_indices]
            ctx.tau = input_tf["round"].tau
            if hasattr(input_tf["round"], "base_samples"):
                ctx.base_samples = input_tf["round"].base_samples.detach()
            # save the probabilities
            if "unnormalize" in input_tf:
                unnormalized_X = input_tf["unnormalize"](X)
            else:
                unnormalized_X = X
            # this is only for the integer parameters
            ctx.prob = input_tf["round"].get_rounding_prob(unnormalized_X)

            if categorical_indices.shape[0] > 0:
                ctx.base_samples_categorical = input_tf[
                    "round"
                ].base_samples_categorical.clone()
            # compute the acquisition function where inputs are rounded according
            # to base_samples < prob
            ctx.tilde_x_samples = tilde_x_samples
            ctx.use_ma_baseline = use_ma_baseline
            acq_values_list = []
            start_idx = 0
            if one_hot_to_numeric is not None:
                tilde_x_samples = one_hot_to_numeric(tilde_x_samples)

            while start_idx < tilde_x_samples.shape[-3]:
                end_idx = min(start_idx + batch_limit, tilde_x_samples.shape[-3])
                acq_values = acq_function(tilde_x_samples[..., start_idx:end_idx, :, :])
                acq_values_list.append(acq_values)
                start_idx += batch_limit
            acq_values = torch.cat(acq_values_list, dim=-1)
            ctx.mean_acq_values = acq_values.mean(
                dim=-1
            )  # average over samples from proposal distribution
            ctx.acq_values = acq_values
            # update moving average baseline
            ctx.ma_hidden = ma_hidden.clone()
            ctx.ma_counter = ctx.ma_counter.clone()
            ctx.ma_decay = ma_decay
            # update in place
            ma_counter.add_(1)
            ma_hidden.sub_((ma_hidden - acq_values.detach().mean()) * (1 - ma_decay))
            return ctx.mean_acq_values.detach()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the gradient of the expectation of the acquisition function
        with respect to the parameters of the proposal distribution using
        Monte Carlo.
        """
        # this is overwriting the entire gradient w.r.t. x'
        # x' has shape batch_shape x q x d
        if ctx.needs_input_grad[0]:
            acq_values = ctx.acq_values
            mean_acq_values = ctx.mean_acq_values
            cont_indices = ctx.cont_indices
            discrete_indices = ctx.discrete_indices
            rounding_component = ctx.rounding_component
            # retrieve only the ordinal parameters
            expanded_acq_values = acq_values.view(*acq_values.shape, 1, 1).expand(
                acq_values.shape + rounding_component.shape[-2:]
            )
            prob = ctx.prob.unsqueeze(-3)
            if not ctx.use_ma_baseline:
                sample_level = expanded_acq_values * (rounding_component - prob)
            else:
                # use reinforce with the moving average baseline
                if ctx.ma_counter == 0:
                    baseline = 0.0
                else:
                    baseline = ctx.ma_hidden / (
                        1.0 - torch.pow(ctx.ma_decay, ctx.ma_counter)
                    )
                sample_level = (expanded_acq_values - baseline) * (
                    rounding_component - prob
                )

            grads = (sample_level / ctx.tau).mean(dim=-3)

            new_grads = (
                grad_output.view(
                    *grad_output.shape,
                    *[1 for _ in range(grads.ndim - grad_output.ndim)],
                )
                .expand(*grad_output.shape, *ctx.X_shape[-2:])
                .clone()
            )
            # multiply upstream grad_output by new gradients
            new_grads[..., discrete_indices] *= grads
            # use autograd for gradients w.r.t. the continuous parameters
            if ctx.cont_X is not None:
                auto_grad = torch.autograd.grad(
                    # note: this multiplies the gradient of mean_acq_values
                    # w.r.t to input by grad_output
                    mean_acq_values,
                    ctx.cont_X,
                    grad_outputs=grad_output,
                )[0]
                # overwrite grad_output since the previous step already
                # applied the chain rule
                new_grads[..., cont_indices] = auto_grad
            return (
                new_grads,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        return None, None, None, None, None, None, None, None, None, None, None, None


class AbstractProbabilisticReparameterizationInputTransform(InputTransform, ABC):
    r"""An abstract input transform to prepare inputs for PR.

    See [Daulton2022bopr]_ for details.

    This will typically be used in conjunction with normalization as
    follows:

    In eval() mode (i.e. after training), the inputs pass
    would typically be normalized to the unit cube (e.g. during candidate
    optimization).
    1. These are unnormalized back to the raw input space.
    2. The discrete values are created.
    3. All values are normalized to the unit cube.
    """

    def __init__(
        self,
        one_hot_bounds: Tensor,
        integer_indices: list[int] | None = None,
        categorical_features: dict[int, int] | None = None,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        mc_samples: int = 128,
        resample: bool = False,
        tau: float = 0.1,
    ) -> None:
        r"""Initialize transform.

        Args:
            one_hot_bounds: The raw search space bounds where categoricals are
                encoded in one-hot representation and the integer parameters
                are not normalized.
            integer_indices: The indices of the integer inputs.
            categorical_features: The indices and cardinality of
                each categorical feature. The features are assumed
                to be one-hot encoded. TODO: generalize to support
                alternative representations.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: False.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            mc_samples: The number of MC samples.
            resample: A boolean indicating whether to resample base samples
                at each forward pass.
            tau: The temperature parameter.
        """
        super().__init__()
        if integer_indices is None and categorical_features is None:
            raise ValueError(
                "integer_indices and/or categorical_features must be provided."
            )
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        discrete_indices = []
        self.categorical_features = categorical_features
        categorical_start_idx = (
            one_hot_bounds.shape[1]
            if self.categorical_features is None
            else min(self.categorical_features.keys())
        )

        if integer_indices is not None and len(integer_indices) > 0:
            error_msg = (
                f"{self.__class__.__name__} requires that the integer "
                "parameters are to the right of continuous parameters, and the "
                "left of categorical parameters."
            )
            leftmost_int, rightmost_int = min(integer_indices), max(integer_indices)
            if categorical_start_idx - rightmost_int != 1:
                # continuous parameters between ints and cats, or overlapping indices
                raise ValueError(error_msg)
            if set(range(leftmost_int, rightmost_int + 1)) != set(integer_indices):
                # non-contiguous integer indices
                raise ValueError(error_msg)
            self.register_buffer(
                "integer_indices", torch.tensor(integer_indices, dtype=torch.long)
            )
            discrete_indices += integer_indices
        else:
            self.integer_indices = None

        if self.categorical_features is not None:
            # check that the trailing dimensions are categoricals
            end = categorical_start_idx
            err_msg = (
                f"{self.__class__.__name__} requires that the categorical "
                "parameters are the rightmost elements."
            )
            for start, card in self.categorical_features.items():
                # the end of one one-hot representation should be followed
                # by the start of the next
                if end != start:
                    raise ValueError(err_msg)
                end = start + card
            if end != one_hot_bounds.shape[1]:
                # check end
                raise ValueError(err_msg)
        categorical_starts = []
        categorical_ends = []
        if self.categorical_features is not None:
            start = None
            for i, n_categories in categorical_features.items():
                if start is None:
                    start = i
                end = start + n_categories
                categorical_starts.append(start)
                categorical_ends.append(end)
                discrete_indices += list(range(start, end))
                start = end
        self.register_buffer(
            "discrete_indices",
            torch.tensor(
                discrete_indices, dtype=torch.long, device=one_hot_bounds.device
            ),
        )
        self.register_buffer(
            "categorical_starts",
            torch.tensor(
                categorical_starts, dtype=torch.long, device=one_hot_bounds.device
            ),
        )
        self.register_buffer(
            "categorical_ends",
            torch.tensor(
                categorical_ends, dtype=torch.long, device=one_hot_bounds.device
            ),
        )
        if integer_indices is None:
            self.register_buffer(
                "integer_bounds",
                torch.tensor([], dtype=torch.long, device=one_hot_bounds.device),
            )
        else:
            self.register_buffer("integer_bounds", one_hot_bounds[:, integer_indices])
        self.tau = tau

    def get_rounding_prob(self, X: Tensor) -> Tensor:
        X_prob = X.detach().clone()
        if self.integer_indices is not None:
            # compute probabilities for integers
            X_int = X_prob[..., self.integer_indices]
            X_int_abs = X_int.abs()
            offset = X_int_abs.floor()
            if self.tau is not None:
                X_prob[..., self.integer_indices] = torch.sigmoid(
                    (X_int_abs - offset - 0.5) / self.tau
                )
            else:
                X_prob[..., self.integer_indices] = X_int_abs - offset
        # compute probabilities for categoricals
        for start, end in zip(self.categorical_starts, self.categorical_ends):
            X_categ = X_prob[..., start:end]
            if self.tau is not None:
                X_prob[..., start:end] = torch.softmax(
                    (X_categ - 0.5) / self.tau, dim=-1
                )
            else:
                X_prob[..., start:end] = X_categ / X_categ.sum(dim=-1)
        return X_prob[..., self.discrete_indices]

    def _check_input_shape(self, X: Tensor) -> None:
        r"""Check that the input shape is valid for this transform."""
        if X.shape[-3] > 1:
            raise ValueError(
                f"Input to {self.__class__.__name__} must have a dimension of size 1 "
                f"at index -3 (got shape {X.shape})."
            )
        if X.shape[-2] > 1:
            raise ValueError(
                f"Input transform {self.__class__.__name__} does not support `n` > 1 "
                f"(got shape {X.shape})."
            )

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        return (
            super().equals(other=other)
            and torch.equal(self.integer_indices, other.integer_indices)
            and self.tau == other.tau
        )


class AnalyticProbabilisticReparameterizationInputTransform(
    AbstractProbabilisticReparameterizationInputTransform
):
    r"""An input transform to prepare inputs for analytic PR.

    See [Daulton2022bopr]_ for details.

    This will typically be used in conjunction with normalization as
    follows:

    In eval() mode (i.e. after training), the inputs pass
    would typically be normalized to the unit cube (e.g. during candidate
    optimization).
    1. These are unnormalized back to the raw input space.
    2. The discrete values are created.
    3. All values are normalized to the unit cube.
    """

    def __init__(
        self,
        one_hot_bounds: Tensor = None,
        integer_indices: list[int] | None = None,
        categorical_features: dict[int, int] | None = None,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        tau: float = 0.1,
    ) -> None:
        r"""Initialize transform.

        Args:
            one_hot_bounds: The raw search space bounds where categoricals are
                encoded in one-hot representation and the integer parameters
                are not normalized.
            integer_indices: The indices of the integer inputs.
            categorical_features: The indices and cardinality of
                each categorical feature. The features are assumed
                to be one-hot encoded. TODO: generalize to support
                alternative representations.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: False.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            mc_samples: The number of MC samples.
            resample: A boolean indicating whether to resample base samples
                at each forward pass.
            tau: The temperature parameter.
        """
        super().__init__(
            one_hot_bounds=one_hot_bounds,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
            transform_on_train=transform_on_train,
            transform_on_eval=transform_on_eval,
            transform_on_fantasize=transform_on_fantasize,
            tau=tau,
        )
        # create cartesian product of discrete options
        discrete_options = []
        dim = one_hot_bounds.shape[1]
        # get number of discrete parameters
        num_discrete_params = 0
        if self.integer_indices is not None:
            num_discrete_params += self.integer_indices.shape[0]
        if self.categorical_features is not None:
            num_discrete_params += len(self.categorical_features)
        # add zeros for continuous params to simplify code
        for _ in range(dim - len(self.discrete_indices)):
            discrete_options.append(
                torch.zeros(
                    1,
                    dtype=torch.long,
                    device=one_hot_bounds.device,
                )
            )
        if integer_indices is not None:
            # FIXME: this assumes that the integer dimensions are after the continuous
            # if we want to enforce this, we should test for it similarly to
            # categoricals
            for i in range(self.integer_bounds.shape[-1]):
                discrete_options.append(
                    torch.arange(
                        self.integer_bounds[0, i],
                        self.integer_bounds[1, i] + 1,
                        dtype=torch.long,
                        device=one_hot_bounds.device,
                    )
                )
        categorical_start_idx = len(discrete_options)
        if categorical_features is not None:
            for idx in sorted(categorical_features.keys()):
                cardinality = categorical_features[idx]
                discrete_options.append(
                    torch.arange(
                        cardinality, dtype=torch.long, device=one_hot_bounds.device
                    )
                )
        # categoricals are in numeric representation
        all_discrete_options = torch.cartesian_prod(*discrete_options)
        # one-hot encode the categoricals
        if categorical_features is not None and len(categorical_features) > 0:
            X_categ = torch.empty(
                *all_discrete_options.shape[:-1], sum(categorical_features.values())
            )
            start = 0
            for i, (idx, cardinality) in enumerate(
                sorted(categorical_features.items(), key=lambda kv: kv[0])
            ):
                start = idx - categorical_start_idx
                X_categ[..., start : start + cardinality] = one_hot(
                    all_discrete_options[..., -len(categorical_features) + i],
                    num_classes=cardinality,
                ).to(X_categ)
            all_discrete_options = torch.cat(
                [all_discrete_options[..., : -len(categorical_features)], X_categ],
                dim=-1,
            )
        self.register_buffer("all_discrete_options", all_discrete_options)

    def get_probs(self, X: Tensor) -> Tensor:
        """
        Args:
            X: a `batch_shape x n x d`-dim tensor

        Returns:
            A `batch_shape x n_discrete x n`-dim tensors of probabilities of each
                discrete config under X.
        """
        # note this method should be differentiable
        X_prob = torch.ones(
            *X.shape[:-2],
            self.all_discrete_options.shape[0],
            X.shape[-2],
            dtype=X.dtype,
            device=X.device,
        )
        # n_discrete x batch_shape x n x d
        all_discrete_options = self.all_discrete_options.view(
            *([1] * (X.ndim - 2)), self.all_discrete_options.shape[0], *X.shape[-2:]
        ).expand(*X.shape[:-2], self.all_discrete_options.shape[0], *X.shape[-2:])
        X = X.unsqueeze(-3)
        if self.integer_indices is not None:
            # compute probabilities for integers
            X_int = X[..., self.integer_indices]
            X_int_abs = X_int.abs()
            offset = X_int_abs.floor()
            # note we don't actually need the sigmoid here
            X_prob_int = torch.sigmoid((X_int_abs - offset - 0.5) / self.tau)
            # X_prob_int = X_int_abs - offset
            for int_idx, idx in enumerate(self.integer_indices):
                offset_i = offset[..., int_idx]
                all_discrete_i = all_discrete_options[..., idx]
                diff = (offset_i + 1) - all_discrete_i
                round_up_mask = diff == 0
                round_down_mask = diff == 1
                neither_mask = ~(round_up_mask | round_down_mask)
                prob = X_prob_int[..., int_idx].expand(round_up_mask.shape)
                # need to be careful with in-place ops here for autograd
                X_prob[round_up_mask] = X_prob[round_up_mask] * prob[round_up_mask]
                X_prob[round_down_mask] = X_prob[round_down_mask] * (
                    1 - prob[round_down_mask]
                )
                X_prob[neither_mask] = X_prob[neither_mask] * 0

        # compute probabilities for categoricals
        for start, end in zip(self.categorical_starts, self.categorical_ends):
            X_categ = X[..., start:end]
            X_prob_c = torch.softmax((X_categ - 0.5) / self.tau, dim=-1).expand(
                *X_categ.shape[:-3], all_discrete_options.shape[-3], *X_categ.shape[-2:]
            )
            for i in range(X_prob_c.shape[-1]):
                mask = all_discrete_options[..., start + i] == 1
                X_prob[mask] = X_prob[mask] * X_prob_c[..., i][mask]

        return X_prob

    def transform(self, X: Tensor) -> Tensor:
        r"""Round the inputs.

        This is not sample-path differentiable.

        Args:
            X: A `batch_shape x 1 x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x num_discrete_options x n x d`-dim tensor of rounded inputs.
        """
        self._check_input_shape(X)

        n_discrete = self.discrete_indices.shape[0]
        all_discrete_options = self.all_discrete_options.view(
            *([1] * (X.ndim - 3)), self.all_discrete_options.shape[0], *X.shape[-2:]
        ).expand(*X.shape[:-3], self.all_discrete_options.shape[0], *X.shape[-2:])
        if X.shape[-1] > n_discrete:
            X = X.expand(
                *X.shape[:-3], self.all_discrete_options.shape[0], *X.shape[-2:]
            )
            return torch.cat(
                [X[..., :-n_discrete], all_discrete_options[..., -n_discrete:]], dim=-1
            )
        return all_discrete_options


class MCProbabilisticReparameterizationInputTransform(
    AbstractProbabilisticReparameterizationInputTransform
):
    r"""An input transform to prepare inputs for Monte Carlo PR.

    See [Daulton2022bopr]_ for details.

    This will typically be used in conjunction with normalization as
    follows:

    In eval() mode (i.e. after training), the inputs pass
    would typically be normalized to the unit cube (e.g. during candidate
    optimization).
    1. These are unnormalized back to the raw input space.
    2. The discrete ordinal values are sampled.
    3. All values are normalized to the unit cube.
    """

    def __init__(
        self,
        one_hot_bounds: Tensor,
        integer_indices: list[int] | None = None,
        categorical_features: dict[int, int] | None = None,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        mc_samples: int = 128,
        resample: bool = False,
        tau: float = 0.1,
    ) -> None:
        r"""Initialize transform.

        Args:
            one_hot_bounds: The raw search space bounds where categoricals are
                encoded in one-hot representation and the integer parameters
                are not normalized.
            integer_indices: The indices of the integer inputs.
            categorical_features: The indices and cardinality of
                each categorical feature. The features are assumed
                to be one-hot encoded. TODO: generalize to support
                alternative representations.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: False.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            mc_samples: The number of MC samples.
            resample: A boolean indicating whether to resample base samples
                at each forward pass.
            tau: The temperature parameter.
        """
        super().__init__(
            one_hot_bounds=one_hot_bounds,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
            transform_on_train=transform_on_train,
            transform_on_eval=transform_on_eval,
            transform_on_fantasize=transform_on_fantasize,
            tau=tau,
        )
        self.mc_samples = mc_samples
        self.resample = resample

    def transform(self, X: Tensor) -> Tensor:
        r"""Round the inputs.

        This is not sample-path differentiable.

        Args:
            X: A `batch_shape x 1 x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x mc_samples x n x d`-dim tensor of rounded inputs.
        """
        self._check_input_shape(X)

        X_expanded = X.expand(*X.shape[:-3], self.mc_samples, *X.shape[-2:]).clone()
        X_prob = self.get_rounding_prob(X=X)
        if self.integer_indices is not None:
            X_int = X[..., self.integer_indices].detach()
            assert X.ndim > 1
            if X.ndim == 2:
                X.unsqueeze(-1)
            if (
                not hasattr(self, "base_samples")
                or self.base_samples.shape[-2:] != X_int.shape[-2:]
                or self.resample
            ):
                # construct sobol base samples
                bounds = torch.zeros(
                    2, X_int.shape[-1], dtype=X_int.dtype, device=X_int.device
                )
                bounds[1] = 1
                self.register_buffer(
                    "base_samples",
                    draw_sobol_samples(
                        bounds=bounds,
                        n=self.mc_samples,
                        q=X_int.shape[-2],
                        seed=torch.randint(0, 100000, (1,)).item(),
                    ),
                )
            X_int_abs = X_int.abs()
            # perform exact rounding
            is_negative = X_int < 0
            offset = X_int_abs.floor()
            prob = X_prob[..., : self.integer_indices.shape[0]]
            rounding_component = (prob >= self.base_samples).to(
                dtype=X.dtype,
            )
            X_abs_rounded = offset + rounding_component
            X_int_new = (-1) ** is_negative.to(offset) * X_abs_rounded
            # clamp to bounds
            X_expanded[..., self.integer_indices] = torch.minimum(
                torch.maximum(X_int_new, self.integer_bounds[0]), self.integer_bounds[1]
            )

        # sample for categoricals
        if self.categorical_features is not None and len(self.categorical_features) > 0:
            if (
                not hasattr(self, "base_samples_categorical")
                or self.base_samples_categorical.shape[-2] != X.shape[-2]
                or self.resample
            ):
                bounds = torch.zeros(
                    2, len(self.categorical_features), dtype=X.dtype, device=X.device
                )
                bounds[1] = 1
                self.register_buffer(
                    "base_samples_categorical",
                    draw_sobol_samples(
                        bounds=bounds,
                        n=self.mc_samples,
                        q=X.shape[-2],
                        seed=torch.randint(0, 100000, (1,)).item(),
                    ),
                )

            # sample from multinomial as argmin_c [sample_c * exp(-x_c)]
            sample_d_start_idx = 0
            X_categ_prob = X_prob
            if self.integer_indices is not None:
                n_ints = self.integer_indices.shape[0]
                if n_ints > 0:
                    X_categ_prob = X_prob[..., n_ints:]

            for i, cardinality in enumerate(self.categorical_features.values()):
                sample_d_end_idx = sample_d_start_idx + cardinality
                start = self.categorical_starts[i]
                end = self.categorical_ends[i]
                cum_prob = X_categ_prob[
                    ..., sample_d_start_idx:sample_d_end_idx
                ].cumsum(dim=-1)
                categories = (
                    (
                        (cum_prob > self.base_samples_categorical[..., i : i + 1])
                        .long()
                        .cumsum(dim=-1)
                        == 1
                    )
                    .long()
                    .argmax(dim=-1)
                )
                # one-hot encode
                X_expanded[..., start:end] = one_hot(
                    categories, num_classes=cardinality
                ).to(X)
                sample_d_start_idx = sample_d_end_idx

        return X_expanded

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        if hasattr(self, "base_samples") and hasattr(other, "base_samples"):
            equal_base_samples = torch.equal(self.base_samples, other.base_samples)
        else:
            equal_base_samples = not hasattr(self, "base_samples") and not hasattr(
                other, "base_samples"
            )
        return (
            super().equals(other=other)
            and (self.resample == other.resample)
            and (self.mc_samples == other.mc_samples)
            and equal_base_samples
        )


def get_probabilistic_reparameterization_input_transform(
    one_hot_bounds: Tensor,
    integer_indices: list[int] | None = None,
    categorical_features: dict[int, int] | None = None,
    use_analytic: bool = False,
    mc_samples: int = 128,
    resample: bool = False,
    tau: float = 0.1,
) -> ChainedInputTransform:
    r"""Construct InputTransform for Probabilistic Reparameterization.

    Note: this is intended to be used only for acquisition optimization
    in via the AnalyticProbabilisticReparameterization and
    MCProbabilisticReparameterization classes. This is not intended to be
    attached to a botorch Model.

    See [Daulton2022bopr]_ for details.

    Args:
        one_hot_bounds: The raw search space bounds where categoricals are
            encoded in one-hot representation and the integer parameters
            are not normalized.
        integer_indices: The indices of the integer parameters
        categorical_features: A dictionary mapping indices to cardinalities
            for the categorical features.
        use_analytic: A boolean indicating whether to use analytic
            probabilistic reparameterization.
        mc_samples: The number of MC samples for MC probabilistic
            reparameterization.
        resample: A boolean indicating whether to resample with MC
            probabilistic reparameterization on each forward pass.
        tau: The temperature parameter used to determine the probabilities.

    Returns:
        The probabilistic reparameterization input transformation.
    """
    tfs = OrderedDict()
    if integer_indices is not None and len(integer_indices) > 0:
        # unnormalize to integer space
        tfs["unnormalize"] = Normalize(
            d=one_hot_bounds.shape[1],
            bounds=one_hot_bounds,
            indices=integer_indices,
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=False,
            reverse=True,
        )
    if use_analytic:
        tfs["round"] = AnalyticProbabilisticReparameterizationInputTransform(
            one_hot_bounds=one_hot_bounds,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
            tau=tau,
        )
    else:
        tfs["round"] = MCProbabilisticReparameterizationInputTransform(
            one_hot_bounds=one_hot_bounds,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
            resample=resample,
            mc_samples=mc_samples,
            tau=tau,
        )
    if integer_indices is not None and len(integer_indices) > 0:
        # normalize to unit cube
        tfs["normalize"] = Normalize(
            d=one_hot_bounds.shape[1],
            bounds=one_hot_bounds,
            indices=integer_indices,
            transform_on_train=False,
            transform_on_eval=True,
            transform_on_fantasize=False,
            reverse=False,
        )
    tf = ChainedInputTransform(**tfs)
    tf.eval()
    return tf


class AbstractProbabilisticReparameterization(AbstractAcquisitionFunctionWrapper):
    r"""Acquisition Function Wrapper that leverages probabilistic reparameterization.

    The forward method is abstract and must be implemented.

    See [Daulton2022bopr]_ for details.
    """

    input_transform: ChainedInputTransform

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        one_hot_bounds: Tensor,
        integer_indices: list[int] | None = None,
        categorical_features: dict[int, int] | None = None,
        batch_limit: int = 32,
        apply_numeric: bool = False,
        **kwargs,
    ) -> None:
        r"""Initialize probabilistic reparameterization (PR).

        Args:
            acq_function: The acquisition function.
            one_hot_bounds: The raw search space bounds where categoricals are
                encoded in one-hot representation and the integer parameters
                are not normalized.
            integer_indices: The indices of the integer parameters
            categorical_features: A dictionary mapping indices to cardinalities
                for the categorical features.
            batch_limit: The chunk size used in evaluating PR to limit memory
                overhead.
            apply_numeric: A boolean indicated if categoricals should be supplied
                to the underlying acquisition function in numeric representation.
        """
        if categorical_features is None and integer_indices is None:
            raise NotImplementedError(
                "Categorical features or integer indices must be provided."
            )
        super().__init__(acq_function=acq_function)
        self.batch_limit = batch_limit

        if apply_numeric:
            self.one_hot_to_numeric = OneHotToNumeric(
                categorical_features=categorical_features,
                transform_on_train=False,
                transform_on_eval=True,
                transform_on_fantasize=False,
            )
            self.one_hot_to_numeric.eval()
        else:
            self.one_hot_to_numeric = None
        discrete_indices = []
        if integer_indices is not None:
            self.register_buffer(
                "integer_indices",
                torch.tensor(
                    integer_indices, dtype=torch.long, device=one_hot_bounds.device
                ),
            )
            self.register_buffer("integer_bounds", one_hot_bounds[:, integer_indices])
            discrete_indices.extend(integer_indices)
        else:
            self.register_buffer(
                "integer_indices",
                torch.tensor([], dtype=torch.long, device=one_hot_bounds.device),
            )
            self.register_buffer(
                "integer_bounds",
                torch.tensor(
                    [], dtype=one_hot_bounds.dtype, device=one_hot_bounds.device
                ),
            )
        dim = one_hot_bounds.shape[1]
        if categorical_features is not None and len(categorical_features) > 0:
            categorical_indices = list(range(min(categorical_features.keys()), dim))
            discrete_indices.extend(categorical_indices)
            self.register_buffer(
                "categorical_indices",
                torch.tensor(
                    categorical_indices,
                    dtype=torch.long,
                    device=one_hot_bounds.device,
                ),
            )
            self.categorical_features = categorical_features
        else:
            self.register_buffer(
                "categorical_indices",
                torch.tensor(
                    [],
                    dtype=torch.long,
                    device=one_hot_bounds.device,
                ),
            )

        self.register_buffer(
            "cont_indices",
            torch.tensor(
                sorted(set(range(dim)) - set(discrete_indices)),
                dtype=torch.long,
                device=one_hot_bounds.device,
            ),
        )
        self.model = acq_function.model  # for sample_around_best heuristic
        # moving average baseline
        self.register_buffer(
            "ma_counter",
            torch.zeros(1, dtype=one_hot_bounds.dtype, device=one_hot_bounds.device),
        )
        self.register_buffer(
            "ma_hidden",
            torch.zeros(1, dtype=one_hot_bounds.dtype, device=one_hot_bounds.device),
        )
        self.register_buffer(
            "ma_baseline",
            torch.zeros(1, dtype=one_hot_bounds.dtype, device=one_hot_bounds.device),
        )

    def sample_candidates(self, X: Tensor) -> Tensor:
        if "unnormalize" in self.input_transform:
            unnormalized_X = self.input_transform["unnormalize"](X)
        else:
            unnormalized_X = X.clone()
        prob = self.input_transform["round"].get_rounding_prob(X=unnormalized_X)
        discrete_idx = 0
        for i in self.integer_indices:
            p = prob[..., discrete_idx]
            rounding_component = torch.distributions.Bernoulli(probs=p).sample()
            unnormalized_X[..., i] = unnormalized_X[..., i].floor() + rounding_component
            discrete_idx += 1
        if len(self.integer_indices) > 0:
            unnormalized_X[..., self.integer_indices] = torch.minimum(
                torch.maximum(
                    unnormalized_X[..., self.integer_indices], self.integer_bounds[0]
                ),
                self.integer_bounds[1],
            )
        # this is the starting index for the categoricals in unnormalized_X
        raw_idx = self.cont_indices.shape[0] + discrete_idx
        if self.categorical_indices.shape[0] > 0:
            for cardinality in self.categorical_features.values():
                discrete_end = discrete_idx + cardinality
                p = prob[..., discrete_idx:discrete_end]
                z = one_hot(
                    torch.distributions.Categorical(probs=p).sample(),
                    num_classes=cardinality,
                )
                raw_end = raw_idx + cardinality
                unnormalized_X[..., raw_idx:raw_end] = z
                discrete_idx = discrete_end
                raw_idx = raw_end
        # normalize X
        if "normalize" in self.input_transform:
            return self.input_transform["normalize"](unnormalized_X)
        return unnormalized_X


class AnalyticProbabilisticReparameterization(AbstractProbabilisticReparameterization):
    """Analytic probabilistic reparameterization.

    Note: this is only reasonable from a computation perspective for relatively
    small numbers of discrete options (probably less than a few thousand).
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        one_hot_bounds: Tensor,
        integer_indices: list[int] | None = None,
        categorical_features: dict[int, int] | None = None,
        batch_limit: int = 32,
        apply_numeric: bool = False,
        tau: float = 0.1,
    ) -> None:
        """Initialize probabilistic reparameterization (PR).

        Args:
            acq_function: The acquisition function.
            one_hot_bounds: The raw search space bounds where categoricals are
                encoded in one-hot representation and the integer parameters
                are not normalized.
            integer_indices: The indices of the integer parameters
            categorical_features: A dictionary mapping indices to cardinalities
                for the categorical features.
            batch_limit: The chunk size used in evaluating PR to limit memory
                overhead.
            apply_numeric: A boolean indicated if categoricals should be supplied
                to the underlying acquisition function in numeric representation.
            tau: The temperature parameter used to determine the probabilities.

        """
        super().__init__(
            acq_function=acq_function,
            integer_indices=integer_indices,
            one_hot_bounds=one_hot_bounds,
            categorical_features=categorical_features,
            batch_limit=batch_limit,
            apply_numeric=apply_numeric,
        )
        # create input transform
        # need to compute cross product of discrete options and weights
        self.input_transform = get_probabilistic_reparameterization_input_transform(
            one_hot_bounds=one_hot_bounds,
            use_analytic=True,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
            tau=tau,
        )

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate PR."""
        X_discrete_all = self.input_transform(X.unsqueeze(-3))
        acq_values_list = []
        start_idx = 0
        if self.one_hot_to_numeric is not None:
            X_discrete_all = self.one_hot_to_numeric(X_discrete_all)
        if X.shape[-2] != 1:
            # PR does not support batch > 1. This is caught by the input transform
            raise NotImplementedError  # pragma: no cover

        # save the probabilities
        if "unnormalize" in self.input_transform:
            unnormalized_X = self.input_transform["unnormalize"](X)
        else:
            unnormalized_X = X
        # this is batch_shape x n_discrete (after squeezing)
        probs = self.input_transform["round"].get_probs(X=unnormalized_X).squeeze(-1)
        # TODO: filter discrete configs with zero probability. This would require
        # padding because there may be a different number in each batch.
        while start_idx < X_discrete_all.shape[-3]:
            end_idx = min(start_idx + self.batch_limit, X_discrete_all.shape[-3])
            acq_values = self.acq_func(X_discrete_all[..., start_idx:end_idx, :, :])
            acq_values_list.append(acq_values)
            start_idx += self.batch_limit
        # this is batch_shape x n_discrete
        acq_values = torch.cat(acq_values_list, dim=-1)
        # now weight the acquisition values by probabilities
        return (acq_values * probs).sum(dim=-1)


class MCProbabilisticReparameterization(AbstractProbabilisticReparameterization):
    r"""MC-based probabilistic reparameterization.

    See [Daulton2022bopr]_ for details.
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        one_hot_bounds: Tensor,
        integer_indices: list[int] | None = None,
        categorical_features: dict[int, int] | None = None,
        batch_limit: int = 32,
        apply_numeric: bool = False,
        mc_samples: int = 128,
        use_ma_baseline: bool = True,
        tau: float = 0.1,
        ma_decay: float = 0.7,
        resample: bool = True,
    ) -> None:
        """Initialize probabilistic reparameterization (PR).

        Args:
            acq_function: The acquisition function.
            one_hot_bounds: The raw search space bounds where categoricals are
                encoded in one-hot representation and the integer parameters
                are not normalized.
            integer_indices: The indices of the integer parameters
            categorical_features: A dictionary mapping indices to cardinalities
                for the categorical features.
            batch_limit: The chunk size used in evaluating PR to limit memory
                overhead.
            apply_numeric: A boolean indicated if categoricals should be supplied
                to the underlying acquisition function in numeric representation.
            mc_samples: The number of MC samples for MC probabilistic
                reparameterization.
            use_ma_baseline: A boolean indicating whether to use a moving average
                baseline for variance reduction.
            tau: The temperature parameter used to determine the probabilities.
            ma_decay: The decay parameter in the moving average baseline.
                Default: 0.7
            resample: A boolean indicating whether to resample with MC
                probabilistic reparameterization on each forward pass.

        """
        super().__init__(
            acq_function=acq_function,
            one_hot_bounds=one_hot_bounds,
            integer_indices=integer_indices,
            categorical_features=categorical_features,
            batch_limit=batch_limit,
            apply_numeric=apply_numeric,
        )
        if self.batch_limit is None:
            self.batch_limit = mc_samples
        self.use_ma_baseline = use_ma_baseline
        self._pr_acq_function = _MCProbabilisticReparameterization
        # create input transform
        self.input_transform = get_probabilistic_reparameterization_input_transform(
            integer_indices=integer_indices,
            one_hot_bounds=one_hot_bounds,
            categorical_features=categorical_features,
            mc_samples=mc_samples,
            tau=tau,
            resample=resample,
        )
        self.ma_decay = ma_decay

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate MC probabilistic reparameterization."""
        return self._pr_acq_function.apply(
            X,
            self.acq_func,
            self.input_transform,
            self.batch_limit,
            self.integer_indices,
            self.cont_indices,
            self.categorical_indices,
            self.use_ma_baseline,
            self.one_hot_to_numeric,
            self.ma_counter,
            self.ma_hidden,
            self.ma_decay,
        )
