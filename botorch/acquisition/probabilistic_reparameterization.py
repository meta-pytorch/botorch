#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Probabilistic Reparameterization (with gradients) using Monte Carlo estimators.

See [Daulton2022bopr]_ for details.
"""

from contextlib import ExitStack

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.wrapper import AbstractAcquisitionFunctionWrapper
from botorch.models.transforms.factory import (
    get_probabilistic_reparameterization_input_transform,
)

from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputTransform,
    OneHotToNumeric,
)
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
                "categorical_features or integer indices must be provided."
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
            raise NotImplementedError

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
