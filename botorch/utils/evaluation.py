#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import log

import torch
from botorch.utils.transforms import is_fully_bayesian
from gpytorch.likelihoods import HadamardGaussianLikelihood
from gpytorch.models.exact_gp import ExactGP

MLL = "MLL"
AIC = "AIC"
BIC = "BIC"


def compute_in_sample_model_fit_metric(model: ExactGP, criterion: str) -> float:
    """Compute a in-sample model fit metric.

    Args:
        model: A fitted ExactGP.
        criterion: Evaluation criterion. One of "MLL", "AIC", "BIC". AIC
            penalizes the MLL based on the number of parameters. BIC uses
            a slightly different penalty based on the number of parameters
            and data points.

    Returns:
        The in-sample evaluation metric.
    """
    if criterion not in (AIC, BIC, MLL):
        raise ValueError(f"Invalid evaluation criterion {criterion}.")
    if is_fully_bayesian(model=model):
        model.train(reset=False)
    else:
        model.train()
    with torch.no_grad():
        output = model(*model.train_inputs)
        likelihood_inputs = model.train_inputs
        map_tasks = getattr(model, "_map_tasks", None)
        if (
            isinstance(model.likelihood, HadamardGaussianLikelihood)
            and map_tasks is not None
        ):
            # MultiTaskGP.forward remaps task labels for its kernel/mean, but
            # leaves train_inputs in their original labeling. The task-noise
            # likelihood requires the same contiguous indices as the kernel.
            task_values = model.train_inputs[0][..., model._task_feature].long()
            likelihood_inputs = (map_tasks(task_values).unsqueeze(-1),)
        output = model.likelihood(output, *likelihood_inputs)
        mll = output.log_prob(model.train_targets)
        # compute average MLL over MCMC samples if the model is fully bayesian
        mll_scalar = mll.mean().item()
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    if is_fully_bayesian(model=model):
        num_params /= mll.shape[0]
    if criterion == AIC:
        return 2 * num_params - 2 * mll_scalar
    elif criterion == BIC:
        return num_params * log(model.train_inputs[0].shape[-2]) - 2 * mll_scalar
    return mll_scalar
