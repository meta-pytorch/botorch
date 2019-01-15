#! /usr/bin/env python3

from typing import Optional

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from torch import Tensor

from .posterior import Posterior


class GPyTorchPosterior(Posterior):
    def __init__(self, mvn: MultivariateNormal) -> None:
        self.mvn = mvn

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample from the posterior

        We follow the convention of GPyTorch which gives base_samples priority
        over sample_shape, but we also check that sample_shape agrees with
        base_samples.
        """
        if sample_shape is not None and base_samples is not None:
            if base_samples.shape[: len(sample_shape)] != sample_shape:
                raise RuntimeError("Sample shape disagrees with base_samples.")
        if sample_shape is None and base_samples is None:
            kwargs = {}
        elif base_samples is not None:
            kwargs = {"base_samples": base_samples}
        elif sample_shape is not None:
            kwargs = {"sample_shape": sample_shape}
        with gpytorch.settings.fast_computations(covar_root_decomposition=False):
            samples = self.mvn.rsample(**kwargs)
        return samples

    @property
    def mean(self):
        return self.mvn.mean

    @property
    def variance(self):
        return self.mvn.variance

    def zero_mean_mvn_samples(self, num_samples: int) -> Tensor:
        return self.mvn.lazy_covariance_matrix.zero_mean_mvn_samples(num_samples)

    def get_base_samples(self, sample_shape: Optional[torch.Size] = None) -> Tensor:
        return self.mvn.get_base_samples(sample_shape=sample_shape)
