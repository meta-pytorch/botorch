# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for converting between PyTorch tensors and JAX arrays."""

import jax.numpy as jnp
import numpy as np
import torch
from jax import Array
from torch import Tensor


def torch_to_jax(t: Tensor) -> Array:
    """Convert a PyTorch tensor to a JAX array."""
    return jnp.array(t.detach().cpu().numpy())


def jax_to_torch(a: Array, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Convert a JAX array to a PyTorch tensor."""
    return torch.tensor(np.asarray(a), device=device, dtype=dtype)
