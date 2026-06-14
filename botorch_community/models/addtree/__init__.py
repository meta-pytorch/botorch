# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""AddTree GP for tree-structured / conditional parameter spaces.

Implements the AddTree covariance kernel and Bayesian-optimisation
utilities from:

.. [Ma2020addtree]
    X. Ma and M. Blaschko. Additive Tree-Structured Covariance Function
    for Conditional Parameter Spaces in Bayesian Optimization. AISTATS
    2020.

User-facing API:

* :class:`AddTreeSpace` -- declarative spec of the conditional parameter
  space (constructed from a Python builder, a nested dict, or a YAML
  file).
* :class:`AddTreeGP` -- the corresponding GP model.
* :func:`encode` / :func:`decode` -- bridge user-facing parameter dicts
  and BFS tensors.
* :func:`~botorch_community.acquisition.addtree.optimize_addtree_acqf`
  -- one-call BO step (a thin wrapper over
  :func:`botorch.optim.optimize_acqf_mixed`).

Contributor: maxc01
"""

from botorch_community.models.addtree.encoding import decode, encode, param_key
from botorch_community.models.addtree.kernels import (
    AddTreeDeltaKernel,
    build_addtree_kernel,
)
from botorch_community.models.addtree.model import AddTreeGP
from botorch_community.models.addtree.space import (
    AddTreeSpace,
    Choice,
    Continuous,
    make_node,
    Node,
)


__all__ = [
    "AddTreeDeltaKernel",
    "AddTreeGP",
    "AddTreeSpace",
    "Choice",
    "Continuous",
    "Node",
    "build_addtree_kernel",
    "decode",
    "encode",
    "make_node",
    "param_key",
]
