# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""3-level binary AddTree synthetic benchmark.

This is a strictly harder analogue of the Jenatton-small example: the
tree has **three** levels of binary categorical decisions (instead of
two), each interior or leaf node carries a 1-D continuous parameter, and
the off-best paths have a structured bias that grows with the Hamming
distance from the global-optimum path. The library must therefore learn
*both* the right path **and** the right continuous values at every level
on that path.

Tree structure::

    root
    ├── A
    │   ├── A1
    │   │   ├── A1a   (continuous v in [0, 1])  <-- best leaf
    │   │   └── A1b
    │   └── A2
    │       ├── A2a
    │       └── A2b
    └── B
        ├── B1
        │   ├── B1a
        │   └── B1b
        └── B2
            ├── B2a
            └── B2b

Eight root-to-leaf paths, BFS-encoded dimension is 29.

Objective (minimised)::

    f(p) = TARGET
         + path_bias(path_id)                  # depends on which path
         + sum_{node on path} (v_node - 0.3)**2

with ``TARGET = 0.05`` reached at path ``A/1/a`` with all on-path
``v = 0.3``. The most "wrong" path (``B/2/b``) has ``path_bias = 0.45``.

Run from the BoTorch repo root::

    python -m botorch_community.models.addtree.examples.jenatton3_synthetic

Contributor: maxc01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch_community.acquisition.addtree import optimize_addtree_acqf
from botorch_community.models.addtree import AddTreeGP, AddTreeSpace, decode, encode
from gpytorch.mlls import ExactMarginalLogLikelihood


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------


def _node1d(name: str, choices: list[dict] | None = None) -> dict:
    """Helper: a node with one continuous param and an optional choice."""
    out: dict = {
        "name": name,
        "continuous": [{"name": "v", "lo": 0.0, "hi": 1.0}],
    }
    if choices:
        out["choices"] = choices
    return out


def jenatton3_spec() -> dict[str, Any]:
    """Return the 3-level binary tree spec."""
    return {
        "name": "root",
        "choices": [
            {
                "name": "L1",
                "options": {
                    "A": _node1d(
                        "A",
                        choices=[
                            {
                                "name": "A.choice",
                                "options": {
                                    "1": _node1d(
                                        "A1",
                                        choices=[
                                            {
                                                "name": "A1.choice",
                                                "options": {
                                                    "a": _node1d("A1a"),
                                                    "b": _node1d("A1b"),
                                                },
                                            }
                                        ],
                                    ),
                                    "2": _node1d(
                                        "A2",
                                        choices=[
                                            {
                                                "name": "A2.choice",
                                                "options": {
                                                    "a": _node1d("A2a"),
                                                    "b": _node1d("A2b"),
                                                },
                                            }
                                        ],
                                    ),
                                },
                            }
                        ],
                    ),
                    "B": _node1d(
                        "B",
                        choices=[
                            {
                                "name": "B.choice",
                                "options": {
                                    "1": _node1d(
                                        "B1",
                                        choices=[
                                            {
                                                "name": "B1.choice",
                                                "options": {
                                                    "a": _node1d("B1a"),
                                                    "b": _node1d("B1b"),
                                                },
                                            }
                                        ],
                                    ),
                                    "2": _node1d(
                                        "B2",
                                        choices=[
                                            {
                                                "name": "B2.choice",
                                                "options": {
                                                    "a": _node1d("B2a"),
                                                    "b": _node1d("B2b"),
                                                },
                                            }
                                        ],
                                    ),
                                },
                            }
                        ],
                    ),
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Objective (library minimises -- pass -y as train_Y)
# ---------------------------------------------------------------------------


TARGET: float = 0.05  # global optimum value at A/1/a, v == 0.3 everywhere
BEST_PATH: tuple[str, str, str] = ("A", "1", "a")
PATH_BIAS_STEP: float = 0.15
CONT_OPT: float = 0.3


def path_bias(L1: str, L2: str, L3: str) -> float:
    """Heuristic bias = 0.15 * Hamming distance from the best path."""
    target = BEST_PATH
    parts = (L1, L2, L3)
    return PATH_BIAS_STEP * sum(1 for a, b in zip(parts, target) if a != b)


def obj_func(params: dict[str, Any]) -> float:
    """Synthetic objective; minimum value is :data:`TARGET`."""
    L1 = params["L1"]
    L2 = params[f"{L1}.choice"]
    n2 = f"{L1}{L2}"
    L3 = params[f"{n2}.choice"]
    n3 = f"{n2}{L3}"

    bias = path_bias(L1, L2, L3)
    cont = sum((params[f"{node}.v"] - CONT_OPT) ** 2 for node in (L1, n2, n3))
    return TARGET + bias + cont


# ---------------------------------------------------------------------------
# Random initial design
# ---------------------------------------------------------------------------


def random_params(rng: np.random.Generator) -> dict[str, Any]:
    L1 = rng.choice(["A", "B"])
    L2 = rng.choice(["1", "2"])
    L3 = rng.choice(["a", "b"])
    n1, n2, n3 = L1, f"{L1}{L2}", f"{L1}{L2}{L3}"
    return {
        "L1": L1,
        f"{L1}.choice": L2,
        f"{n2}.choice": L3,
        f"{n1}.v": float(rng.random()),
        f"{n2}.v": float(rng.random()),
        f"{n3}.v": float(rng.random()),
    }


# ---------------------------------------------------------------------------
# BO loop
# ---------------------------------------------------------------------------


def main(
    n_init: int = 20,
    n_iter: int = 60,
    beta: float = 3.0,
    q: int = 1,
    seed: int = 0,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run AddTree v2 BO on the 3-level synthetic. Returns summary statistics."""
    logger = logging.getLogger("addtree.examples.jenatton3_synthetic")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)

    space = AddTreeSpace.from_dict(jenatton3_spec())
    logger.info(
        "space: dim=%d  num_paths=%d  paths=%s",
        space.dim,
        space.num_paths,
        space.path_ids,
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    Xs: list[torch.Tensor] = []
    Ys: list[float] = []
    history: list[dict[str, Any]] = []

    # ---------- random init ----------
    for i in range(n_init):
        p = random_params(rng)
        y = obj_func(p)
        Xs.append(encode(space, p))
        Ys.append(-y)
        history.append({"iteration": i + 1, "phase": "init", "params": p, "value": y})
        logger.info(
            "iter %3d (init):  y=%.4f (best=%.4f)",
            i + 1,
            y,
            -max(Ys),
        )

    # ---------- BO loop ----------
    iter_idx = n_init
    while iter_idx < n_iter:
        train_X = torch.stack(Xs)
        train_Y = torch.tensor(Ys, dtype=torch.float64).unsqueeze(-1)

        model = AddTreeGP(space, train_X, train_Y)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

        candidates, _ = optimize_addtree_acqf(
            model,
            beta=beta,
            q=q,
            num_restarts=2,
            raw_samples=64,
        )

        for j in range(candidates.shape[0]):
            x_new = candidates[j]
            p = decode(space, x_new)
            y = obj_func(p)
            Xs.append(x_new)
            Ys.append(-y)
            iter_idx += 1
            history.append(
                {
                    "iteration": iter_idx,
                    "phase": "bo",
                    "params": p,
                    "value": y,
                }
            )
            logger.info(
                "iter %3d (bo):  y=%.4f (best=%.4f)",
                iter_idx,
                y,
                -max(Ys),
            )
            if output_dir is not None:
                with open(os.path.join(output_dir, f"iter_{iter_idx}.json"), "w") as f:
                    json.dump(history[-1], f)
            if iter_idx >= n_iter:
                break

    best_y = -max(Ys)
    summary = {
        "best_value": best_y,
        "gap_to_target": best_y - TARGET,
        "n_init": n_init,
        "n_iter": n_iter,
        "history": history,
    }
    logger.info(
        "FINAL best_value = %.4f (target = %.4f, gap = %.4f)",
        best_y,
        TARGET,
        best_y - TARGET,
    )
    return summary


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_init", type=int, default=20)
    p.add_argument("--n_iter", type=int, default=60)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--q", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()
    main(**vars(args))


if __name__ == "__main__":
    _cli()
