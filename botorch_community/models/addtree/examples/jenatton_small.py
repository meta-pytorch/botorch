# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""End-to-end AddTree BO example on the Jenatton-small synthetic function.

Reproduces the Jenatton-small benchmark from the AddTree paper using
the v2 declarative API. Compared to the upstream ``addtree`` repo this
example has *no* registry-management ritual, *no* per-tree adapter
function, and *no* manual BFS-encoding logic --- everything goes
through :class:`AddTreeSpace` and :func:`optimize_addtree_acqf`.

Run from the BoTorch repo root::

    python -m botorch_community.models.addtree.examples.jenatton_small

The optimum of Jenatton-small is ``0.1``; ~30 BO iterations should
reliably reach it.

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
# 1. Declare the parameter space.
# ---------------------------------------------------------------------------


def jenatton_small_spec() -> dict[str, Any]:
    r"""Return the Jenatton-small spec as a nested dict.

    Tree (each leaf has a 1-D continuous parameter named ``v``)::

                       root
                       /  \
                  L1=left  L1=right
                 (.v)        (.v)
                 /   \       /    \
              L2=l L2=r   L2=l  L2=r
              (.v) (.v)  (.v)  (.v)
    """
    leaf = lambda name: {  # noqa: E731 - tiny helper
        "name": name,
        "continuous": [{"name": "v", "lo": 0.0, "hi": 1.0}],
    }
    return {
        "name": "root",
        "choices": [
            {
                "name": "L1",
                "options": {
                    "left": {
                        "name": "L1_left",
                        "continuous": [{"name": "v", "lo": 0.0, "hi": 1.0}],
                        "choices": [
                            {
                                "name": "L2",
                                "options": {
                                    "left": leaf("L2_ll"),
                                    "right": leaf("L2_lr"),
                                },
                            }
                        ],
                    },
                    "right": {
                        "name": "L1_right",
                        "continuous": [{"name": "v", "lo": 0.0, "hi": 1.0}],
                        "choices": [
                            {
                                "name": "L2",
                                "options": {
                                    "left": leaf("L2_rl"),
                                    "right": leaf("L2_rr"),
                                },
                            }
                        ],
                    },
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# 2. Define the (library-minimised) objective. Optimum value is 0.1.
# ---------------------------------------------------------------------------


def obj_func(params: dict[str, Any]) -> float:
    """Synthetic objective from Jenatton 2017."""
    SHIFT = 0.5
    L1 = params["L1"]
    L2 = params["L2"]
    L1_v = params[f"L1_{L1}.v"]
    L2_v = params[f"L2_{L1[0]}{L2[0]}.v"]
    bias = {"left,left": 0.1, "left,right": 0.2, "right,left": 0.3, "right,right": 0.4}[
        f"{L1},{L2}"
    ]
    return L1_v + (L2_v - SHIFT) ** 2 + bias


# ---------------------------------------------------------------------------
# 3. Random initial design (uniform over options + uniform on continuous).
# ---------------------------------------------------------------------------


def random_params(rng: np.random.Generator) -> dict[str, Any]:
    L1 = rng.choice(["left", "right"])
    L2 = rng.choice(["left", "right"])
    return {
        "L1": L1,
        f"L1_{L1}.v": float(rng.random()),
        "L2": L2,
        f"L2_{L1[0]}{L2[0]}.v": float(rng.random()),
    }


# ---------------------------------------------------------------------------
# 4. BO loop.
# ---------------------------------------------------------------------------


def main(
    n_init: int = 10,
    n_iter: int = 100,
    beta: float = 3.0,
    seed: int = 0,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run AddTree v2 BO on Jenatton-small. Returns summary statistics."""
    logger = logging.getLogger("addtree.examples.jenatton_small")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)

    space = AddTreeSpace.from_dict(jenatton_small_spec())

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
        Ys.append(-y)  # negate for maximisation
        record = {"iteration": i + 1, "phase": "init", "params": p, "value": y}
        history.append(record)
        logger.info(
            "iter %3d (init):  y=%.4f (best=%.4f)",
            i + 1,
            y,
            -max(Ys),
        )
        if output_dir is not None:
            with open(os.path.join(output_dir, f"iter_{i+1}.json"), "w") as f:
                json.dump(record, f)

    # ---------- BO loop ----------
    for i in range(n_init, n_iter):
        train_X = torch.stack(Xs)
        train_Y = torch.tensor(Ys, dtype=torch.float64).unsqueeze(-1)

        model = AddTreeGP(space, train_X, train_Y)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

        candidates, _ = optimize_addtree_acqf(
            model,
            beta=beta,
            q=1,
            num_restarts=2,
            raw_samples=128,
        )
        x_new = candidates[0]
        p = decode(space, x_new)
        y = obj_func(p)
        Xs.append(x_new)
        Ys.append(-y)

        record = {"iteration": i + 1, "phase": "bo", "params": p, "value": y}
        history.append(record)
        logger.info(
            "iter %3d (bo):  y=%.4f (best=%.4f)",
            i + 1,
            y,
            -max(Ys),
        )
        if output_dir is not None:
            with open(os.path.join(output_dir, f"iter_{i+1}.json"), "w") as f:
                json.dump(record, f)

    best_y = -max(Ys)
    summary = {
        "best_value": best_y,
        "best_iteration": int(np.argmin([h["value"] for h in history])) + 1,
        "n_init": n_init,
        "n_iter": n_iter,
        "history": history,
    }
    logger.info("FINAL best_value = %.4f (optimum = 0.1)", best_y)
    return summary


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_init", type=int, default=10)
    p.add_argument("--n_iter", type=int, default=100)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()
    main(**vars(args))


if __name__ == "__main__":
    _cli()
