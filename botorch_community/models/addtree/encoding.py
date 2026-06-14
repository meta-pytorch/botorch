# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Encoding / decoding between user-facing parameter dicts and BFS tensors.

Two formats:

1. **User dict** -- a plain ``dict[str, str | float]`` keyed by:

   * ``"<choice_name>"``  -> ``str`` option name (e.g. ``"left"``)
   * ``"<node_name>.<continuous_name>"`` -> ``float``

   The dict only needs to contain entries for nodes / choices that are on
   the path (off-path parameters are implicit).

2. **BFS tensor** -- a 1-d ``torch.Tensor`` of length
   :attr:`AddTreeSpace.dim`, with ``-1`` (the sentinel) in off-path slots,
   the integer ``local_id`` in on-path categorical-flag slots, and the
   actual value in on-path continuous slots.

Round-trip:

>>> space.decode(space.encode(p)) == p   # for any feasible ``p``

Contributor: maxc01
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from botorch_community.models.addtree.space import AddTreeSpace
from torch import Tensor


__all__ = ["encode", "decode", "param_key"]


def param_key(node_name: str, cont_name: str) -> str:
    """The user-dict key for a continuous parameter.

    By convention the key is ``f"{node_name}.{cont_name}"``.
    """
    return f"{node_name}.{cont_name}"


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


def encode(
    space: AddTreeSpace,
    params: Mapping[str, Any],
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    r"""Encode a user-facing parameter dict into a BFS tensor.

    Args:
        space: The :class:`AddTreeSpace` defining the encoding.
        params: A mapping with entries ``"<choice_name>" -> "<option_name>"``
            for every choice the path traverses, and
            ``"<node_name>.<continuous_name>" -> float`` for every
            continuous parameter on that path. Off-path entries are
            ignored if present (so users can pass over-specified dicts).
        dtype: Output tensor dtype. Defaults to ``torch.float64``.
        device: Output tensor device. Defaults to CPU.

    Returns:
        A ``(space.dim,)`` tensor in BFS encoding.

    Raises:
        KeyError: if a required choice or continuous parameter is missing.
        ValueError: if a choice value is not a known option, or a
            continuous value is outside ``[lo, hi]``.
    """
    out = torch.full((space.dim,), space.SENTINEL, dtype=dtype, device=device)

    # Walk from root, following the user's choice picks.
    rec = space._all_records[0]
    out[rec.flag_index] = float(rec.local_id)
    _write_continuous(out, rec, params)

    while rec.node.choice is not None:
        choice = rec.node.choice
        if choice.name not in params:
            raise KeyError(
                f"Missing choice '{choice.name}' in params; "
                f"valid options: {[o for o, _ in choice.options]}."
            )
        chosen = params[choice.name]
        match_idx = None
        for opt_idx, (opt_name, _child) in enumerate(choice.options):
            if opt_name == chosen:
                match_idx = opt_idx
                break
        if match_idx is None:
            raise ValueError(
                f"Choice '{choice.name}': value {chosen!r} is not a valid "
                f"option (known: {[o for o, _ in choice.options]})."
            )
        child_node = choice.options[match_idx][1]
        rec = space._record_by_name(child_node.name)
        out[rec.flag_index] = float(rec.local_id)
        _write_continuous(out, rec, params)

    return out


def _write_continuous(out: Tensor, rec, params: Mapping[str, Any]) -> None:
    for ax, c in zip(rec.cont_indices, rec.node.continuous):
        key = param_key(rec.node.name, c.name)
        if key not in params:
            raise KeyError(
                f"Missing continuous parameter '{key}' in params for node "
                f"'{rec.node.name}'."
            )
        v = float(params[key])
        if not (c.lo - 1e-12 <= v <= c.hi + 1e-12):
            raise ValueError(
                f"Continuous '{key}': value {v} out of bounds " f"[{c.lo}, {c.hi}]."
            )
        out[ax] = v


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def decode(space: AddTreeSpace, x: Tensor) -> dict[str, Any]:
    r"""Decode a BFS tensor back to a user-facing parameter dict.

    Args:
        space: The :class:`AddTreeSpace` defining the encoding.
        x: A ``(space.dim,)`` tensor in BFS encoding (off-path slots ==
            ``space.SENTINEL``, on-path flag slots == ``local_id``,
            on-path continuous slots == data).

    Returns:
        A flat ``dict`` with keys ``"<choice_name>"`` for traversed
        choices and ``"<node_name>.<continuous_name>"`` for on-path
        continuous parameters.
    """
    if x.shape[-1] != space.dim:
        raise ValueError(
            f"Decode expects last-dim {space.dim}; got shape {tuple(x.shape)}."
        )
    if x.ndim > 1:
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError(
                "decode expects a 1-d tensor or a 1xdim batch; got "
                f"shape {tuple(x.shape)}."
            )
        x = x.squeeze(0)

    out: dict[str, Any] = {}
    # Walk from the root, reading the chosen option from each on-path
    # categorical-flag slot.
    rec = space._all_records[0]
    _read_continuous(out, rec, x)

    while rec.node.choice is not None:
        choice = rec.node.choice
        # Find which child has its flag set (i.e. != sentinel).
        chosen_child_idx: int | None = None
        chosen_option_name: str | None = None
        for opt_idx, (opt_name, child_node) in enumerate(choice.options):
            child_rec = space._record_by_name(child_node.name)
            if float(x[child_rec.flag_index]) != float(space.SENTINEL):
                if chosen_child_idx is not None:
                    raise ValueError(
                        f"Decoded path is ambiguous at choice '{choice.name}'"
                        f": multiple options have non-sentinel flags."
                    )
                chosen_child_idx = opt_idx
                chosen_option_name = opt_name
        if chosen_child_idx is None:
            # No child on path; tree must end here, but the choice
            # dictates that a child is expected. This is a malformed
            # encoding.
            raise ValueError(
                f"Decoded path stops short of choice '{choice.name}'"
                f": none of its option flags are set."
            )
        out[choice.name] = chosen_option_name
        rec = space._record_by_name(choice.options[chosen_child_idx][1].name)
        _read_continuous(out, rec, x)

    return out


def _read_continuous(out: dict[str, Any], rec, x: Tensor) -> None:
    for ax, c in zip(rec.cont_indices, rec.node.continuous):
        out[param_key(rec.node.name, c.name)] = float(x[ax])
