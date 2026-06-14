# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Declarative spec for AddTree parameter spaces.

The user describes the conditional parameter space as a tree: each node
optionally owns a list of continuous parameters; each node may also have
exactly one categorical *choice* whose options branch into child nodes.

A spec is a plain Python dict (or YAML file) with the following shape:

.. code-block:: yaml

    name: root
    continuous: []                    # optional, list of {name, lo, hi}
    choices:                          # optional; at most one entry in v2
      - name: L1
        options:
          left:
            name: L1_left
            continuous: [{name: cont_value, lo: 0.0, hi: 1.0}]
            choices:
              - name: L2
                options:
                  left:
                    name: L1_left.L2_left
                    continuous: [{name: cont_value, lo: 0.0, hi: 1.0}]
                  right:
                    name: L1_left.L2_right
                    continuous: [{name: cont_value, lo: 0.0, hi: 1.0}]
          right:
            name: L1_right
            ...

From this declaration the library computes:

* ``space.dim``             — total length of the BFS-encoded vector
* ``space.bounds``          — ``(2, dim)`` tensor of lower/upper bounds
* ``space.cat_dims``        — indices of the categorical (flag) slots
* ``space.cont_dims``       — indices of the continuous data slots
* ``space.fixed_features_list`` — one ``dict[int, float]`` per leaf path,
  ready to pass to :func:`botorch.optim.optimize_acqf_mixed`.
* ``space.path_ids``        — string ids matching the fixed-features list

There is **no** module-level state; multiple :class:`AddTreeSpace` objects
can coexist freely.

Contributor: maxc01
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


__all__ = [
    "AddTreeSpace",
    "Continuous",
    "Choice",
    "Node",
]


# ---------------------------------------------------------------------------
# Public dataclasses (the immutable spec, post-validation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Continuous:
    """A scalar continuous parameter with hard ``[lo, hi]`` bounds."""

    name: str
    lo: float = 0.0
    hi: float = 1.0

    def __post_init__(self) -> None:
        """Validate bounds."""
        if not self.lo < self.hi:
            raise ValueError(
                f"Continuous '{self.name}': lo ({self.lo}) must be < hi "
                f"({self.hi})."
            )


@dataclass(frozen=True)
class Choice:
    """A categorical decision branching into named child nodes."""

    name: str
    options: tuple[tuple[str, "Node"], ...]
    """Ordered ``(option_name, child_node)`` pairs.

    Order determines the integer ``local_id`` used in the BFS encoding,
    so two specs that differ only in option order produce different
    (but equivalent) wire formats. Reorder deliberately.
    """

    def __post_init__(self) -> None:
        """Validate options."""
        if len(self.options) < 1:
            raise ValueError(f"Choice '{self.name}' must have at least one option.")
        seen = set()
        for opt_name, _ in self.options:
            if opt_name in seen:
                raise ValueError(
                    f"Choice '{self.name}' has duplicate option '{opt_name}'."
                )
            seen.add(opt_name)


@dataclass(frozen=True)
class Node:
    """A node in the AddTree.

    Each node owns:

    * an unique ``name`` (used as a key in encoded/decoded dicts),
    * an optional list of :class:`Continuous` parameters,
    * an optional :class:`Choice` (a single categorical decision).

    A node with neither continuous parameters nor a choice is a leaf
    that contributes only its categorical flag to the kernel.
    """

    name: str
    continuous: tuple[Continuous, ...] = ()
    choice: Choice | None = None

    def __post_init__(self) -> None:
        """Validate parameter names."""
        seen = set()
        for c in self.continuous:
            if c.name in seen:
                raise ValueError(
                    f"Node '{self.name}': duplicate continuous parameter "
                    f"'{c.name}'."
                )
            seen.add(c.name)


# ---------------------------------------------------------------------------
# Parsing helpers (dict / yaml -> Node)
# ---------------------------------------------------------------------------


def _node_from_dict(d: Mapping[str, Any]) -> Node:
    """Recursively build a :class:`Node` tree from a plain dict."""
    if "name" not in d:
        raise ValueError(f"Node spec missing 'name': {d!r}")
    name = str(d["name"])

    continuous: list[Continuous] = []
    for c in d.get("continuous", []) or []:
        continuous.append(
            Continuous(name=str(c["name"]), lo=float(c["lo"]), hi=float(c["hi"]))
        )

    choice: Choice | None = None
    raw_choices = d.get("choices") or []
    if len(raw_choices) > 1:
        raise ValueError(
            f"Node '{name}' has {len(raw_choices)} choices; v2 supports at most "
            "one choice per node. Use intermediate nodes to model multi-choice "
            "structures."
        )
    if raw_choices:
        rc = raw_choices[0]
        if "name" not in rc or "options" not in rc:
            raise ValueError(
                f"Node '{name}': choice must have 'name' and 'options' keys."
            )
        # Preserve the user-provided option order. Python dicts preserve
        # insertion order in 3.7+, and our YAML loader uses a SafeLoader
        # which also preserves order.
        opts: list[tuple[str, Node]] = []
        for opt_name, opt_spec in rc["options"].items():
            opts.append((str(opt_name), _node_from_dict(opt_spec)))
        choice = Choice(name=str(rc["name"]), options=tuple(opts))

    return Node(name=name, continuous=tuple(continuous), choice=choice)


# ---------------------------------------------------------------------------
# AddTreeSpace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NodeRecord:
    """Internal: BFS-ordered record describing one node's slots."""

    node: Node
    flag_index: int
    """BFS slot holding the categorical flag (== ``local_id`` if this node
    is on the path, else the sentinel ``-1.0``).
    """
    cont_indices: tuple[int, ...]
    """BFS slots holding the continuous data."""
    local_id: int
    """The integer choice value among siblings (0 for root)."""
    parent_record_index: int | None
    """Index into ``AddTreeSpace.records`` of the parent node, or ``None``
    for the root.
    """


@dataclass(frozen=True)
class _PathRecord:
    """Internal: a root-to-leaf path."""

    path_id: str
    """String of option names separated by '/' (excluding the root)."""
    node_indices: tuple[int, ...]
    """Indices into ``AddTreeSpace.records`` of nodes on the path."""


class AddTreeSpace:
    r"""An immutable conditional parameter space for AddTree GP models.

    Construct from a nested dict, a YAML file, or by hand from
    :class:`Node` / :class:`Choice` / :class:`Continuous` instances.

    Examples:
        >>> space = AddTreeSpace.from_dict({
        ...     "name": "root",
        ...     "choices": [{
        ...         "name": "L1",
        ...         "options": {
        ...             "left":  {"name": "L1_left",
        ...                       "continuous": [{"name": "v", "lo": 0., "hi": 1.}]},
        ...             "right": {"name": "L1_right",
        ...                       "continuous": [{"name": "v", "lo": 0., "hi": 1.}]},
        ...         },
        ...     }],
        ... })
        >>> space.dim
        4
        >>> space.path_ids
        ('left', 'right')
    """

    SENTINEL: float = -1.0
    """Slot value used to mark "this node is not on the encoded path"."""

    def __init__(self, root: Node):
        """Build the immutable spec from a finalised :class:`Node` tree."""
        if not isinstance(root, Node):
            raise TypeError(
                f"AddTreeSpace expects a Node root; got {type(root).__name__}."
            )
        self._root = root
        records, paths, name_to_record_index = _bfs_layout(root)
        self._records: tuple[_NodeRecord, ...] = records
        self._paths: tuple[_PathRecord, ...] = paths
        self._name_to_record_index: dict[str, int] = name_to_record_index
        self._dim: int = sum(1 + len(r.node.continuous) for r in records)

        # Pre-compute bounds, cat/cont dims, fixed_features_list.
        import torch

        # Build the (2, dim) bounds tensor used by optimize_acqf_mixed.
        # Every slot pinned by fixed_features will be overridden, but the
        # bounds still need ``bounds[0] < bounds[1]`` everywhere so that
        # the Sobol-based initial-condition generator is well-defined.
        # We follow the upstream-AddTree convention of using ``[0, 1]``
        # for *every* slot and rely on ``fixed_features`` to pin the
        # off-path / flag slots to the correct value at solve time.
        bounds = torch.empty(2, self._dim, dtype=torch.float64)
        bounds[0, :] = 0.0
        bounds[1, :] = 1.0
        cat_dims: list[int] = []
        cont_dims: list[int] = []
        for rec in records:
            cat_dims.append(rec.flag_index)
            for ax, c in zip(rec.cont_indices, rec.node.continuous):
                bounds[0, ax] = c.lo
                bounds[1, ax] = c.hi
                cont_dims.append(ax)
        self._bounds: torch.Tensor = bounds
        self._cat_dims: tuple[int, ...] = tuple(cat_dims)
        self._cont_dims: tuple[int, ...] = tuple(cont_dims)

        # fixed_features_list: one dict per leaf path. For each path, pin
        # off-path slots to SENTINEL, on-path flag slots to their local_id,
        # and leave on-path continuous slots unpinned.
        ff_list: list[dict[int, float]] = []
        for path in paths:
            on_path_flags: dict[int, float] = {}
            on_path_cont: set[int] = set()
            for ri in path.node_indices:
                rec = records[ri]
                on_path_flags[rec.flag_index] = float(rec.local_id)
                on_path_cont.update(rec.cont_indices)
            ff: dict[int, float] = {}
            for i in range(self._dim):
                if i in on_path_cont:
                    continue  # let the optimiser choose
                if i in on_path_flags:
                    ff[i] = on_path_flags[i]
                else:
                    ff[i] = self.SENTINEL
            ff_list.append(ff)
        self._fixed_features_list: tuple[dict[int, float], ...] = tuple(ff_list)

    # ---- alternative constructors -----------------------------------------

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AddTreeSpace":
        """Build an :class:`AddTreeSpace` from a nested mapping."""
        return cls(_node_from_dict(d))

    @classmethod
    def from_yaml(cls, path: str) -> "AddTreeSpace":
        """Build an :class:`AddTreeSpace` from a YAML file path.

        Requires the optional ``pyyaml`` dependency.
        """
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "AddTreeSpace.from_yaml requires PyYAML; install it via "
                "`pip install pyyaml`."
            ) from e
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    # ---- public properties (immutable views) ------------------------------

    @property
    def root(self) -> Node:
        """The root :class:`Node` of the spec."""
        return self._root

    @property
    def dim(self) -> int:
        """Total length of the BFS-encoded vector."""
        return self._dim

    @property
    def bounds(self):
        """``(2, dim)`` tensor of lower/upper bounds for ``optimize_acqf*``."""
        return self._bounds

    @property
    def cat_dims(self) -> tuple[int, ...]:
        """Slot indices of the categorical flags."""
        return self._cat_dims

    @property
    def cont_dims(self) -> tuple[int, ...]:
        """Slot indices of the continuous data."""
        return self._cont_dims

    @property
    def fixed_features_list(self) -> tuple[dict[int, float], ...]:
        """One dict per leaf path; pass to ``optimize_acqf_mixed``."""
        # Return shallow copies so callers can mutate the dicts.
        return tuple(dict(ff) for ff in self._fixed_features_list)

    @property
    def path_ids(self) -> tuple[str, ...]:
        """The path id strings, in the same order as ``fixed_features_list``."""
        return tuple(p.path_id for p in self._paths)

    @property
    def num_paths(self) -> int:
        """Total number of root-to-leaf paths."""
        return len(self._paths)

    # ---- internal accessors used by encoding & kernel modules -------------

    def _records_for_path(self, path_id: str) -> tuple[_NodeRecord, ...]:
        """Return the BFS records visited by ``path_id``, root first."""
        for p in self._paths:
            if p.path_id == path_id:
                return tuple(self._records[i] for i in p.node_indices)
        raise KeyError(f"Unknown path_id {path_id!r}; valid: {self.path_ids}")

    def _record_by_name(self, node_name: str) -> _NodeRecord:
        if node_name not in self._name_to_record_index:
            raise KeyError(f"Unknown node name {node_name!r}.")
        return self._records[self._name_to_record_index[node_name]]

    @property
    def _all_records(self) -> tuple[_NodeRecord, ...]:
        return self._records

    def __repr__(self) -> str:
        return (
            f"AddTreeSpace(root={self._root.name!r}, dim={self._dim}, "
            f"num_paths={self.num_paths})"
        )


# ---------------------------------------------------------------------------
# BFS layout
# ---------------------------------------------------------------------------


def _bfs_layout(
    root: Node,
) -> tuple[tuple[_NodeRecord, ...], tuple[_PathRecord, ...], dict[str, int]]:
    """Compute BFS-ordered slot indices for every node + enumerate paths.

    Returns:
        records: BFS-ordered tuple of :class:`_NodeRecord`s.
        paths:   tuple of :class:`_PathRecord`, one per leaf.
        name_to_index: name -> index into ``records``.
    """
    # ---- BFS over (node, parent_record_index, local_id) ----
    queue: list[tuple[Node, int | None, int]] = [(root, None, 0)]
    records: list[_NodeRecord] = []
    name_to_index: dict[str, int] = {}
    next_slot = 0
    while queue:
        node, parent_idx, local_id = queue.pop(0)
        if node.name in name_to_index:
            raise ValueError(f"Duplicate node name {node.name!r} in AddTree spec.")
        flag_index = next_slot
        cont_indices = tuple(range(next_slot + 1, next_slot + 1 + len(node.continuous)))
        next_slot += 1 + len(node.continuous)
        rec = _NodeRecord(
            node=node,
            flag_index=flag_index,
            cont_indices=cont_indices,
            local_id=local_id,
            parent_record_index=parent_idx,
        )
        rec_idx = len(records)
        records.append(rec)
        name_to_index[node.name] = rec_idx

        if node.choice is not None:
            for opt_local_id, (_opt_name, child) in enumerate(node.choice.options):
                queue.append((child, rec_idx, opt_local_id))

    # ---- DFS to enumerate root-to-leaf paths ----
    paths: list[_PathRecord] = []

    def _walk(
        rec_idx: int, prefix_record_indices: list[int], prefix_option_names: list[str]
    ) -> None:
        prefix_record_indices = prefix_record_indices + [rec_idx]
        rec = records[rec_idx]
        if rec.node.choice is None:
            path_id = "/".join(prefix_option_names)
            paths.append(
                _PathRecord(
                    path_id=path_id,
                    node_indices=tuple(prefix_record_indices),
                )
            )
            return
        for opt_local_id, (opt_name, _child) in enumerate(rec.node.choice.options):
            # Find the child's record index. It was inserted into ``records``
            # in BFS order; the children are not contiguous, so look up by
            # name.
            child = rec.node.choice.options[opt_local_id][1]
            child_idx = name_to_index[child.name]
            _walk(child_idx, prefix_record_indices, prefix_option_names + [opt_name])

    _walk(0, [], [])
    return tuple(records), tuple(paths), name_to_index


# ---------------------------------------------------------------------------
# Sequence helpers (re-exports for convenience builder use)
# ---------------------------------------------------------------------------


def _coerce_choice(c: Choice | Mapping[str, Any]) -> Choice:
    """Allow Choice() instances and dicts side-by-side in the Python builder."""
    if isinstance(c, Choice):
        return c
    options: list[tuple[str, Node]] = []
    for opt_name, opt_node in c["options"].items():
        if isinstance(opt_node, Node):
            options.append((str(opt_name), opt_node))
        else:
            options.append((str(opt_name), _node_from_dict(opt_node)))
    return Choice(name=str(c["name"]), options=tuple(options))


def make_node(
    name: str,
    *,
    continuous: Sequence[Continuous] | Sequence[Mapping[str, Any]] = (),
    choice: Choice | Mapping[str, Any] | None = None,
) -> Node:
    """Convenience builder for :class:`Node`.

    Accepts both already-typed dataclasses and raw dicts/mappings, so
    Python builder code and YAML-derived dicts compose freely.
    """
    cont: list[Continuous] = []
    for c in continuous:
        if isinstance(c, Continuous):
            cont.append(c)
        else:
            cont.append(
                Continuous(name=str(c["name"]), lo=float(c["lo"]), hi=float(c["hi"]))
            )
    ch: Choice | None = None
    if choice is not None:
        ch = _coerce_choice(choice)
    return Node(name=name, continuous=tuple(cont), choice=ch)
