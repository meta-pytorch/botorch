# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Content-addressed cache for pre-trained priors.

WHY
---
Pre-training dominates the cost of this study. A single LCBench cell spends ~61 min in
HyperBO pre-training alone (measured: `hyperbo_iters=25000` -> 3662 s, vs 35 s at the
default), before a single BO iteration runs. Stage 1 re-pays that for every OFAT cell,
even though most factors change only the EM side or the BO loop, not the HyperBO prior.

Caching pre-trained priors buys three distinct things:

1. **Reuse.** Cells that share a pre-training configuration pay for it once.
2. **Post-hoc re-analysis.** BO loops and surrogate variants can be re-run against a
   frozen prior without repeating the expensive fit, so new variants become cheap.
3. **Variance of the fit itself.** The seed is part of the key, so repeated
   pre-training under different seeds produces *separate retained entries* rather than
   overwriting. That turns "how much does the trained model vary?" into a question the
   cache can answer directly, instead of a quantity we assume away.

SAFETY
------
A mis-keyed cache silently substitutes the wrong prior into every downstream number,
which is the worst failure mode this project has (nine silent no-ops and counting). Two
defences, both load-bearing:

* **The key is derived from the DATA, not only from flags.** `data_digest` hashes the
  actual pre-training tensors. Forgetting to enumerate some flag that changes the data
  therefore cannot produce a false hit -- the digest changes anyway. Flags are folded in
  on top for things that change the *fit* without changing the data.
* **Verification on load, not trust.** The full config is stored beside the payload and
  compared field-by-field on read. A hash collision, a truncated write, or a stale entry
  from an older code version is rejected and recomputed rather than used.

The cache is therefore safe to delete at any time: it is a pure accelerator, and a miss
is always correct.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from typing import Any, Callable

import torch

logger: logging.Logger = logging.getLogger(__name__)

# Bump when the meaning of a cached payload changes in a way the config cannot express
# (e.g. a bug fix inside pretrain_hyperbo). Every existing entry is then a miss.
CACHE_FORMAT_VERSION = 1


def tensor_digest(obj: Any, _depth: int = 0) -> str:
    """Stable content hash of tensors / datasets / nested containers.

    This is the primary key material: it makes the cache correct by construction even if
    the caller forgets to declare a flag, because anything that changes the pre-training
    inputs changes the digest.
    """
    h = hashlib.sha256()

    def _feed(x: Any, depth: int) -> None:
        if depth > 8:
            raise ValueError("tensor_digest: structure nested too deeply to hash")
        if torch.is_tensor(x):
            t = x.detach().to("cpu")
            h.update(str(tuple(t.shape)).encode())
            h.update(str(t.dtype).encode())
            # float64 view of the raw bytes keeps this exact and dtype-stable
            h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(x, (list, tuple)):
            h.update(f"seq{len(x)}".encode())
            for v in x:
                _feed(v, depth + 1)
        elif isinstance(x, dict):
            h.update(f"map{len(x)}".encode())
            for k in sorted(x):
                h.update(str(k).encode())
                _feed(x[k], depth + 1)
        elif hasattr(x, "X") and hasattr(x, "Y"):  # SupervisedDataset-like
            _feed(x.X, depth + 1)
            _feed(x.Y, depth + 1)
        elif x is None or isinstance(x, (str, int, float, bool)):
            h.update(repr(x).encode())
        else:
            raise TypeError(
                f"tensor_digest cannot hash {type(x).__name__}. Add support rather "
                "than silently excluding it from the key -- an unhashed input is "
                "exactly how a cache returns the wrong prior."
            )

    _feed(obj, _depth)
    return h.hexdigest()


def fingerprint(kind: str, data_digest: str, params: dict[str, Any]) -> str:
    """Cache key: what is being fitted, on what data, with which hyperparameters."""
    payload = {
        "v": CACHE_FORMAT_VERSION,
        "kind": kind,
        "data": data_digest,
        "params": {k: params[k] for k in sorted(params)},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _config_of(kind: str, data_digest: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "kind": kind,
        "data_digest": data_digest,
        "params": {k: params[k] for k in sorted(params)},
    }


def _path(cache_dir: str, kind: str, fp: str) -> str:
    return os.path.join(cache_dir, kind, f"{fp}.pt")


def load(
    cache_dir: str, kind: str, data_digest: str, params: dict[str, Any]
) -> Any | None:
    """Return the cached payload, or None on any doubt whatsoever.

    Every failure path returns None (a miss recomputes and is always correct) rather
    than raising, EXCEPT a config mismatch, which is logged loudly because it means the
    key is not capturing something it should.
    """
    fp = fingerprint(kind, data_digest, params)
    path = _path(cache_dir, kind, fp)
    if not os.path.exists(path):
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # truncated write, torch version skew, ...
        logger.warning("prior cache: unreadable entry %s (%r); recomputing", path, exc)
        return None

    want = _config_of(kind, data_digest, params)
    got = blob.get("config")
    if got != want:
        # Same key, different config => the key is incomplete or collided. Never use it.
        diff = [
            k for k in set(want) | set(got or {}) if (got or {}).get(k) != want.get(k)
        ]
        logger.error(
            "prior cache: CONFIG MISMATCH at %s (differing: %s). Refusing the entry "
            "and recomputing. This means the fingerprint is not capturing something it "
            "should -- investigate rather than ignoring.",
            path,
            diff,
        )
        return None
    return blob.get("payload")


def save(
    cache_dir: str, kind: str, data_digest: str, params: dict[str, Any], payload: Any
) -> str:
    """Write atomically so a crash mid-write cannot leave a half-entry to be loaded."""
    fp = fingerprint(kind, data_digest, params)
    path = _path(cache_dir, kind, fp)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = {"config": _config_of(kind, data_digest, params), "payload": payload}
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(blob, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def cached(
    cache_dir: str | None,
    mode: str,
    kind: str,
    data: Any,
    params: dict[str, Any],
    compute: Callable[[], Any],
    stats: dict[str, Any] | None = None,
) -> Any:
    """Get-or-compute. ``mode`` is one of off / read / write / readwrite.

    ``read`` without ``write`` is useful for reproducing a previous run exactly: a miss
    recomputes but does not pollute the cache.

    ``stats``, if given, is populated with ``hit`` (bool) and ``compute_s`` (float,
    0.0 on a hit). Pass it whenever the caller records a pre-training cost. Timing the
    wrapper instead of the compute is why every non-``base`` cell on disk reports
    ``hyperbo_pretrain_s = 0.0``: those are cache hits, not free fits, and a cost
    analysis that averages them silently concludes pre-training is free (S30.1).
    """
    if stats is not None:
        stats["hit"] = False
        stats["compute_s"] = 0.0

    def _compute_timed() -> Any:
        t0 = time.time()
        out = compute()
        if stats is not None:
            stats["compute_s"] = time.time() - t0
        return out

    if not cache_dir or mode == "off":
        return _compute_timed()
    digest = tensor_digest(data)
    if mode in ("read", "readwrite"):
        hit = load(cache_dir, kind, digest, params)
        if hit is not None:
            logger.info(
                "prior cache HIT  %s %s", kind, fingerprint(kind, digest, params)
            )
            if stats is not None:
                stats["hit"] = True
            return hit
    out = _compute_timed()
    if mode in ("write", "readwrite") and out is not None:
        p = save(cache_dir, kind, digest, params, out)
        logger.info("prior cache WRITE %s -> %s", kind, p)
    return out


def describe(cache_dir: str) -> list[dict[str, Any]]:
    """Inventory of what has been fitted, for post-hoc analysis.

    This is what makes the seed-in-the-key useful: group by everything except the seed
    to recover the set of independent fits of the same model, then compare them.
    """
    out: list[dict[str, Any]] = []
    for root, _dirs, files in os.walk(cache_dir):
        for f in files:
            if not f.endswith(".pt"):
                continue
            path = os.path.join(root, f)
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                continue
            cfg = blob.get("config", {})
            out.append(
                {
                    "path": path,
                    "kind": cfg.get("kind"),
                    "data_digest": cfg.get("data_digest"),
                    **{f"param.{k}": v for k, v in (cfg.get("params") or {}).items()},
                }
            )
    return out
