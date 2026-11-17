"""Compare two complete selector modules in the same CPU/CUDA environment.

The source files are snapshotted before loading, the protocol precedes timing,
and output equality is checked separately from performance. No kernels or
validation paths are bypassed. This script does not modify either module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import Tensor

CASES = ((1, 64, 2), (1, 256, 3), (32, 64, 3), (32, 256, 3), (32, 256, 5), (1, 1024, 3))
FUNCTIONS = ("knee_region", "select_preferred")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load selector source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for function in FUNCTIONS:
        if not callable(getattr(module, function, None)):
            raise TypeError(f"{path} does not define {function}")
    return module


def make_inputs(
    shape: tuple[int, int, int], dtype: torch.dtype, device: str, seed: int
):
    # Generate once in CPU double precision, then cast/transfer. CPU and GPU
    # receive identical numbers at a given dtype, independent of RNG backend.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    Y = torch.randn(shape, generator=generator, dtype=torch.double).to(dtype=dtype)
    p = (
        torch.rand((shape[0], shape[2]), generator=generator, dtype=torch.double) + 0.1
    ).to(dtype=dtype)
    fingerprint = hashlib.sha256(Y.numpy().tobytes() + p.numpy().tobytes()).hexdigest()
    return Y.to(device), p.to(device), fingerprint


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def measure(function: Any, Y: Tensor, p: Tensor, name: str, warmups: int, repeats: int):
    device = str(Y.device)

    def evaluate():
        if name == "knee_region":
            return function(Y, alpha=0.5, maximize=True)
        return function(Y, p, alpha=0.5, maximize=True)

    for _ in range(warmups):
        temporary = evaluate()
        synchronize(device)
        del temporary
    timings, peaks, extra_peaks = [], [], []
    result = None
    for _ in range(repeats):
        if result is not None:
            del result
            result = None
        synchronize(device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
            base = torch.cuda.memory_allocated(device)
        started = time.perf_counter()
        result = evaluate()
        synchronize(device)
        timings.append(time.perf_counter() - started)
        if device.startswith("cuda"):
            peak = torch.cuda.max_memory_allocated(device)
            peaks.append(peak)
            extra_peaks.append(peak - base)
    return result, {
        "seconds": timings,
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "cuda_peak_allocated_bytes": peaks,
        "cuda_peak_extra_allocated_bytes": extra_peaks,
        "cuda_peak_allocated_bytes_max": max(peaks) if peaks else None,
        "cuda_peak_extra_allocated_bytes_max": max(extra_peaks)
        if extra_peaks
        else None,
    }


def compare_outputs(before: Tensor, after: Tensor) -> dict[str, Any]:
    same_structure = (
        before.shape == after.shape
        and before.dtype == after.dtype
        and before.device == after.device
    )
    matches = int((before == after).sum()) if same_structure else 0
    return {
        "same_structure": same_structure,
        "matched_outputs": matches,
        "compared_outputs": before.numel(),
        "exact_match": same_structure and torch.equal(before, after),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def make_report(
    output: Path, records: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    selected = [r for r in records if r["function"] == "select_preferred"]
    matched = sum(r["matched_outputs"] for r in selected)
    compared = sum(r["compared_outputs"] for r in selected)
    all_exact = all(r["exact_match"] for r in records)
    settings = metadata["settings"]
    lines = [
        "# Selector before/after performance",
        "",
        (
            f"Full selector indices matched {matched}/{compared} comparisons. "
            f"Both indices and knee masks matched exactly in every case: {all_exact}."
        ),
        "",
        (
            f"Values below are medians of {settings['repeats']} timed calls after {settings['warmups']} warmup calls. "
            "Speedup is before/after: values below one mean the candidate is slower."
        ),
        "",
        "| Device | dtype | (batch, n, m) | Before ms | After ms | Speedup | Exact indices |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in selected:
        lines.append(
            f"| {row['device']} | {row['dtype']} | ({row['batch']}, {row['n']}, {row['m']}) "
            f"| {row['before_median_seconds'] * 1000:.4g} | {row['after_median_seconds'] * 1000:.4g} "
            f"| {row['speedup']:.3f}x | {row['exact_match']} |"
        )
    regressions = [r for r in selected if r["speedup"] < 1]
    lines.extend(
        [
            "",
            (
                f"The candidate was slower in {len(regressions)}/{len(selected)} full-selection configurations. "
                "No settings were tuned after observing these measurements. All per-function timings, ranges, "
                "repetitions and CUDA allocator peaks are preserved in timings.csv/timings.json."
            ),
            "",
            "## Protocol and limits",
            "",
            (
                f"- Seed {settings['seed']}; standard normal objective tensors; positive batched preferences; alpha=0.5; maximization. "
                "Each dtype/device pair receives identical CPU-generated input values. Source snapshots and hashes "
                "are saved beside the protocol before timing."
            ),
            (
                "- Both complete public functions run their original validation and Pareto processing. "
                "The candidate reuses installed BoTorch Pareto dispatch. Shapes that trigger its loop fallback "
                "can introduce CPU/GPU synchronization and regressions; none are hidden by alternative kernels."
            ),
            (
                f"- Both functions use the same interpreter, Torch/BoTorch versions, hardware and {metadata['torch_threads']} Torch CPU threads. "
                "Before/after order is counterbalanced deterministically across case/dtype/device/function. "
                "GPU execution is synchronized immediately before and after each measured call."
            ),
            (
                "- CUDA memory is peak live allocator bytes, plus the increase above the per-call live baseline. "
                "It is not total device/process usage or reserved cache memory. CPU memory is not measured."
            ),
            (
                f"- {settings['repeats']} repetitions on one machine give descriptive ranges, not confidence intervals or a "
                "cross-hardware speed claim. Import/context initialization, input construction, copies, "
                "equality checks and serialization are outside the timing. A warmup does not eliminate every "
                "effect of GPU clocking or background activity."
            ),
            (
                "- Random continuous inputs avoid deliberately introduced ties. Exact mismatches, if any, remain "
                "visible; this benchmark does not silently accept near-threshold differences."
            ),
            "",
            (
                f"Environment: Python {metadata['python']}, Torch {metadata['versions']['torch']}, "
                f"BoTorch {metadata['versions']['botorch']}; CUDA device {metadata.get('cuda_device')}. "
                "See provenance.json for source hashes and full environment information."
            ),
            "",
            "Observed conditions: " + metadata["conditions"],
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"]
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float32", "float64"],
        default=["float32", "float64"],
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--conditions",
        default="Background workload and machine isolation were not controlled.",
    )
    args = parser.parse_args()
    if min(args.warmups, args.repeats, args.threads) < 1 or args.seed < 0:
        parser.error(
            "warmups/repeats/threads must be positive; seed must be nonnegative"
        )
    if "cuda" in args.devices and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable in this interpreter")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    output = args.output_dir.resolve()
    if (output / "protocol.json").exists():
        raise FileExistsError(
            "Use a new output directory to preserve the recorded protocol"
        )
    source = output / "source"
    source.mkdir(parents=True, exist_ok=True)
    for origin, target in (
        (args.before, source / "before.py"),
        (args.after, source / "after.py"),
        (Path(__file__), source / "benchmark_selector.py"),
    ):
        shutil.copyfile(origin, target)
    write_json(
        output / "protocol.json",
        {
            "recorded_utc": datetime.now(UTC).isoformat(),
            "cases": CASES,
            "functions": FUNCTIONS,
            "seed": args.seed,
            "devices": args.devices,
            "dtypes": args.dtypes,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "threads": args.threads,
            "alpha": 0.5,
            "maximize": True,
            "timing": "synchronized perf_counter, full public functions",
            "input_generation": "CPU float64 normal Y and uniform+0.1 batched preferences, cast/transfer before timing",
            "order": "before/after counterbalanced by case+dtype+device+function index parity",
            "conditions": args.conditions,
            "command": subprocess_command(),
        },
    )
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": {
            p: importlib.metadata.version(p) for p in ("torch", "botorch", "numpy")
        },
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name()
        if torch.cuda.is_available()
        else None,
        "original_sources": {
            "before": str(args.before.resolve()),
            "after": str(args.after.resolve()),
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.glob("*.py"))
        },
        "settings": {
            "seed": args.seed,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "conditions": args.conditions,
    }
    write_json(output / "provenance.json", metadata)
    modules = {
        "before": load_module(source / "before.py", "selector_before_benchmark"),
        "after": load_module(source / "after.py", "selector_after_benchmark"),
    }
    records, raw = [], []
    for case_index, shape in enumerate(CASES):
        for dtype_index, dtype_name in enumerate(args.dtypes):
            for device_index, device in enumerate(args.devices):
                Y, p, fingerprint = make_inputs(
                    shape, getattr(torch, dtype_name), device, args.seed
                )
                synchronize(device)
                for function_index, name in enumerate(FUNCTIONS):
                    order = (
                        ("before", "after")
                        if (case_index + dtype_index + device_index + function_index)
                        % 2
                        == 0
                        else ("after", "before")
                    )
                    results, metrics = {}, {}
                    for implementation in order:
                        results[implementation], metrics[implementation] = measure(
                            getattr(modules[implementation], name),
                            Y,
                            p,
                            name,
                            args.warmups,
                            args.repeats,
                        )
                    equality = compare_outputs(results["before"], results["after"])
                    record = {
                        "batch": shape[0],
                        "n": shape[1],
                        "m": shape[2],
                        "dtype": dtype_name,
                        "device": device,
                        "function": name,
                        "order": ",".join(order),
                        "input_sha256": fingerprint,
                    }
                    for implementation in ("before", "after"):
                        record.update(
                            {
                                implementation + "_" + key: value
                                for key, value in metrics[implementation].items()
                                if not isinstance(value, list)
                            }
                        )
                    record.update(equality)
                    record["speedup"] = (
                        metrics["before"]["median_seconds"]
                        / metrics["after"]["median_seconds"]
                    )
                    records.append(record)
                    raw.append({**record, "repetitions": metrics})
                    print(json.dumps(record), flush=True)
                    write_json(output / "timings.json", raw)
    write_csv(output / "timings.csv", records)
    make_report(output, records, metadata)
    print(
        f"Saved {len(records)} paired function configurations to {output}", flush=True
    )


def subprocess_command() -> str:
    import subprocess

    return subprocess.list2cmdline([sys.executable, *sys.argv])


if __name__ == "__main__":
    main()
