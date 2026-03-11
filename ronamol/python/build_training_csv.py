#!/usr/bin/env python3
"""
Build a Ronamol training CSV by joining:

1) gem5 sweep results (e.g. sweep_results.csv): provides `cpi` + uarch params + benchmark name
2) per-benchmark ronamol program features: <traces>/<benchmark>/ronamol/program_features.json
3) per-benchmark cache latency summary:   <traces>/<benchmark>/ronamol/cache_latency_summary.csv
   matched by sweep columns (l1i_size,l1d_size,l2_size) → (l1i_kb,l1d_kb,l2_kb)

Output: one row per sweep row with appended program + cache-lat columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMG]i?B)\s*$", re.IGNORECASE)


def _parse_size_to_kb(value: str) -> int:
    """
    Parse sizes like "128KiB", "4MiB" into integer KB.
    Falls back to int(value) assuming it's already KB.
    """
    s = str(value).strip()
    m = _SIZE_RE.fullmatch(s)
    if not m:
        return int(float(s))
    num = float(m.group(1))
    unit = m.group(2).upper()
    if unit.startswith("K"):
        return int(num)
    if unit.startswith("M"):
        return int(num * 1024)
    if unit.startswith("G"):
        return int(num * 1024 * 1024)
    raise ValueError(f"Unrecognized size unit: {value!r}")


@lru_cache(maxsize=None)
def _load_program_features(traces_root: str, benchmark: str) -> Dict[str, Any]:
    p = Path(traces_root) / benchmark / "ronamol" / "program_features.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing program features for {benchmark!r}: {p}")
    return json.loads(p.read_text())


@lru_cache(maxsize=None)
def _load_bp_rates(traces_root: str, benchmark: str) -> Dict[str, float]:
    """
    Load branch predictor misprediction rates for a benchmark.

    Expected JSON format at: <traces_root>/<benchmark>/trace_bp.json
      {"local": <float>, "tage": <float>}
    """
    p = Path(traces_root) / benchmark / "trace_bp.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing trace_bp.json for {benchmark!r}: {p}")
    data = json.loads(p.read_text())
    return {
        "local": float(data.get("local", 0.0)),
        "tage": float(data.get("tage", 0.0)),
    }


@lru_cache(maxsize=None)
def _load_cache_latency_index(
    traces_root: str, benchmark: str
) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    """
    Return dict keyed by (l1i_kb,l1d_kb,l2_kb) to the cache latency feature row.
    """
    p = Path(traces_root) / benchmark / "ronamol" / "cache_latency_summary.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing cache latency summary for {benchmark!r}: {p}")

    idx: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    with p.open(newline="") as f:
        r = csv.DictReader(f)
        required = {"l1i_kb", "l1d_kb", "l2_kb"}
        if not required.issubset(set(r.fieldnames or [])):
            raise ValueError(f"{p} missing required columns {sorted(required)}")
        for row in r:
            key = (int(row["l1i_kb"]), int(row["l1d_kb"]), int(row["l2_kb"]))
            idx[key] = row
    return idx


def _iter_sweep_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-results", required=True, help="Path to sweep_results.csv")
    ap.add_argument(
        "--traces-root",
        default="traces",
        help="Root containing per-benchmark trace folders (default: traces)",
    )
    ap.add_argument("-o", "--out", required=True, help="Output CSV path")
    ap.add_argument(
        "--program-prefix",
        default="prog_",
        help="Prefix for program feature columns (default: prog_). Use '' for none.",
    )
    ap.add_argument(
        "--cache-prefix",
        default="cache_",
        help="Prefix for cache latency columns (default: cache_). Use '' for none.",
    )
    args = ap.parse_args()

    sweep_path = Path(args.sweep_results)
    traces_root = Path(args.traces_root)
    out_path = Path(args.out)

    # First pass: discover output columns (keeps ordering stable).
    first_row = None
    for row in _iter_sweep_rows(sweep_path):
        first_row = row
        break
    if first_row is None:
        raise ValueError(f"Empty sweep CSV: {sweep_path}")

    bench0 = first_row.get("benchmark")
    if not bench0:
        raise ValueError("sweep_results.csv missing required column: benchmark")

    prog0 = _load_program_features(str(traces_root), bench0)
    cache0 = _load_cache_latency_index(str(traces_root), bench0)
    cache_example = next(iter(cache0.values())) if cache0 else {}

    # Drop branch_predictor from output; we replace it with a numeric misprediction_rate.
    sweep_cols = [c for c in first_row.keys() if c != "branch_predictor"]

    # Drop misprediction-related fields from program_features; they will be recomputed
    # per-config from trace_bp.json instead.
    prog_exclude = {"mispred_rate_local", "mispred_rate_tage", "mispred_rate_delta"}
    prog_keys = [k for k in sorted(prog0.keys()) if k not in prog_exclude]
    prog_cols = [f"{args.program_prefix}{k}" for k in prog_keys]
    cache_cols = [
        f"{args.cache_prefix}{k}"
        for k in sorted(cache_example.keys())
        if k not in ("l1i_kb", "l1d_kb", "l2_kb", "config_idx")
    ]

    # Single scalar misprediction rate derived from trace_bp.json for the active predictor.
    mispred_col = "misprediction_rate"

    out_cols = sweep_cols + [mispred_col] + prog_cols + cache_cols
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as out_f:
        w = csv.DictWriter(out_f, fieldnames=out_cols)
        w.writeheader()

        for row in _iter_sweep_rows(sweep_path):
            benchmark = row.get("benchmark", "")
            if not benchmark:
                raise ValueError("Encountered sweep row without benchmark")

            prog = _load_program_features(str(traces_root), benchmark)

            # Select misprediction rate from trace_bp.json based on branch_predictor config.
            bp_cfg = (row.get("branch_predictor") or "").strip().lower()
            bp_rates = _load_bp_rates(str(traces_root), benchmark)
            if bp_cfg in ("local", "0"):
                mispred = bp_rates["local"]
            elif bp_cfg in ("tage", "1"):
                mispred = bp_rates["tage"]
            else:
                raise ValueError(f"Unknown branch_predictor value {bp_cfg!r} for benchmark {benchmark!r}")

            try:
                l1d_kb = _parse_size_to_kb(row["l1d_size"])
                l1i_kb = _parse_size_to_kb(row["l1i_size"])
                l2_kb = _parse_size_to_kb(row["l2_size"])
            except KeyError as e:
                raise ValueError(f"sweep_results.csv missing cache size column: {e}") from e

            cache_idx = _load_cache_latency_index(str(traces_root), benchmark)
            cache_row = cache_idx.get((l1i_kb, l1d_kb, l2_kb))
            if cache_row is None:
                raise KeyError(
                    f"No cache latency row for benchmark={benchmark!r} "
                    f"(l1i_kb,l1d_kb,l2_kb)=({l1i_kb},{l1d_kb},{l2_kb}). "
                    f"Check {traces_root/benchmark/'ronamol'/'cache_latency_summary.csv'}"
                )

            out_row: Dict[str, Any] = dict(row)
            # Drop branch_predictor so it matches the header (we replace it with misprediction_rate).
            out_row.pop("branch_predictor", None)
            out_row[mispred_col] = mispred

            for k in prog_keys:
                out_row[f"{args.program_prefix}{k}"] = prog[k]
            for k, v in cache_row.items():
                if k in ("l1i_kb", "l1d_kb", "l2_kb", "config_idx"):
                    continue
                out_row[f"{args.cache_prefix}{k}"] = v

            w.writerow(out_row)


if __name__ == "__main__":
    main()

