#!/usr/bin/env python3
"""
Generate a parameter sweep for peregrine.py.
Computes total combinations (product of all *config* param value list sizes),
samples N random indices in [0, total), and decodes each index to a parameter
config via mixed-radix (product order: last key varies fastest). Each sampled
config is then expanded across all benchmarks, yielding N * num_benchmarks rows.
Supports arbitrarily large total (e.g. > 10^20): no range(total) or product iteration;
sampling uses randrange and index decoding uses O(num_params) integer arithmetic.

Output CSV has the same format as training_data.csv (from build_training_csv.py)
but without the cpi column, for use as inference input by the trained ML model.

Program features (program_features.json) do not include misprediction rates;
misprediction_rate is set per row from trace_bp.json using the row's
branch_predictor value. branch_predictor is not written to the output CSV.
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from build_training_csv import (
    _load_bp_rates,
    _load_cache_latency_index,
    _load_program_features,
    _parse_size_to_kb,
)

# Benchmarks are expanded *after* sampling config points.
BENCHMARKS = [
    "branch_storm",
    "collatz",
    "dhrystone",
    "linpack",
    "sieve",
    "sparse",
    "towers",
    "whetstone",
]

# Predefined values per *config* parameter (excludes benchmark). Modify as needed.
PARAM_VALUES = {
    "int_reg_issue_width": list(range(1, 8 + 1)),
    "int_mult_div_issue_width": list(range(1, 8 + 1)),
    "fp_reg_issue_width": list(range(1, 8 + 1)),
    "fp_mult_div_issue_width": list(range(1, 8 + 1)),
    "read_port_issue_width": list(range(1, 8 + 1)),
    "rdwr_port_issue_width": list(range(1, 8 + 1)),
    "simd_unit_issue_width": [1],
    "fetch_width": list(range(1, 12 + 1)),
    "decode_width": list(range(1, 12 + 1)),
    "rename_width": list(range(1, 12 + 1)),
    "commit_width": list(range(1, 12 + 1)),
    "rob_size": list(range(1, 1024 + 1)),
    "lq_entries": list(range(1, 256 + 1)),
    "sq_entries": list(range(1, 256 + 1)),
    "branch_predictor": ["local", "tage"],
    "l1d_size": ["16KiB", "32KiB", "64KiB", "128KiB", "256KiB"],
    "l1i_size": ["16KiB", "32KiB", "64KiB", "128KiB", "256KiB"],
    "l2_size": ["512KiB", "1MiB", "2MiB", "4MiB"],
    "max_icache_fills": list(range(1, 32 + 1)),
    "stride_prefetcher_degree": [0, 4],
}

# Fixed order: same as itertools.product(*value_lists), last varies fastest
PARAM_KEYS = sorted(PARAM_VALUES.keys())

# Output column order matching training_data.csv (from build_training_csv) but without cpi.
# Sweep columns: same as training_data sweep cols (no branch_predictor in output; it is used
# only to look up misprediction_rate from trace_bp.json).
SWEEP_COLS_NO_CPI = [
    "benchmark",
    "commit_width",
    "decode_width",
    "fetch_width",
    "fp_mult_div_issue_width",
    "fp_reg_issue_width",
    "int_mult_div_issue_width",
    "int_reg_issue_width",
    "l1d_size",
    "l1i_size",
    "l2_size",
    "lq_entries",
    "max_icache_fills",
    "rdwr_port_issue_width",
    "read_port_issue_width",
    "rename_width",
    "rob_size",
    "simd_unit_issue_width",
    "sq_entries",
    "stride_prefetcher_degree",
]
MISPRED_COL = "misprediction_rate"
CACHE_EXCLUDE = {"l1i_kb", "l1d_kb", "l2_kb", "config_idx"}


def _discover_prog_and_cache_cols(
    traces_root: str,
    benchmark: str,
    program_prefix: str,
    cache_prefix: str,
) -> Tuple[List[str], List[str], List[str]]:
    """Discover prog_keys, prog_cols, and cache_cols (same logic as build_training_csv)."""
    prog = _load_program_features(traces_root, benchmark)
    prog_keys = sorted(prog.keys())
    prog_cols = [f"{program_prefix}{k}" for k in prog_keys]

    cache_idx = _load_cache_latency_index(traces_root, benchmark)
    example = next(iter(cache_idx.values())) if cache_idx else {}
    cache_cols = [
        f"{cache_prefix}{k}"
        for k in sorted(example.keys())
        if k not in CACHE_EXCLUDE
    ]
    return prog_keys, prog_cols, cache_cols


def _combination_to_inference_row(
    row: Dict[str, Any],
    traces_root: str,
    program_prefix: str,
    cache_prefix: str,
    prog_keys: List[str],
    cache_cols: List[str],
) -> Dict[str, Any]:
    """
    Convert one parameter combination (sweep row) to an inference CSV row.
    Same join as build_training_csv: program features + cache latency + misprediction_rate
    (from trace_bp.json using row["branch_predictor"]). branch_predictor is not in output.
    No cpi.
    """
    benchmark = row["benchmark"]
    prog = _load_program_features(traces_root, benchmark)
    bp_cfg = (row.get("branch_predictor") or "").strip().lower()
    bp_rates = _load_bp_rates(traces_root, benchmark)
    if bp_cfg in ("local", "0"):
        mispred = bp_rates["local"]
    elif bp_cfg in ("tage", "1"):
        mispred = bp_rates["tage"]
    else:
        raise ValueError(f"Unknown branch_predictor {bp_cfg!r} for benchmark {benchmark!r}")

    l1d_kb = _parse_size_to_kb(row["l1d_size"])
    l1i_kb = _parse_size_to_kb(row["l1i_size"])
    l2_kb = _parse_size_to_kb(row["l2_size"])

    cache_idx = _load_cache_latency_index(traces_root, benchmark)
    cache_row = cache_idx.get((l1i_kb, l1d_kb, l2_kb))
    if cache_row is None:
        raise KeyError(
            f"No cache latency row for benchmark={benchmark!r} "
            f"(l1i_kb,l1d_kb,l2_kb)=({l1i_kb},{l1d_kb},{l2_kb}). "
            f"Check {traces_root}/{benchmark}/ronamol/cache_latency_summary.csv"
        )

    out: Dict[str, Any] = {}
    for c in SWEEP_COLS_NO_CPI:
        out[c] = row[c]
    out[MISPRED_COL] = mispred
    for k in prog_keys:
        out[f"{program_prefix}{k}"] = prog[k]
    for k in cache_cols:
        raw_k = k[len(cache_prefix):]
        out[k] = cache_row[raw_k]
    return out


def total_combinations(param_values):
    """Total number of combinations (product of all value list sizes)."""
    n = 1
    for k in PARAM_KEYS:
        n *= len(param_values[k])
    return n


def _compute_strides(param_values):
    """Strides for mixed-radix decode: strides[i] = product of sizes[i+1:]."""
    sizes = [len(param_values[k]) for k in PARAM_KEYS]
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    return sizes, strides


def index_to_combination(index, param_values, sizes=None, strides=None):
    """
    Decode a linear index into a parameter dict using mixed-radix.
    Order matches itertools.product(*value_lists) with PARAM_KEYS: last varies fastest.
    Pass precomputed sizes, strides (from _compute_strides) when decoding many indices
    to avoid recomputation and to keep working with huge totals.
    """
    value_lists = [param_values[k] for k in PARAM_KEYS]
    if sizes is None or strides is None:
        sizes, strides = _compute_strides(param_values)
    digits = [(index // strides[i]) % sizes[i] for i in range(len(sizes))]
    return dict(
        zip(PARAM_KEYS, [value_lists[i][d] for i, d in enumerate(digits)])
    )


# When n >= total we would return all indices; only materialize if total is small
_MAX_INDICES_MATERIALIZE = 10**9


def sample_random_indices(total, n):
    """
    Return n distinct random indices in [0, total) without materializing range(total).
    Safe for arbitrarily large total (e.g. > 10^20); uses randrange and a set.
    When n >= total, returns all indices only if total <= _MAX_INDICES_MATERIALIZE.
    """
    if n >= total:
        if total <= _MAX_INDICES_MATERIALIZE:
            return list(range(total))
        raise ValueError(
            f"total combinations ({total}) is very large; "
            "num_combinations must be less than total (use -n to sample a subset)."
        )
    chosen = []
    seen = set()
    while len(chosen) < n:
        idx = random.randrange(total)
        if idx not in seen:
            seen.add(idx)
            chosen.append(idx)
    return chosen


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate parameter sweep CSV for ML inference (same format as training_data.csv, no cpi). "
            "Samples N random *config* points, then expands each across all benchmarks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--num-configs",
        dest="num_configs",
        type=int,
        default=2**14,
        help=(
            "Number of distinct *config* indices to sample. "
            f"Output rows = N * {len(BENCHMARKS)} benchmarks."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="prediction_sweep.csv",
        help="Output CSV path (inference-ready, same columns as training_data.csv without cpi).",
    )
    parser.add_argument(
        "--traces-root",
        type=str,
        default="traces",
        help="Root containing per-benchmark trace folders (program_features.json, cache_latency_summary.csv, trace_bp.json).",
    )
    parser.add_argument(
        "--program-prefix",
        type=str,
        default="prog_",
        help="Prefix for program feature columns (must match training_data.csv).",
    )
    parser.add_argument(
        "--cache-prefix",
        type=str,
        default="cache_",
        help="Prefix for cache latency columns (must match training_data.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=262,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    traces_root = Path(args.traces_root)
    if not traces_root.is_dir():
        raise FileNotFoundError(f"traces-root not found: {traces_root}")

    total = total_combinations(PARAM_VALUES)
    n = min(args.num_configs, total)

    print(
        f"Total config combinations: {total:.2e}. "
        f"Sampling {n} random config indices; expanding across {len(BENCHMARKS)} benchmarks "
        f"-> {n * len(BENCHMARKS)} rows."
    )

    random_indices = sample_random_indices(total, n)
    sizes, strides = _compute_strides(PARAM_VALUES)
    configs = [
        index_to_combination(k, PARAM_VALUES, sizes=sizes, strides=strides)
        for k in random_indices
    ]
    cfg_df = pd.DataFrame(configs)[PARAM_KEYS]

    expanded_rows: List[Dict[str, Any]] = []
    for _, cfg_row in cfg_df.iterrows():
        cfg_dict = cfg_row.to_dict()
        for bench in BENCHMARKS:
            expanded_rows.append({"benchmark": bench, **cfg_dict})
    df = pd.DataFrame(expanded_rows)

    # Discover output columns from first row (same as build_training_csv)
    first_bench = df.iloc[0]["benchmark"]
    prog_keys, prog_cols, cache_cols = _discover_prog_and_cache_cols(
        str(traces_root), first_bench, args.program_prefix, args.cache_prefix
    )
    out_cols = SWEEP_COLS_NO_CPI + [MISPRED_COL] + prog_cols + cache_cols

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for _, row in df.iterrows():
            out_row = _combination_to_inference_row(
                row.to_dict(),
                str(traces_root),
                args.program_prefix,
                args.cache_prefix,
                prog_keys,
                cache_cols,
            )
            w.writerow(out_row)

    print(f"Wrote {len(df)} rows to {args.output}")

if __name__ == "__main__":
    main()
