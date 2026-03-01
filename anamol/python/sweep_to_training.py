"""
sweep_to_training.py — Generate training data from gem5 simulation sweep results.

For each row in a sweep CSV (produced by a gem5 parameter sweep), this script:
  1. Maps gem5 parameter names/formats → anamol Config
  2. Runs the analytical model (anamol binary) for that specific config on the
     matching benchmark trace (traces/<benchmark>-pin/)
  3. Computes training features directly from the output .npy files
     (no lookup table required)
  4. Appends the CPI target from the CSV row
  5. Collects all rows into a single training DataFrame and saves it

Results are cached per (benchmark, config) so duplicate rows are not re-computed.

With --do-sweep: runs the full parameter sweep once per unique benchmark instead,
builds a lookup table, and uses it for all rows of that benchmark.

Usage:
  python sweep_to_training.py \\
      --sweep-csv sim_sweeps/sim_sweep_02_24_2026/sweep_results.csv \\
      [--benchmarks collatz,sparse] \\
      [--traces-dir traces/] \\
      [--window-size 400] \\
      [--do-sweep] \\
      [--output-dir .cache/sweep_out/] \\
      [--format csv|pkl] \\
      -o training_data.pkl
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Local imports (run from anamol/ directory)
import registry
import utils
import models
from gen_training_data import (
    compute_pipeline_stall_features,
    compute_rob_latency_features,
    get_config_scalar_features,
    generate_training_matrix,
)

# ── gem5 → anamol Config parameter name mapping (auto-derived from registry) ──

def _build_gem5_mappings():
    """
    Build gem5→Config field mappings from registry.yaml PARAMS.

    Returns (direct_map, size_map, sum_map):
      direct_map: {gem5_col: config_field}    — plain int copy
      size_map:   {gem5_col: config_field}    — parse "256KiB"→256 KB int
      sum_map:    {config_field: [gem5_cols]} — param = sum of gem5 cols
    branch_predictor is handled separately (string → int conversion).
    """
    direct: dict = {}
    size: dict = {}
    sums: dict = {}
    for p in registry.PARAMS:
        if p.name_gem5 is None or p.name == "branch_predictor":
            continue
        if isinstance(p.name_gem5, list):
            sums[p.name] = p.name_gem5
        elif p.name.endswith("_kb"):
            size[p.name_gem5] = p.name
        else:
            direct[p.name_gem5] = p.name
    return direct, size, sums

_GEM5_DIRECT, _GEM5_SIZE, _GEM5_SUM = _build_gem5_mappings()


def _parse_size_to_kb(value: str) -> int:
    """Parse a size string like '256KiB', '2MiB', '512KiB' into KB integer."""
    value = str(value).strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGk]i?[Bb]?)", value, re.IGNORECASE)
    if not m:
        # Try plain integer (assume KB)
        return int(float(value))
    num = float(m.group(1))
    unit = m.group(2).upper()
    if unit.startswith("K"):
        return int(num)
    elif unit.startswith("M"):
        return int(num * 1024)
    elif unit.startswith("G"):
        return int(num * 1024 * 1024)
    raise ValueError(f"Unrecognised size string: {value!r}")


def _parse_branch_predictor(value: str) -> int:
    """Map gem5 branch predictor name to Config integer (0=local, 1=tage)."""
    v = str(value).strip().lower()
    if v == "local":
        return 0
    elif v == "tage":
        return 1
    # Fall back: try to parse as integer
    try:
        return int(value)
    except (ValueError, TypeError):
        raise ValueError(f"Unknown branch_predictor value: {value!r}")


def parse_gem5_row(row: pd.Series) -> models.Config:
    """
    Convert a gem5 sweep CSV row into an anamol Config object.

    Handles:
    - Name remapping (e.g. lq_entries → load_queue_size)
    - Size string parsing (e.g. "256KiB" → 256)
    - Branch predictor mapping ("local" → 0, "tage" → 1)
    - ls_issue_width derived as rdwr + read port widths
    - misprediction_percent defaulted (not in gem5 CSV)
    - simd_unit_issue_width ignored (no Config field)
    """
    config_dict = {}

    # Direct int mappings (from registry name_gem5 str fields)
    for gem5_col, config_field in _GEM5_DIRECT.items():
        if gem5_col in row.index:
            config_dict[config_field] = int(row[gem5_col])

    # Size string → KB int mappings (params ending in _kb)
    for gem5_col, config_field in _GEM5_SIZE.items():
        if gem5_col in row.index:
            config_dict[config_field] = _parse_size_to_kb(row[gem5_col])

    # Sum mappings (from registry name_gem5 list fields, e.g. ls_issue_width)
    for config_field, gem5_cols in _GEM5_SUM.items():
        if all(c in row.index for c in gem5_cols):
            config_dict[config_field] = sum(int(row[c]) for c in gem5_cols)

    # Branch predictor (string → int, handled separately)
    if "branch_predictor" in row.index:
        config_dict["branch_predictor"] = _parse_branch_predictor(
            row["branch_predictor"]
        )

    # misprediction_percent: not in gem5 CSV — use Config default
    # (no entry → dataclass default of 5 is used)

    return models.Config(**config_dict)


def _config_to_hash(config: models.Config) -> str:
    """Deterministic short hash of all Config param values for cache keying."""
    items = sorted(
        (p.name, getattr(config, p.name)) for p in registry.ENABLED_PARAMS
    )
    key = "_".join(f"{k}{v}" for k, v in items)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _load_bp_rate(trace_dir: Path, branch_predictor: int) -> Optional[float]:
    """Load misprediction rate for this BP type from trace_bp.json. Returns None if missing."""
    bp_json = trace_dir / "trace_bp.json"
    if not bp_json.exists():
        return None
    with open(bp_json) as f:
        data = json.load(f)
    bp_name = "local" if branch_predictor == 0 else "tage"
    return data.get(bp_name)


def _get_config_idx(trace_dir: Path, config: models.Config) -> Optional[int]:
    """
    Look up the cache config index for this config in trace_configs.json.

    The JSON format is: {"configs": [[l1i_kb, l1d_kb, l2_kb], ...]}
    Returns None if the config is not found.
    """
    configs_path = trace_dir / "trace_configs.json"
    if not configs_path.exists():
        return None

    with open(configs_path) as f:
        meta = json.load(f)

    target = [
        config.l1i_cache_kb,
        config.l1d_cache_kb,
        config.l2_cache_kb,
    ]
    for idx, cfg in enumerate(meta.get("configs", [])):
        if list(cfg) == target:
            return idx
    return None


def _run_anamol(
    anamol_bin: str,
    trace_dir: Path,
    output_dir: Path,
    window_size: int,
    config: Optional[models.Config] = None,
) -> None:
    """
    Run the anamol binary on the given trace directory.

    If config is provided, passes --config-json for single-config mode.
    Otherwise runs the full parameter sweep.
    """
    trace_csv = trace_dir / "trace.csv"
    latencies_npy = trace_dir / "trace_latencies.npy"

    cmd = [
        anamol_bin,
        "-t", str(trace_csv),
        "-w", str(window_size),
        "-o", str(output_dir),
    ]

    if latencies_npy.exists():
        cmd += ["-l", str(latencies_npy)]

    if config is not None:
        # Build JSON config string from all enabled params
        config_dict = {
            p.name: getattr(config, p.name)
            for p in registry.ENABLED_PARAMS
        }
        cmd += ["--config-json", json.dumps(config_dict)]

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Running: {' '.join(cmd[:4])} ... -o {output_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"anamol failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )


def _compute_thr_features_from_output(
    output_dir: Path,
    config: models.Config,
    trace_dir: Path,
) -> pd.DataFrame:
    """
    Compute per-resource throughput CDF features directly from .npy files,
    without going through a lookup table.

    Produces the same column layout as ThroughputLookupTable.get_config_features():
      for each enabled resource in registry order:
        {res}_raw_p{p}, {res}_weighted_p{p}, {res}_mean   (101 cols each)

    For latency-independent resources: loads from output_dir/thr_{res}.npy
    For latency-dependent resources:  loads from output_dir/config_{idx:04d}/thr_{res}.npy
    """
    config_idx = _get_config_idx(trace_dir, config)

    feature_dict = {}

    for res_def in registry.ENABLED_RESOURCES:
        res_name = res_def.name
        is_dep = res_name in registry.LATENCY_DEPENDENT_RESOURCES

        if is_dep:
            if config_idx is None:
                print(
                    f"  WARNING: config_idx not found for {res_name} "
                    f"(l1i={config.l1i_cache_kb}, l1d={config.l1d_cache_kb}, "
                    f"l2={config.l2_cache_kb}, bp={config.branch_predictor}); "
                    f"skipping — features will be zero"
                )
                for p in utils.PERCENTILE_POINTS:
                    feature_dict[f"{res_name}_raw_p{int(p)}"] = 0.0
                for p in utils.PERCENTILE_POINTS:
                    feature_dict[f"{res_name}_weighted_p{int(p)}"] = 0.0
                feature_dict[f"{res_name}_mean"] = 0.0
                continue
            npy_path = output_dir / f"config_{config_idx:04d}" / f"thr_{res_name}.npy"
        else:
            npy_path = output_dir / f"thr_{res_name}.npy"

        if not npy_path.exists():
            print(f"  WARNING: {npy_path} not found; features for {res_name} will be zero")
            for p in utils.PERCENTILE_POINTS:
                feature_dict[f"{res_name}_raw_p{int(p)}"] = 0.0
            for p in utils.PERCENTILE_POINTS:
                feature_dict[f"{res_name}_weighted_p{int(p)}"] = 0.0
            feature_dict[f"{res_name}_mean"] = 0.0
            continue

        _params, thr = utils.load_resource_file(str(npy_path), res_name)

        if thr.shape[0] == 0:
            throughputs = np.array([0.0])
        else:
            # Single-config run produces 1 row; take it.
            # (Full sweep also works: pick the row matching the config's params.)
            throughputs = thr[0]

        cdf_raw, cdf_weighted, mean_val = utils.compute_cdf_features(
            throughputs, num_points=50
        )

        for j, p in enumerate(utils.PERCENTILE_POINTS):
            feature_dict[f"{res_name}_raw_p{int(p)}"] = cdf_raw[j]
        for j, p in enumerate(utils.PERCENTILE_POINTS):
            feature_dict[f"{res_name}_weighted_p{int(p)}"] = cdf_weighted[j]
        feature_dict[f"{res_name}_mean"] = mean_val

    return pd.DataFrame([feature_dict])


def process_sweep_csv(
    sweep_csv: str,
    traces_dir: str = "traces",
    benchmarks: Optional[list] = None,
    window_size: int = 400,
    do_sweep: bool = False,
    output_dir: str = ".cache/sweep_out",
    output_format: str = "pkl",
    output_path: str = "training/sweep_training.pkl",
    anamol_bin: str = "./anamol",
    precomputed_dir: Optional[str] = None,
    workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Main pipeline: read sweep CSV → generate training data for each row.

    Args:
        sweep_csv:       Path to gem5 sweep results CSV (columns: cpi, benchmark, ...)
        traces_dir:      Base directory for benchmark traces; expects <dir>/<benchmark>-pin/
        benchmarks:      Whitelist of benchmark names to process (None = all)
        window_size:     Sliding window size for the analytical model
        do_sweep:        If True, build a lookup table from sweep outputs and use it for
                         feature extraction (more efficient for large CSVs)
        output_dir:      Base directory for intermediate anamol outputs / cache
        output_format:   "pkl" or "csv"
        output_path:     Where to save the final training DataFrame
        anamol_bin:      Path to the anamol executable
        precomputed_dir: Directory containing already-computed anamol sweep outputs,
                         organized as <dir>/<benchmark>/ (each with thr_*.npy and
                         config_*/ subdirs). Implies --do-sweep; skips running the
                         binary. Lookup tables are built/cached alongside the outputs.

    Returns:
        DataFrame with one row per input CSV row (filtered) and all training features
    """
    traces_dir = Path(traces_dir)
    output_dir = Path(output_dir)
    output_path = Path(output_path)

    # ── Load and filter sweep CSV ─────────────────────────────────────────────
    print(f"Loading sweep CSV: {sweep_csv}")
    df = pd.read_csv(sweep_csv)
    print(f"  Total rows: {len(df)}")

    if benchmarks:
        df = df[df["benchmark"].isin(benchmarks)].reset_index(drop=True)
        print(f"  After benchmark filter ({benchmarks}): {len(df)} rows")

    if df.empty:
        print("No rows to process after filtering.")
        return pd.DataFrame()

    # ── Validate trace directories ────────────────────────────────────────────
    unique_benchmarks = df["benchmark"].unique().tolist()
    valid_benchmarks = []
    for bm in unique_benchmarks:
        trace_dir = traces_dir / f"{bm}-pin"
        if not (trace_dir / "trace.csv").exists():
            print(f"  WARNING: trace not found for '{bm}' at {trace_dir}/trace.csv — skipping")
        else:
            valid_benchmarks.append(bm)

    df = df[df["benchmark"].isin(valid_benchmarks)].reset_index(drop=True)
    if df.empty:
        print("No rows remain after trace validation.")
        return pd.DataFrame()

    # ── do_sweep / precomputed mode: build lookup table once per benchmark ────
    if do_sweep or precomputed_dir is not None:
        result = _process_with_sweep(
            df, traces_dir, window_size, output_dir,
            anamol_bin, precomputed_dir=precomputed_dir, workers=workers,
        )
        if not result.empty:
            _save_output(result, Path(output_path), output_format)
        return result

    # ── Single-config mode: process each row individually ────────────────────
    stall_features_cache: dict = {}
    all_rows = []
    n_total = len(df)

    for row_idx, row in df.iterrows():
        benchmark = row["benchmark"]
        cpi = float(row["cpi"])
        trace_dir = traces_dir / f"{benchmark}-pin"

        print(f"\n[{row_idx + 1}/{n_total}] benchmark={benchmark}, cpi={cpi:.4f}")

        # Parse config from gem5 row
        try:
            config = parse_gem5_row(row)
        except Exception as e:
            print(f"  WARNING: Failed to parse config: {e} — skipping row")
            continue

        # Populate misprediction_percent from trace BP sim results
        bp_rate = _load_bp_rate(trace_dir, config.branch_predictor)
        if bp_rate is not None:
            config.misprediction_percent = round(bp_rate * 100)

        # Deterministic cache directory per (benchmark, config)
        config_hash = _config_to_hash(config)
        row_output_dir = output_dir / benchmark / config_hash

        # Run anamol if not already cached
        needs_run = not (row_output_dir / "thr_rob.npy").exists()
        if needs_run:
            print(f"  Running anamol (single-config) → {row_output_dir}")
            try:
                _run_anamol(anamol_bin, trace_dir, row_output_dir, window_size, config)
            except RuntimeError as e:
                print(f"  WARNING: anamol failed: {e} — skipping row")
                continue
        else:
            print(f"  Using cached output: {row_output_dir}")

        # Compute throughput features directly from .npy files
        try:
            thr_features = _compute_thr_features_from_output(
                row_output_dir, config, trace_dir
            )
        except Exception as e:
            print(f"  WARNING: Feature extraction failed: {e} — skipping row")
            continue

        # Pipeline stall features (cached per benchmark)
        if benchmark not in stall_features_cache:
            trace_csv = trace_dir / "trace.csv"
            stall_features_cache[benchmark] = compute_pipeline_stall_features(
                str(trace_csv), window_size
            )
        stall_features = stall_features_cache[benchmark]

        # ROB latency features
        config_idx = _get_config_idx(trace_dir, config)
        try:
            rob_features = compute_rob_latency_features(
                str(row_output_dir), config_idx
            )
        except FileNotFoundError as e:
            print(f"  WARNING: ROB latency features not found: {e} — skipping row")
            continue

        # Config scalar features
        config_features = pd.DataFrame([get_config_scalar_features(config)])

        # Combine all features
        row_data = pd.concat(
            [thr_features, stall_features, rob_features, config_features],
            axis=1,
        )
        row_data["cpi"] = cpi
        all_rows.append(row_data)

    if not all_rows:
        print("No training rows generated.")
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    _save_output(result, output_path, output_format)
    return result


def _process_single_row(
    row_idx: int,
    row: pd.Series,
    n_total: int,
    traces_dir: Path,
    lookup_tables: dict,
    bm_output_dirs: dict,
    window_size: int,
) -> Optional[pd.DataFrame]:
    """Process one sweep CSV row → training DataFrame row (or None on failure)."""
    benchmark = row["benchmark"]
    cpi = float(row["cpi"])
    trace_dir = traces_dir / f"{benchmark}-pin"
    bm_output_dir = bm_output_dirs.get(benchmark)

    print(f"\n[{row_idx + 1}/{n_total}] benchmark={benchmark}, cpi={cpi:.4f}")

    if benchmark not in lookup_tables or bm_output_dir is None:
        print(f"  WARNING: No lookup table for '{benchmark}' — skipping row")
        return None

    try:
        config = parse_gem5_row(row)
    except Exception as e:
        print(f"  WARNING: Failed to parse config: {e} — skipping row")
        return None

    bp_rate = _load_bp_rate(trace_dir, config.branch_predictor)
    if bp_rate is not None:
        config.misprediction_percent = round(bp_rate * 100)

    try:
        row_df = generate_training_matrix(
            [config],
            lookup_tables[benchmark],
            str(trace_dir / "trace.csv"),
            window_size,
            include_latency_features=True,
            output_dir=str(bm_output_dir),
        )
    except Exception as e:
        print(f"  WARNING: generate_training_matrix failed: {e} — skipping row")
        return None

    row_df["cpi"] = cpi
    return row_df


def _process_with_sweep(
    df: pd.DataFrame,
    traces_dir: Path,
    window_size: int,
    output_dir: Path,
    anamol_bin: str,
    precomputed_dir: Optional[str] = None,
    workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Sweep mode: use a lookup table (built from full sweep outputs) for each benchmark.

    If precomputed_dir is provided, assumes anamol has already been run and its outputs
    are at <precomputed_dir>/<benchmark>/  — skips the binary run entirely.
    Otherwise runs the full sweep first.
    """
    from build_throughput_lookup import ThroughputLookupTable

    unique_benchmarks = df["benchmark"].unique().tolist()
    lookup_tables: dict = {}
    bm_output_dirs: dict = {}  # benchmark → Path of anamol output dir

    # Resolve per-benchmark output directories and build/load lookup tables
    for benchmark in unique_benchmarks:
        trace_dir = traces_dir / f"{benchmark}-pin"

        if precomputed_dir is not None:
            # Use existing outputs; skip binary run.
            # anamol writes to output/<trace_dir_name>/ which is <benchmark>-pin.
            bm_output_dir = Path(precomputed_dir) / f"{benchmark}-pin"
            if not bm_output_dir.exists():
                print(
                    f"  WARNING: precomputed output not found at {bm_output_dir} "
                    f"— skipping {benchmark}"
                )
                continue
            print(f"\nUsing precomputed output for {benchmark}: {bm_output_dir}")
        else:
            bm_output_dir = output_dir / benchmark / "sweep"

        bm_output_dirs[benchmark] = bm_output_dir
        lookup_path = bm_output_dir / "throughput_lookup.pkl"

        if lookup_path.exists():
            print(f"  Loading cached lookup table: {lookup_path}")
            lookup_tables[benchmark] = ThroughputLookupTable.load(str(lookup_path))
        else:
            if precomputed_dir is None:
                print(f"  Running full sweep for {benchmark} → {bm_output_dir}")
                _run_anamol(anamol_bin, trace_dir, bm_output_dir, window_size, config=None)

            configs_json = trace_dir / "trace_configs.json"
            print(f"  Building lookup table for {benchmark}...")
            lt = ThroughputLookupTable(
                str(bm_output_dir),
                configs_json=str(configs_json) if configs_json.exists() else None,
            )
            lt.save(str(lookup_path))
            lookup_tables[benchmark] = lt

    # Process rows in parallel using lookup tables
    n_total = len(df)
    max_workers = workers or os.cpu_count() or 1
    print(f"\nProcessing {n_total} rows with {max_workers} workers...")

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, (_, row) in enumerate(df.iterrows()):
            fut = executor.submit(
                _process_single_row,
                i, row, n_total,
                traces_dir, lookup_tables, bm_output_dirs, window_size,
            )
            futures_map[fut] = i

        all_rows = []
        for fut in as_completed(futures_map):
            result = fut.result()
            if result is not None:
                all_rows.append(result)

    if not all_rows:
        print("No training rows generated.")
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def _save_output(df: pd.DataFrame, output_path: Path, fmt: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(output_path, index=False)
        print(f"\nSaved {df.shape[0]} rows × {df.shape[1]} cols → {output_path} (CSV)")
    else:
        df.to_pickle(output_path)
        print(f"\nSaved {df.shape[0]} rows × {df.shape[1]} cols → {output_path} (pkl)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from gem5 sweep results CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sweep-csv",
        required=True,
        help="Path to gem5 sweep results CSV (columns: cpi, benchmark, ...)",
    )
    parser.add_argument(
        "--benchmarks",
        default=None,
        help="Comma-separated whitelist of benchmarks (default: all)",
    )
    parser.add_argument(
        "--traces-dir",
        default="traces",
        help="Base directory for benchmark traces (default: traces/)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=400,
        help="Sliding window size for the analytical model (default: 400)",
    )
    parser.add_argument(
        "--do-sweep",
        action="store_true",
        help=(
            "Run full parameter sweep once per benchmark and use a lookup table "
            "for feature extraction (more efficient for large CSVs)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".cache/sweep_out",
        help="Base directory for intermediate anamol outputs (default: .cache/sweep_out)",
    )
    parser.add_argument(
        "--format",
        choices=["pkl", "csv"],
        default="pkl",
        help="Output file format (default: pkl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output path for training DataFrame (.pkl or .csv)",
    )
    parser.add_argument(
        "--precomputed-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory containing already-computed anamol sweep outputs, "
            "organized as <DIR>/<benchmark>/ (each subdirectory must contain "
            "thr_*.npy and config_*/ subdirs from a full sweep run). "
            "Skips running the binary; implies lookup-table mode. "
            "Example: --precomputed-dir output"
        ),
    )
    parser.add_argument(
        "--anamol-bin",
        default=None,
        help="Path to anamol binary (default: ./anamol relative to script's parent)",
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel worker threads (default: os.cpu_count())",
    )

    args = parser.parse_args()

    # Resolve anamol binary path
    if args.anamol_bin:
        anamol_bin = args.anamol_bin
    else:
        # Default: anamol/ directory (one level up from python/)
        script_dir = Path(__file__).parent
        anamol_bin = str(script_dir.parent / "anamol")

    # Only require the binary when we'll actually run it
    needs_binary = args.precomputed_dir is None and not (
        # do_sweep with existing .pkl files also doesn't need binary,
        # but we can't know that ahead of time — check at runtime
        False
    )
    if needs_binary and not Path(anamol_bin).exists():
        print(f"ERROR: anamol binary not found at {anamol_bin}", file=sys.stderr)
        print("Build it first with: make anamol CXX=g++-15", file=sys.stderr)
        sys.exit(1)

    benchmarks = (
        [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        if args.benchmarks
        else None
    )

    # Change to anamol root directory so relative paths work
    anamol_root = Path(__file__).parent.parent
    os.chdir(anamol_root)

    result = process_sweep_csv(
        sweep_csv=args.sweep_csv,
        traces_dir=args.traces_dir,
        benchmarks=benchmarks,
        window_size=args.window_size,
        do_sweep=args.do_sweep,
        output_dir=args.output_dir,
        output_format=args.format,
        output_path=args.output,
        anamol_bin=anamol_bin,
        precomputed_dir=args.precomputed_dir,
        workers=args.workers,
    )

    if result.empty:
        print("No training data generated.")
        sys.exit(1)


if __name__ == "__main__":
    main()
