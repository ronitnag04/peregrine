#!/usr/bin/env python3
"""
Read spec_v3_sweep.csv, sample N rows at random (across all benchmarks, or from a specific benchmark
if --benchmark is given), copy matching trace archives from S3, and gunzip them under ronamol/traces/.
Downloads use a thread pool (-j / --jobs) so many traces can be fetched in parallel.

Sweep row k (1-based among data rows) maps to:
  <s3-prefix>/row_<k>.trace.csv.gz
where line 2 of the CSV (first data row) is k=1 (i.e. k = physical_line_number - 1).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def _find_aws_cli() -> str:
    aws = shutil.which("aws")
    if not aws:
        sys.exit(
            "aws CLI not found in PATH; install it or ensure it is on PATH for S3 copies."
        )
    return aws


def sweep_row_id_from_df_index(idx: int) -> int:
    """Default RangeIndex: first data row is index 0 -> S3 row_1 (same as file line - 1)."""
    return int(idx) + 1


def load_sweep_and_filter(csv_path: Path, benchmark: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "benchmark" not in df.columns:
        sys.exit("CSV must have a 'benchmark' column.")
    if benchmark is None:
        return df.copy()
    return df.loc[df["benchmark"] == benchmark].copy()


def fetch_one_trace(
    rid: int,
    s3_prefix: str,
    dest_root: Path,
    aws: str,
    dry_run: bool,
) -> tuple[int, str, Path, str | None]:
    """Download one trace and gunzip it. Returns (rid, s3_key, local_gz_path, error_or_none)."""
    key = f"{s3_prefix}/row_{rid}.trace.csv.gz"
    subdir = dest_root / f"row_{rid}"
    local_gz = subdir / "trace.csv.gz"
    if dry_run:
        return (rid, key, local_gz, None)
    try:
        subdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [aws, "s3", "cp", key, str(local_gz)],
            check=True,
        )
        subprocess.run(
            ["gunzip", "-f", str(local_gz)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return (rid, key, local_gz, f"subprocess failed (exit {e.returncode})")
    except OSError as e:
        return (rid, key, local_gz, str(e))
    return (rid, key, local_gz, None)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sample N sweep rows from all benchmarks (or a specific benchmark with --benchmark) and fetch trace .csv.gz from S3."
    )
    p.add_argument(
        "-n",
        "--count",
        type=int,
        required=True,
        help="Number of random rows to fetch",
    )
    p.add_argument(
        "--benchmark",
        help="Optional: benchmark name as in CSV (e.g. 541.leela_r). If omitted, samples across all benchmarks.",
    )
    p.add_argument(
        "--sim-sweep-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "spec_v3_sweep.csv",
        help="Path to simulation sweep CSV (e.g. spec_v3_sweep.csv)",
    )
    p.add_argument(
        "--s3-prefix",
        default="s3://ronitnag04-peregrine/spec/spec-v3/traces_04_25_2026",
        help="S3 URI prefix (no trailing slash) containing row_<k>.trace.csv.gz",
    )
    p.add_argument(
        "--output-base",
        type=Path,
        default=Path("/home/ubuntu/peregrine/ronamol"),
        help="Directory under which <benchmark>_traces/ is created",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=262,
        help="RNG seed for reproducible sampling",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without copying or decompressing",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Max parallel trace downloads (default: min(64, CPU*4, number of traces); "
            "use 1 for strictly sequential I/O)"
        ),
    )
    args = p.parse_args()

    if args.count < 1:
        sys.exit("--count must be >= 1")
    if args.jobs is not None and args.jobs < 1:
        sys.exit("--jobs must be >= 1")

    csv_path = args.sim_sweep_csv.resolve()
    if not csv_path.is_file():
        sys.exit(f"CSV not found: {csv_path}")

    s3_prefix = args.s3_prefix.rstrip("/")

    print(f"Loading {csv_path} with pandas ...", flush=True)
    filtered = load_sweep_and_filter(csv_path, args.benchmark)
    if filtered.empty:
        if args.benchmark:
            sys.exit(f"No rows found for benchmark {args.benchmark!r}")
        else:
            sys.exit("No rows found in CSV")

    n = min(args.count, len(filtered))
    if n < args.count:
        bench_msg = f" for benchmark {args.benchmark!r}" if args.benchmark else ""
        print(
            f"Warning: only {len(filtered)} matching rows{bench_msg}; sampling {n} instead of {args.count}.",
            file=sys.stderr,
        )

    # Stratified sampling: sample proportionally from each benchmark
    sampling_frac = n / len(filtered)
    sample_df = filtered.groupby("benchmark", group_keys=False).apply(
        lambda x: x.sample(frac=sampling_frac, random_state=args.seed)
    )

    # If we're slightly under due to rounding, top up with additional random samples
    if len(sample_df) < n:
        remaining = n - len(sample_df)
        unsampled = filtered.drop(sample_df.index)
        additional = unsampled.sample(n=remaining, random_state=args.seed + 1)
        sample_df = pd.concat([sample_df, additional])

    chosen = sorted(sweep_row_id_from_df_index(i) for i in sample_df.index)
    print(f"Picked {len(chosen)} sweep row id(s):")
    benchmark_count = sample_df["benchmark"].value_counts()

    if args.benchmark:
        bench_safe = args.benchmark.replace("/", "_")
        dest_root = args.output_base.resolve() / f"{bench_safe}_traces"
    else:
        dest_root = args.output_base.resolve() / "fetched_traces"
    dest_root.mkdir(parents=True, exist_ok=True)

    aws = _find_aws_cli()

    if args.jobs is None:
        max_workers = max(1, min(len(chosen), (os.cpu_count() or 4) * 4, 64))
    else:
        max_workers = min(max(1, args.jobs), len(chosen))

    print(
        f"Fetching {len(chosen)} trace(s) with up to {max_workers} parallel worker(s) ...",
        flush=True,
    )

    errors: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                fetch_one_trace,
                rid,
                s3_prefix,
                dest_root,
                aws,
                args.dry_run,
            ): rid
            for rid in chosen
        }
        for fut in as_completed(future_map):
            rid, key, local_gz, err = fut.result()
            line = f"  row_{rid}: {key} -> {local_gz}"
            if err:
                line += f"  ERROR: {err}"
                errors.append((rid, err))
            print(line, flush=True)

    if errors:
        for rid, msg in errors:
            print(f"Failed row_{rid}: {msg}", file=sys.stderr)
        sys.exit(1)

    print("Benchmark distribution among chosen rows:")
    for benchmark, count in benchmark_count.items():
        print(f"  {benchmark}: {count}")

    print(f"Done. Output under {dest_root}")


if __name__ == "__main__":
    main()
