"""
Build the per-row input file consumed by run_sweep_parallel.sh.

Reads:
  - sim_region_param_sweep.csv   (row_id = line-number-after-header)
  - sweep_outputs_v3/sweep_results.csv  (cpi + exact copy of the sweep row)

Writes:
  - <out>  with one TAB-separated record per row:
      row_id \t <full row of sim_region_param_sweep.csv> \t <cpi>

Rows whose parameter tuple is not present in sweep_results.csv are skipped
(with a summary line on stderr).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-csv", required=True, help="sim_region_param_sweep.csv")
    ap.add_argument("--results-csv", required=True, help="sweep_results.csv (has cpi)")
    ap.add_argument("-o", "--output", required=True, help="TSV written for GNU parallel")
    ap.add_argument("--row-limit", type=int, default=0,
                    help="0 = all rows, otherwise cap (useful for dry runs)")
    args = ap.parse_args()

    # Load sweep_results.csv into a dict keyed by the row-text that follows cpi.
    results_path = Path(args.results_csv)
    print(f"Indexing CPI from {results_path} ...", file=sys.stderr)
    cpi_by_row: dict[str, str] = {}
    with open(results_path, newline="") as f:
        header = f.readline()
        if not header.startswith("cpi,"):
            print(f"ERROR: {results_path} does not start with 'cpi,' — got {header[:60]!r}",
                  file=sys.stderr)
            return 2
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # split off the leading cpi column — sweep rows never contain commas
            # inside quoted fields, so a single comma split is safe.
            cpi, _, rest = line.partition(",")
            cpi_by_row[rest] = cpi
    print(f"  {len(cpi_by_row):,} CPI records loaded", file=sys.stderr)

    sweep_path = Path(args.sweep_csv)
    print(f"Building row list from {sweep_path} ...", file=sys.stderr)
    n_out = 0
    n_missing = 0
    with open(sweep_path, newline="") as f_in, \
         open(args.output, "w", newline="") as f_out:
        header = f_in.readline()  # consume
        if not header.strip():
            print("ERROR: empty sweep CSV header", file=sys.stderr)
            return 2
        row_id = 0
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                continue
            row_id += 1
            if args.row_limit and row_id > args.row_limit:
                break
            cpi = cpi_by_row.get(line)
            if cpi is None:
                n_missing += 1
                continue
            # TSV: row_id \t full_row \t cpi
            f_out.write(f"{row_id}\t{line}\t{cpi}\n")
            n_out += 1
    print(f"  wrote {n_out:,} rows to {args.output} ({n_missing:,} missing CPI)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
