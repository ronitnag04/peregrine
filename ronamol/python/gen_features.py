"""
CLI for Ronamol simple analytical features.

Expects the following directory structure:
  <trace_dir>/
  ├── trace.csv
  ├── cache_latencies/
  │   ├── l1i_16_l1d_16_l2_512_cache_latencies.npy
  │   ├── l1i_16_l1d_16_l2_1024_cache_latencies.npy
  │   └── ... (other cache configurations)
  └── bp_rates/
      ├── local_bp_rate.npy
      └── tage_bp_rate.npy

Writes:
  - <trace_dir>/ronamol/program_features.(csv|json)
  - <trace_dir>/ronamol/cache_latency_summary.(csv|json)
  - <trace_dir>/ronamol/bp_rates_summary.(csv|json)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

try:
    from simple_model import compute_program_features, summarize_cache_latencies, summarize_bp_rates
except ModuleNotFoundError:
    _HERE = Path(__file__).resolve()
    sys.path.insert(0, str(_HERE.parent))
    from simple_model import compute_program_features, summarize_cache_latencies, summarize_bp_rates  # type: ignore


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_csv", help="Path to trace.csv")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <trace_dir>/ronamol/)",
    )
    ap.add_argument("--latencies-npy", default=None, help="Optional path to cache latencies .npy file")
    ap.add_argument("--configs-json", default=None, help="Optional path to cache configs .json file")
    ap.add_argument("--bp-rates-dir", default=None, help="Optional path to bp_rates directory")
    ap.add_argument(
        "--program-features-format",
        choices=["json", "csv"],
        default="csv",
        help="Format for program features output",
    )
    ap.add_argument(
        "--cache-lat-format",
        choices=["json", "csv"],
        default="csv",
        help="Format for cache latency summary output",
    )
    ap.add_argument(
        "--bp-rates-format",
        choices=["json", "csv"], 
        default="csv",
        help="Format for branch predictor rates summary output",
    )
    args = ap.parse_args()

    trace_csv = Path(args.trace_csv)
    out_dir = Path(args.out_dir) if args.out_dir is not None else (trace_csv.parent / "ronamol")
    print(f"Generating features for trace: {trace_csv}")

    prog = compute_program_features(trace_csv)
    prog_row = prog.to_dict()
    if args.program_features_format == "json":
        _write_json(out_dir / "program_features.json", prog_row)
        print(f" - Wrote program_features.json to {out_dir / 'program_features.json'}")
    else:
        _write_csv(out_dir / "program_features.csv", [prog_row])
        print(f" - Wrote program_features.csv to {out_dir / 'program_features.csv'}")

    # Generate cache latency summary
    rows = summarize_cache_latencies(
        trace_csv,
        latencies_npy=args.latencies_npy,
        configs_json=args.configs_json,
    )
    if args.cache_lat_format == "json":
        _write_json(out_dir / "cache_latency_summary.json", rows)
        print(f" - Wrote cache_latency_summary.json to {out_dir / 'cache_latency_summary.json'}")
    else:
        _write_csv(out_dir / "cache_latency_summary.csv", rows)
        print(f" - Wrote cache_latency_summary.csv to {out_dir / 'cache_latency_summary.csv'}")
        
    # Generate branch predictor rates summary
    rates = summarize_bp_rates(
        trace_csv,
        bp_rates_dir=args.bp_rates_dir,
    )
    # Convert dictionary to list of dictionaries for CSV/JSON output
    rows = [{"bp_type": bp_type, "misprediction_rate": rate} 
            for bp_type, rate in sorted(rates.items())]
    
    if args.bp_rates_format == "json":
        _write_json(out_dir / "bp_rates_summary.json", rows)
        print(f" - Wrote bp_rates_summary.json to {out_dir / 'bp_rates_summary.json'}")
    else:
        _write_csv(out_dir / "bp_rates_summary.csv", rows)
        print(f" - Wrote bp_rates_summary.csv to {out_dir / 'bp_rates_summary.csv'}")
        
    print(f" - Finished generating features for trace: {trace_csv}")


if __name__ == "__main__":
    main()

