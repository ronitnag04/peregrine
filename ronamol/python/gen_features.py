"""
CLI for Ronamol simple analytical features.

Writes:
  - <trace_dir>/ronamol/program_features.json
  - <trace_dir>/ronamol/cache_latency_summary.(csv|json)   (optional)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

try:
    from simple_model import compute_program_features, summarize_cache_latencies
except ModuleNotFoundError:
    _HERE = Path(__file__).resolve()
    sys.path.insert(0, str(_HERE.parent))
    from simple_model import compute_program_features, summarize_cache_latencies  # type: ignore


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
    ap.add_argument("--latencies-npy", default=None, help="Optional path to *_latencies.npy")
    ap.add_argument("--configs-json", default=None, help="Optional path to *_configs.json")
    ap.add_argument(
        "--no-cache-lat-summary",
        action="store_true",
        help="Skip per-cache-config latency summary generation",
    )
    ap.add_argument(
        "--cache-lat-format",
        choices=["json", "csv"],
        default="csv",
        help="Format for cache latency summary output",
    )
    args = ap.parse_args()

    trace_csv = Path(args.trace_csv)
    out_dir = Path(args.out_dir) if args.out_dir is not None else (trace_csv.parent / "ronamol")

    prog = compute_program_features(trace_csv)
    _write_json(out_dir / "program_features.json", prog.to_dict())

    if not args.no_cache_lat_summary:
        rows = summarize_cache_latencies(
            trace_csv,
            latencies_npy=args.latencies_npy,
            configs_json=args.configs_json,
        )
        if args.cache_lat_format == "json":
            _write_json(out_dir / "cache_latency_summary.json", rows)
        else:
            _write_csv(out_dir / "cache_latency_summary.csv", rows)


if __name__ == "__main__":
    main()

