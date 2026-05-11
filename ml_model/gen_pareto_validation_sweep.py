#!/usr/bin/env python3
"""
Generate a validation sweep CSV from a pareto_front.json.

Each validation-candidate entry in the pareto front supplies a hardware config.
Each row listed in the sampled-traces file supplies a workload config
(benchmark, checkpoint, fast_forward) drawn from the source sweep CSV.
The output is the cross product, in the same schema as the source sweep CSV,
ready to be consumed by sim_benchmarks.sh.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

WORKLOAD_KEYS = ("benchmark", "checkpoint", "fast_forward")

PARAM_KEYS = (
    "branch_predictor",
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
)

OUTPUT_COLUMNS = list(WORKLOAD_KEYS) + list(PARAM_KEYS)


def _extract_config(cfg: Dict[str, object], pareto_path: Path, label: str) -> Dict[str, object]:
    missing = [k for k in PARAM_KEYS if k not in cfg]
    if missing:
        raise ValueError(
            f"{label} in {pareto_path} missing keys: {missing}"
        )
    return {k: cfg[k] for k in PARAM_KEYS}


def load_validation_configs(pareto_path: Path) -> List[Dict[str, object]]:
    with pareto_path.open() as f:
        data = json.load(f)

    configs: List[Dict[str, object]] = []

    baseline = data.get("baseline")
    if baseline and "config" in baseline:
        configs.append(_extract_config(baseline["config"], pareto_path, "Baseline"))

    for entry in data.get("pareto_front", []):
        if not entry.get("validation_candidate"):
            continue
        configs.append(_extract_config(entry["config"], pareto_path, "Validation candidate"))

    return configs


def load_sampled_row_ids(trace_list_path: Path) -> List[int]:
    ids: List[int] = []
    with trace_list_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("row_"):
                raise ValueError(
                    f"Unexpected entry in {trace_list_path}: {line!r} "
                    "(expected 'row_<int>')"
                )
            ids.append(int(line[len("row_"):]))
    return ids


def load_workloads(sweep_csv: Path, row_ids: List[int]) -> List[Dict[str, str]]:
    """
    Read the rows at 1-indexed positions `row_ids` from `sweep_csv` (matching
    sim_benchmarks.sh indexing: the first data row after the header is row 1).
    """
    wanted = set(row_ids)
    workloads: Dict[int, Dict[str, str]] = {}
    with sweep_csv.open() as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if idx in wanted:
                workloads[idx] = {k: row[k] for k in WORKLOAD_KEYS}
                if len(workloads) == len(wanted):
                    break

    missing = [i for i in row_ids if i not in workloads]
    if missing:
        raise ValueError(
            f"{len(missing)} row ids not found in {sweep_csv} "
            f"(first few: {missing[:5]})"
        )
    return [workloads[i] for i in row_ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pareto-front",
        type=Path,
        required=True,
        help="Path to a pareto_front.json.",
    )
    parser.add_argument(
        "--sampled-rows",
        type=Path,
        default=Path(
            "/home/ubuntu/peregrine/ml_model/pareto_fronts/"
            "sampled_2048_fetched_traces.txt"
        ),
        help="Text file listing sampled row ids as 'row_<N>' per line.",
    )
    parser.add_argument(
        "--sweep-csv",
        type=Path,
        default=Path(
            "/home/ubuntu/peregrine/ml_model/gem5_sweeps/"
            "spec_v3_region_param_sweep.csv"
        ),
        help=(
            "Source sweep CSV whose rows supply (benchmark, checkpoint, "
            "fast_forward) for each sampled row id."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output sweep CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    validation_configs = load_validation_configs(args.pareto_front)
    if not validation_configs:
        raise SystemExit(
            f"No baseline or validation candidates found in {args.pareto_front}"
        )

    row_ids = load_sampled_row_ids(args.sampled_rows)
    workloads = load_workloads(args.sweep_csv, row_ids)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for cfg in validation_configs:
            for workload in workloads:
                row = {**workload, **cfg}
                writer.writerow(row)

    total = len(validation_configs) * len(workloads)
    print(
        f"Wrote {total} rows to {args.output_csv} "
        f"({len(validation_configs)} validation configs "
        f"x {len(workloads)} workloads)"
    )


if __name__ == "__main__":
    main()
