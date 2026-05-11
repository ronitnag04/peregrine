"""
Per-row worker: download one S3 trace, run anamol + ronamol feature extractors,
and append a single row to training_data.csv.

Invoked in parallel via run_sweep_parallel.sh.

Args (positional):
    1. row_id             integer (1-based line number inside sim_region_param_sweep.csv data)
    2. row                the CSV row (everything after the benchmark column header)
    3. cpi                the CPI target (string, as it appears in sweep_results.csv)
    4. training_csv       output CSV path (locked-append)
    5. training_lock      lock file for training_csv (flock-style)
    6. scratch_dir        per-row scratch directory (caller pre-creates it)

On any failure the row is dropped (message goes to stderr) and the script exits 0 so
that the GNU parallel job does not kill the whole sweep.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Local imports resolve because the driver adds anamol/python/ to PYTHONPATH.
import registry
import utils
import models

# The feature helpers print progress chatter. With hundreds of parallel workers
# that gets unreadable — mute stdout just for these imports/calls.
class _Silence:
    def __enter__(self):
        self._orig = sys.stdout
        sys.stdout = open(os.devnull, "w")
        return self

    def __exit__(self, *_a):
        sys.stdout.close()
        sys.stdout = self._orig

from gen_training_data import (  # noqa: E402
    compute_pipeline_stall_features,
    compute_rob_latency_features,
    get_config_scalar_features,
)

# ── Paths / tunables ─────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
ANAMOL_ROOT = SCRIPT_DIR.parent
ANAMOL_BIN = ANAMOL_ROOT / "anamol"
PEREGRINE_ROOT = ANAMOL_ROOT.parent

WINDOW_SIZE = 400
S3_PREFIX = "s3://ronitnag04-peregrine/spec/spec-v3/traces_04_25_2026"
AWS_PROFILE = "default"  # the 'peregrine' profile name referenced in the task
                         # is not actually configured on this box; the default
                         # credentials grant the needed read access.

# sim_region_param_sweep.csv column order (no cpi).
SWEEP_COLS = [
    "benchmark", "checkpoint", "fast_forward", "branch_predictor",
    "commit_width", "decode_width", "fetch_width",
    "fp_mult_div_issue_width", "fp_reg_issue_width",
    "int_mult_div_issue_width", "int_reg_issue_width",
    "l1d_size", "l1i_size", "l2_size",
    "lq_entries", "max_icache_fills",
    "rdwr_port_issue_width", "read_port_issue_width",
    "rename_width", "rob_size", "simd_unit_issue_width",
    "sq_entries", "stride_prefetcher_degree",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_size_kb(s: str) -> int:
    s = s.strip()
    if s.endswith("KiB"):
        return int(s[:-3])
    if s.endswith("MiB"):
        return int(s[:-3]) * 1024
    raise ValueError(f"Unsupported size: {s!r}")


def _bp_to_int(bp: str) -> int:
    bp = bp.strip().lower()
    if bp == "local":
        return 0
    if bp == "tage":
        return 1
    raise ValueError(f"Unsupported branch predictor: {bp!r}")


def _row_to_config(row: dict) -> models.Config:
    """Map a sim_region_param_sweep row (dict of str values) to an anamol Config."""
    return models.Config(
        rob_size=int(row["rob_size"]),
        commit_width=int(row["commit_width"]),
        load_queue_size=int(row["lq_entries"]),
        store_queue_size=int(row["sq_entries"]),
        alu_issue_width=int(row["int_reg_issue_width"]),
        alu_mult_div_issue_width=int(row["int_mult_div_issue_width"]),
        fp_issue_width=int(row["fp_reg_issue_width"]),
        fp_mult_div_issue_width=int(row["fp_mult_div_issue_width"]),
        ls_issue_width=int(row["rdwr_port_issue_width"]) + int(row["read_port_issue_width"]),
        num_ls_pipes=int(row["rdwr_port_issue_width"]),
        num_load_pipes=int(row["read_port_issue_width"]),
        fetch_width=int(row["fetch_width"]),
        decode_width=int(row["decode_width"]),
        rename_width=int(row["rename_width"]),
        max_icache_fills=int(row["max_icache_fills"]),
        branch_predictor=_bp_to_int(row["branch_predictor"]),
        l1d_cache_kb=_parse_size_kb(row["l1d_size"]),
        l1i_cache_kb=_parse_size_kb(row["l1i_size"]),
        l2_cache_kb=_parse_size_kb(row["l2_size"]),
        l1d_stride_prefetch=int(row["stride_prefetcher_degree"]),
        # misprediction_percent is filled in after the BP sim runs
    )


def _run(cmd, **kw):
    """Run a subprocess, raising with captured stdout/stderr on failure."""
    res = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if res.returncode != 0:
        raise RuntimeError(
            f"command failed (exit {res.returncode}): {' '.join(map(str, cmd))}\n"
            f"stdout tail: {res.stdout[-1500:]}\n"
            f"stderr tail: {res.stderr[-1500:]}"
        )
    return res


def _download_trace(row_id: int, dest_csv: Path) -> None:
    """Pull row_<id>.trace.csv.gz from S3 and gunzip into dest_csv."""
    gz_path = dest_csv.with_suffix(dest_csv.suffix + ".gz")  # .csv.gz
    s3_key = f"{S3_PREFIX}/row_{row_id}.trace.csv.gz"
    env = os.environ.copy()
    env["AWS_PROFILE"] = AWS_PROFILE
    _run(
        ["aws", "s3", "cp", s3_key, str(gz_path)],
        env=env,
    )
    # gunzip → dest_csv
    with gzip.open(gz_path, "rb") as gz, open(dest_csv, "wb") as out:
        shutil.copyfileobj(gz, out)
    gz_path.unlink(missing_ok=True)


def _build_latencies_npy(trace_csv: Path, config: models.Config, out_npy: Path) -> None:
    """
    Run the evantrace cache sim for this row's specific cache config
    and write a single-config (shape (1, N_instrs, 2) uint16) .npy that
    the anamol binary can consume with the -l flag.
    """
    # Import here so workers don't import evantrace unnecessarily when they fail early.
    sys.path.insert(0, str(PEREGRINE_ROOT))
    from gen_cache_latency import run_sim  # noqa: E402

    latencies = run_sim(
        str(trace_csv),
        l1i_kb=config.l1i_cache_kb,
        l1d_kb=config.l1d_cache_kb,
        l2_kb=config.l2_cache_kb,
    )
    arr = np.asarray(latencies, dtype=np.uint16).reshape(1, -1, 2)
    np.save(out_npy, arr)


def _compute_bp_rate(trace_csv: Path, bp_type: str) -> float:
    """Return the misprediction rate for the given BP on this trace."""
    sys.path.insert(0, str(PEREGRINE_ROOT))
    from gen_bp_rate import run_sim as bp_run_sim  # noqa: E402
    return float(bp_run_sim(str(trace_csv), bp_type))


def _run_anamol(trace_csv: Path, latencies_npy: Path, out_dir: Path,
                config: models.Config) -> None:
    """Run the anamol binary in single-config, latency-aware mode."""
    config_dict = {
        p.name: float(getattr(config, p.name))
        if p.param_type == "float"
        else int(getattr(config, p.name))
        for p in registry.ENABLED_PARAMS
    }
    # The binary's JSON parser expects integers; encode misprediction_percent
    # as a percent integer (e.g. 0.05 → 5). It isn't used by any enabled model.
    config_dict["misprediction_percent"] = int(round(float(config.misprediction_percent) * 100))

    cmd = [
        str(ANAMOL_BIN),
        "-t", str(trace_csv),
        "-w", str(WINDOW_SIZE),
        "-o", str(out_dir),
        "-l", str(latencies_npy),
        "-c", json.dumps(config_dict),
    ]
    _run(cmd)


def _extract_throughput_features(out_dir: Path, config_idx: int) -> dict:
    """
    Mirror of sweep_to_training._compute_thr_features_from_output but returns
    a plain dict so we can stay away from pandas concats in the hot path.
    """
    feats: dict = {}
    for res in registry.ENABLED_RESOURCES:
        if res.name in registry.LATENCY_DEPENDENT_RESOURCES:
            npy_path = out_dir / f"config_{config_idx:04d}" / f"thr_{res.name}.npy"
        else:
            npy_path = out_dir / f"thr_{res.name}.npy"

        if not npy_path.exists():
            # anamol didn't write this resource; zero-fill so column schema stays stable.
            for p in utils.PERCENTILE_POINTS:
                feats[f"{res.name}_raw_p{int(p)}"] = 0.0
            for p in utils.PERCENTILE_POINTS:
                feats[f"{res.name}_weighted_p{int(p)}"] = 0.0
            feats[f"{res.name}_mean"] = 0.0
            continue

        _params, thr = utils.load_resource_file(str(npy_path), res.name)
        throughputs = thr[0] if thr.shape[0] else np.array([0.0])
        cdf_raw, cdf_w, mean_v = utils.compute_cdf_features(throughputs, num_points=50)
        for j, p in enumerate(utils.PERCENTILE_POINTS):
            feats[f"{res.name}_raw_p{int(p)}"] = float(cdf_raw[j])
        for j, p in enumerate(utils.PERCENTILE_POINTS):
            feats[f"{res.name}_weighted_p{int(p)}"] = float(cdf_w[j])
        feats[f"{res.name}_mean"] = float(mean_v)
    return feats


def _flock_append(lock_path: Path, csv_path: Path, header_line: str, value_line: str) -> None:
    """Atomically append a line to csv_path, writing the header if the file is new."""
    import fcntl
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if not csv_path.exists() or csv_path.stat().st_size == 0:
                with open(csv_path, "w") as f:
                    f.write(header_line + "\n")
                    f.write(value_line + "\n")
            else:
                # Verify header matches before appending — catches schema drift.
                with open(csv_path) as f:
                    existing_header = f.readline().rstrip("\n")
                if existing_header != header_line:
                    raise RuntimeError(
                        f"Header mismatch in {csv_path}: expected {len(header_line)} "
                        f"chars, existing file has {len(existing_header)} chars"
                    )
                with open(csv_path, "a") as f:
                    f.write(value_line + "\n")
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


# ── Main ─────────────────────────────────────────────────────────────────────

def process_row(row_id: int, row_csv: str, cpi: str,
                training_csv: Path, training_lock: Path,
                scratch_dir: Path) -> None:
    # Parse row
    fields = row_csv.split(",")
    if len(fields) != len(SWEEP_COLS):
        raise ValueError(f"row {row_id}: expected {len(SWEEP_COLS)} cols, got {len(fields)}")
    row = dict(zip(SWEEP_COLS, (f.strip() for f in fields)))

    config = _row_to_config(row)

    scratch_dir.mkdir(parents=True, exist_ok=True)
    trace_csv = scratch_dir / "trace.csv"
    lat_npy = scratch_dir / "latencies.npy"
    anamol_out = scratch_dir / "anamol_out"

    try:
        _download_trace(row_id, trace_csv)

        # Cache latency annotation + BP misprediction rate (inputs to anamol / features).
        _build_latencies_npy(trace_csv, config, lat_npy)
        bp_name = "tage" if config.branch_predictor == 1 else "local"
        config.misprediction_percent = _compute_bp_rate(trace_csv, bp_name)

        # Analytical model — single cache config → config_0000/ holds latency-dependent outputs.
        _run_anamol(trace_csv, lat_npy, anamol_out, config)

        # ── Feature extraction ────────────────────────────────────────────
        with _Silence():
            thr_feats = _extract_throughput_features(anamol_out, config_idx=0)
            stall_df = compute_pipeline_stall_features(str(trace_csv), WINDOW_SIZE)
            rob_df = compute_rob_latency_features(str(anamol_out), config_idx=0)
            cfg_feats = get_config_scalar_features(config)

        # ── Compose training row ─────────────────────────────────────────
        out = {"cpi": float(cpi), "row_id": row_id}
        out.update(row)  # original sweep params (strings preserved)
        out.update(thr_feats)
        out.update({c: float(stall_df.iloc[0][c]) for c in stall_df.columns})
        out.update({c: float(rob_df.iloc[0][c]) for c in rob_df.columns})
        # cfg_feats keys already include everything in get_config_scalar_features
        out.update(cfg_feats)

        header_line = ",".join(out.keys())
        value_line = ",".join(_csv_fmt(v) for v in out.values())
        _flock_append(training_lock, training_csv, header_line, value_line)

    finally:
        # Always wipe the local trace (big file) and any intermediates.
        shutil.rmtree(scratch_dir, ignore_errors=True)


def _csv_fmt(v) -> str:
    if isinstance(v, float):
        # repr preserves precision; guard against commas / quotes (shouldn't appear).
        return repr(v)
    s = str(v)
    # No row in this dataset should contain a comma or quote, but be safe.
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def main() -> int:
    if len(sys.argv) != 7:
        print(
            "Usage: process_sweep_row.py <row_id> <row_csv> <cpi> "
            "<training_csv> <training_lock> <scratch_dir>",
            file=sys.stderr,
        )
        return 2

    row_id = int(sys.argv[1])
    row_csv = sys.argv[2]
    cpi = sys.argv[3]
    training_csv = Path(sys.argv[4])
    training_lock = Path(sys.argv[5])
    scratch_dir = Path(sys.argv[6])

    try:
        process_row(row_id, row_csv, cpi, training_csv, training_lock, scratch_dir)
    except Exception as e:
        print(f"[row {row_id}] FAILED: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Exit 0 so GNU parallel records the row as "done" and moves on.
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
