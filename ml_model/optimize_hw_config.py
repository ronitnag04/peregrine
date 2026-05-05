#!/usr/bin/env python3
"""
Cost model + CPI evaluation for a Peregrine O3CPU hardware configuration.

Feature preprocessing reuses ``train.pre_process_features`` so the rows seen
by the model here match exactly what the model was trained on.

Requires the Neuron PyTorch venv:
  source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch_xla
from sklearn.preprocessing import StandardScaler
import joblib

from model import PeregrineMLModel
from train import pre_process_features

BATCH_CONFIGS = 1

PARAM_VALUES = {
    "int_reg_issue_width": list(range(1, 8 + 1)),
    "int_mult_div_issue_width": list(range(1, 8 + 1)),
    "fp_reg_issue_width": list(range(1, 8 + 1)),
    "fp_mult_div_issue_width": list(range(1, 8 + 1)),
    "read_port_issue_width": list(range(1, 8 + 1)),
    "rdwr_port_issue_width": list(range(1, 8 + 1)),
    "simd_unit_issue_width": [1],           # Update cost model if SIMD width is parameterized.
    "fetch_width": list(range(1, 12 + 1)),
    "decode_width": list(range(1, 12 + 1)),
    "rename_width": list(range(1, 12 + 1)),
    "commit_width": list(range(1, 12 + 1)),
    "rob_size": list(range(1, 1024 + 1)),
    "lq_entries": list(range(1, 256 + 1)),
    "sq_entries": list(range(1, 256 + 1)),
    "branch_predictor": ["local", "tage"],  # Update cost model if more predictors are added.
    "l1d_size": ["16KiB", "32KiB", "64KiB", "128KiB", "256KiB"],
    "l1i_size": ["16KiB", "32KiB", "64KiB", "128KiB", "256KiB"],
    "l2_size": ["512KiB", "1MiB", "2MiB", "4MiB"],
    "max_icache_fills": list(range(1, 32 + 1)),
    "stride_prefetcher_degree": [0, 4],     # Update cost model if more degrees are added.
}

DEFAULT_CONFIG: dict[str, Any] = {
    "int_reg_issue_width": 2,
    "int_mult_div_issue_width": 2,
    "fp_reg_issue_width": 2,
    "fp_mult_div_issue_width": 2,
    "read_port_issue_width": 2,
    "rdwr_port_issue_width": 2,
    "simd_unit_issue_width": 1,
    "fetch_width": 8,
    "decode_width": 8,
    "rename_width": 8,
    "commit_width": 8,
    "rob_size": 192,
    "lq_entries": 32,
    "sq_entries": 32,
    "branch_predictor": "local",
    "l1d_size": "32KiB",
    "l1i_size": "32KiB",
    "l2_size": "512KiB",
    "max_icache_fills": 4,
    "stride_prefetcher_degree": 4,
}

# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def hardware_cost(cfgs: list[dict[str, Any]]) -> list[float]:
    """
    Relative area/cost estimate (state elements + combinatorial logic proxy)
    for a Peregrine O3CPU configuration. Total = cpu_cost + memory_cost.

    Weightings are grounded in what each gem5 parameter actually instantiates
    (see peregrine-gem5/configs/peregrine/peregrine.py, src/cpu/o3/FuncUnitConfig.py,
    and src/cpu/pred/*).
    """
    if not cfgs:
        return []

    df = pd.DataFrame(cfgs)
    int_cols = [
        "int_reg_issue_width",
        "int_mult_div_issue_width",
        "fp_reg_issue_width",
        "fp_mult_div_issue_width",
        "read_port_issue_width",
        "rdwr_port_issue_width",
        "fetch_width",
        "decode_width",
        "rename_width",
        "commit_width",
        "rob_size",
        "lq_entries",
        "sq_entries",
        "max_icache_fills",
        "stride_prefetcher_degree",
    ]
    df[int_cols] = df[int_cols].astype(int)

    # ---------- CPU cost ----------
    # Per-unit functional unit cost (combinatorial logic + small per-unit state).
    # Weightings reflect the op classes each FU handles (FuncUnitConfig.py):
    #   int_reg       : single-cycle 64-bit IntAlu (baseline).
    #   int_mult_div  : IntMult (wide combinational multiplier, ~O(64^2)) +
    #                   IntDiv (non-pipelined iterative state machine, 20 cyc).
    #   fp_reg        : FloatAdd/Cmp/Cvt/Bf16Cvt (exponent align, mantissa add,
    #                   normalize, round).
    #   fp_mult_div   : FloatMult/MultAcc/Misc + non-pipelined FloatDiv (12) and
    #                   FloatSqrt (24) — widest mantissa mult + iterative state.
    #   read_port     : MemRead AGU + LSQ CAM read port + L1D tag read.
    #   rdwr_port     : MemRead+MemWrite AGU + LSQ CAM R/W port + L1D R/W tag.
    fu_cost = (
        1.0 * df["int_reg_issue_width"]
        + 5.0 * df["int_mult_div_issue_width"]
        + 3.0 * df["fp_reg_issue_width"]
        + 8.0 * df["fp_mult_div_issue_width"]
        + 3.0 * df["read_port_issue_width"]
        + 4.0 * df["rdwr_port_issue_width"]
    )

    # Total issue width = sum of parameterized FU counts (simd is fixed at 1
    # and unparameterized, so excluded). This width sets issueWidth / wbWidth
    # in peregrine.py, driving the wakeup/select network and common data bus.
    # Wakeup+select is ~quadratic in issue width: each source tag broadcasts
    # against every issue slot's destination tags.
    total_issue_width = (
        df["int_reg_issue_width"]
        + df["int_mult_div_issue_width"]
        + df["fp_reg_issue_width"]
        + df["fp_mult_div_issue_width"]
        + df["read_port_issue_width"]
        + df["rdwr_port_issue_width"]
    )
    issue_network_cost = 0.4 * total_issue_width ** 2

    # Pipeline widths drive multi-ported structures and wide buses.
    #   fetch_width  : fetch buffer + I-cache fetch width + predecode slots.
    #   decode_width : wide x86 decoder stack (variable-length decode is heavy
    #                  combinational logic per slot).
    #   rename_width : rename map table read/write ports; dependency-resolution
    #                  logic across the group scales roughly with width^2, but
    #                  with the modest widths here a large linear weight captures
    #                  most of it — so give rename the highest per-slot weight.
    #   commit_width : retirement ports + architectural-state update CDB.
    pipe_cost = (
        2.0 * df["fetch_width"]
        + 3.0 * df["decode_width"]
        + 4.0 * df["rename_width"]
        + 2.0 * df["commit_width"]
    )

    # Major out-of-order state arrays.
    #   ROB: ~100+ bits/entry (PC, dest-reg map, exception/flags, status) plus
    #        age/commit CAM ports. Large but per-entry cost is moderate.
    #   LSQ (LQ+SQ): each entry carries an address-matching CAM for
    #        load-store forwarding and memory disambiguation, plus data and
    #        dependency tracking — several× the per-entry cost of a ROB slot.
    #   I-cache MSHRs (max_icache_fills): each tracks one outstanding block
    #        fill as a CAM entry with merging logic for secondary misses.
    rob_cost = 0.8 * df["rob_size"]
    lsq_cost = 2.5 * (df["lq_entries"] + df["sq_entries"])
    mshr_cost = 3.0 * df["max_icache_fills"]

    # Branch predictor (conditional core + shared infrastructure).
    #   LocalBP : one table of 2048 × 2-bit saturating counters (~0.5 KiB) plus
    #             a PC-indexed lookup — very little logic.
    #   TAGE    : 1 bimodal table + 7 tagged tables (tag + 3-bit ctr + 2 u-bits
    #             per entry), multi-way tag compare, folded-history and
    #             path-history LFSRs, useful-counter reset logic. Roughly an
    #             order of magnitude more state and substantially more combo
    #             logic than LocalBP.
    # Shared BPU infrastructure (SimpleBTB 4096 × ~80 bits + 16-entry RAS +
    # SimpleIndirectPredictor 256 sets × 2 ways) is present for both.
    bp_core_map = {"local": 10.0, "tage": 100.0}
    bp_core = df["branch_predictor"].astype(str).str.lower().map(bp_core_map)
    if bp_core.isna().any():
        bad = df.loc[bp_core.isna(), "branch_predictor"].unique().tolist()
        raise ValueError(f"Unrecognized branch_predictor values: {bad}")
    bp_cost = bp_core + 30.0

    # Stride prefetcher is effectively a binary knob (degree 0 = off, 4 = on).
    # Adds a small stride-tracking table and per-access comparison logic.
    prefetch_cost = (df["stride_prefetcher_degree"] > 0).astype(float) * 15.0

    cpu_cost = (
        fu_cost
        + issue_network_cost
        + pipe_cost
        + rob_cost
        + lsq_cost
        + mshr_cost
        + bp_cost
        + prefetch_cost
    )

    # ---------- Memory cost ----------
    # L1 caches are small, multi-ported, tight-timing SRAM with heavy
    # tag/valid/dirty overhead + LSU integration (L1D) or predecode (L1I), so
    # per-KB area is much higher than the larger, single-ported,
    # latency-tolerant L2. Typical rule of thumb is a ~5× ratio.
    l1i_kb = df["l1i_size"].map(_parse_size_to_kb)
    l1d_kb = df["l1d_size"].map(_parse_size_to_kb)
    l2_kb = df["l2_size"].map(_parse_size_to_kb)
    L1_COST_PER_KB = 5.0
    L2_COST_PER_KB = 1.0
    memory_cost = (
        L1_COST_PER_KB * (l1i_kb + l1d_kb)
        + L2_COST_PER_KB * l2_kb
    ) * 0.1  # Weight memory cost less than CPU cost

    return (cpu_cost + memory_cost).tolist()


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------

@dataclass
class TraceBundle:
    root: Path
    program: dict[str, float]          # prog_* columns
    cache_latency: pd.DataFrame        # raw cache_latency_summary.csv
    bp_rates: dict[str, float]         # {"local": ..., "tage": ...}


def _read_program_features_csv(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected exactly 1 row in {path}, got {len(df)}")
    # Match build_training_csv.py program-feature prefixing.
    return {f"prog_{k}": float(v) for k, v in df.iloc[0].to_dict().items()}


def _read_bp_rates_csv(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {str(r["bp_type"]): float(r["misprediction_rate"]) for _, r in df.iterrows()}


def load_trace_bundle(trace_dir: Path) -> TraceBundle:
    ron = trace_dir / "ronamol"
    return TraceBundle(
        root=trace_dir.resolve(),
        program=_read_program_features_csv(ron / "program_features.csv"),
        cache_latency=pd.read_csv(ron / "cache_latency_summary.csv"),
        bp_rates=_read_bp_rates_csv(ron / "bp_rates_summary.csv"),
    )


def discover_trace_dirs(traces_dir: Path) -> list[Path]:
    traces_dir = traces_dir.expanduser().resolve()
    if (traces_dir / "ronamol" / "program_features.csv").is_file():
        return [traces_dir]
    found = [
        c for c in sorted(traces_dir.iterdir())
        if c.is_dir() and (c / "ronamol" / "program_features.csv").is_file()
    ]
    if not found:
        raise ValueError(f"No traces under {traces_dir}")
    return found


# ---------------------------------------------------------------------------
# Feature assembly — build rows matching the training CSV, then reuse
# train.pre_process_features for identical preprocessing.
# ---------------------------------------------------------------------------

def _parse_size_to_kb(s: str) -> int:
    s = str(s).strip()
    m = re.match(r"^(\d+)\s*KiB$", s, re.I)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d+)\s*MiB$", s, re.I)
    if m:
        return int(m.group(1)) * 1024
    raise ValueError(f"Unrecognized cache size: {s!r}")


def _size_str_to_bytes(s: str) -> int:
    return _parse_size_to_kb(s) * 1024


def _cache_latency_row(bundle: TraceBundle, l1i_kb: int, l1d_kb: int, l2_kb: int) -> pd.Series:
    df = bundle.cache_latency
    m = (df["l1i_kb"] == l1i_kb) & (df["l1d_kb"] == l1d_kb) & (df["l2_kb"] == l2_kb)
    hit = df.loc[m]
    if len(hit) == 0:
        raise KeyError(
            f"No cache latency row for (l1i_kb,l1d_kb,l2_kb)=({l1i_kb},{l1d_kb},{l2_kb}) "
            f"in {bundle.root}"
        )
    return hit.iloc[0]


CACHE_LATENCY_COLS = (
    "fetch_mean", "exec_mean",
    "fetch_p50", "fetch_p75", "fetch_p95",
    "exec_p50", "exec_p75", "exec_p95",
)


def build_training_shaped_frame(
    bundles: list[TraceBundle], cfgs: list[dict[str, Any]]
) -> pd.DataFrame:
    """
    ``n_cfgs * n_traces`` rows (cfg-major), schema matching the training CSV
    pre-preprocessing. train.pre_process_features can then be applied directly.
    """
    rows: list[dict[str, Any]] = []
    for cfg in cfgs:
        l1i_kb = _parse_size_to_kb(cfg["l1i_size"])
        l1d_kb = _parse_size_to_kb(cfg["l1d_size"])
        l2_kb = _parse_size_to_kb(cfg["l2_size"])
        bp = str(cfg["branch_predictor"]).lower()
        for tb in bundles:
            row: dict[str, Any] = {
                # Dropped in pre_process_features but required by its signature.
                "cpi": 0.0,
                "benchmark": "none",
                "checkpoint": "none",
                "fast_forward": 0,
                # HW config.
                "branch_predictor": bp,
                "commit_width": int(cfg["commit_width"]),
                "decode_width": int(cfg["decode_width"]),
                "fetch_width": int(cfg["fetch_width"]),
                "fp_mult_div_issue_width": int(cfg["fp_mult_div_issue_width"]),
                "fp_reg_issue_width": int(cfg["fp_reg_issue_width"]),
                "int_mult_div_issue_width": int(cfg["int_mult_div_issue_width"]),
                "int_reg_issue_width": int(cfg["int_reg_issue_width"]),
                "l1d_size": str(cfg["l1d_size"]),
                "l1i_size": str(cfg["l1i_size"]),
                "l2_size": str(cfg["l2_size"]),
                "lq_entries": int(cfg["lq_entries"]),
                "max_icache_fills": int(cfg["max_icache_fills"]),
                "rdwr_port_issue_width": int(cfg["rdwr_port_issue_width"]),
                "read_port_issue_width": int(cfg["read_port_issue_width"]),
                "rename_width": int(cfg["rename_width"]),
                "rob_size": int(cfg["rob_size"]),
                "simd_unit_issue_width": int(cfg["simd_unit_issue_width"]),
                "sq_entries": int(cfg["sq_entries"]),
                "stride_prefetcher_degree": int(cfg["stride_prefetcher_degree"]),
                # Per-trace: BP misprediction rate for the chosen predictor.
                "bp_misprediction_rate": float(tb.bp_rates[bp]),
            }
            # Per-trace: program features (prog_*) and cache latency features (cache_*).
            row.update(tb.program)
            clat = _cache_latency_row(tb, l1i_kb, l1d_kb, l2_kb)
            for k in CACHE_LATENCY_COLS:
                row[f"cache_{k}"] = float(clat[k])
            rows.append(row)

    return pd.DataFrame(rows)


def build_scaled_features(
    bundles: list[TraceBundle],
    cfgs: list[dict[str, Any]],
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Training-schema DataFrame → pre_process_features → scaler.transform (as in predict.py)."""
    raw = build_training_shaped_frame(bundles, cfgs)
    features = pre_process_features(raw)

    # Align to the columns the scaler was fit with. get_dummies may omit
    # stride_prefetcher_degree_X when no config in the batch picks that value.
    feature_names = [str(c) for c in scaler.feature_names_in_]
    for name in feature_names:
        if name not in features.columns:
            features[name] = 0.0

    features = features[feature_names].copy().astype(float)
    features[features.columns] = scaler.transform(features[features.columns])
    return features


# ---------------------------------------------------------------------------
# Model + evaluation
# ---------------------------------------------------------------------------

def load_model(path: Path, input_size: int, device: str) -> PeregrineMLModel:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = PeregrineMLModel(input_size=input_size, hidden_dims=[256, 128], output_size=1)
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    model.eval()
    return model


def resolve_artifact_paths(checkpoint_dir: Path) -> tuple[Path, Path]:
    d = checkpoint_dir.expanduser().resolve()
    if not d.is_dir():
        raise FileNotFoundError(f"--checkpoint must be a directory: {d}")
    ckpt = d / "checkpoint.pt"
    scl = d / "scaler.joblib"
    if not ckpt.is_file() or not scl.is_file():
        raise FileNotFoundError(f"Expected checkpoint.pt and scaler.joblib in {d}")
    return ckpt, scl


def evaluate_configs(
    model: PeregrineMLModel,
    bundles: list[TraceBundle],
    cfgs: list[dict[str, Any]],
    scaler: StandardScaler,
    device: str,
) -> list[float]:
    """Mean CPI across traces, per HW config."""
    features = build_scaled_features(bundles, cfgs, scaler)
    x = torch.from_numpy(features.to_numpy(copy=True)).float().to(device)
    with torch.no_grad():
        y = model(x).reshape(len(cfgs), len(bundles))
        mean_per_cfg = y.mean(dim=1)
    torch_xla.sync()
    return mean_per_cfg.detach().float().cpu().tolist()


# ---------------------------------------------------------------------------
# Entry point: evaluate a single config (default) as a sanity check.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate cost + mean CPI for one HW config.")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Directory containing checkpoint.pt and scaler.joblib.")
    p.add_argument("--traces-dir", type=Path, required=True,
                   help="Parent of trace dirs, or one trace dir containing ronamol/.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path, scaler_path = resolve_artifact_paths(args.checkpoint)
    bundles = [load_trace_bundle(p) for p in discover_trace_dirs(args.traces_dir)]

    scaler: StandardScaler = joblib.load(scaler_path)
    if not hasattr(scaler, "feature_names_in_"):
        raise ValueError("Scaler must have been fit with a pandas DataFrame (feature_names_in_).")
    n_features = len(scaler.feature_names_in_)

    device = "xla"
    model = load_model(ckpt_path, input_size=n_features, device=device)

    cfgs = [dict(DEFAULT_CONFIG)] * BATCH_CONFIGS
    mean_cpis = evaluate_configs(model, bundles, cfgs, scaler, device)
    costs = hardware_cost(cfgs)
    for i, (cpi, cost) in enumerate(zip(mean_cpis, costs)):
        print(f"cfg[{i}]  n_traces={len(bundles)}  mean_cpi={cpi:.6f}  cost={cost:.4f}")


if __name__ == "__main__":
    main()
