#!/usr/bin/env python3
"""
Cost model + CPI evaluation for a Peregrine O3CPU hardware configuration.

Given a trained PeregrineMLModel and a set of traces, evaluates the mean CPI
across traces for a single HW config, and exposes a scalar cost function over
the config. The search/optimization loop is intentionally not included yet.

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

BATCH_CONFIGS = 512

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
    """Scalar cost per HW config. Intentionally simple — tune later."""
    if not cfgs:
        return []

    df = pd.DataFrame(cfgs)
    issue_cols = [
        "int_reg_issue_width",
        "int_mult_div_issue_width",
        "fp_reg_issue_width",
        "fp_mult_div_issue_width",
        "read_port_issue_width",
        "rdwr_port_issue_width",
        "simd_unit_issue_width",
    ]
    width_cols = ["fetch_width", "decode_width", "rename_width", "commit_width"]

    df[issue_cols] = df[issue_cols].astype(int)
    df[width_cols] = df[width_cols].astype(int)
    df["rob_size"] = df["rob_size"].astype(int)
    df["lq_entries"] = df["lq_entries"].astype(int)
    df["sq_entries"] = df["sq_entries"].astype(int)
    df["max_icache_fills"] = df["max_icache_fills"].astype(int)

    pipe = df[width_cols].sum(axis=1)
    issue = df[issue_cols].sum(axis=1)
    cache_bytes = (
        df["l1i_size"].map(_parse_size_to_kb)
        + df["l1d_size"].map(_parse_size_to_kb)
        + df["l2_size"].map(_parse_size_to_kb)
    ) * 1024

    bp_cost_map = {"local": 2.0, "tage": 4.0}
    bp_cost = df["branch_predictor"].astype(str).str.lower().map(bp_cost_map)
    if bp_cost.isna().any():
        bad_values = df.loc[bp_cost.isna(), "branch_predictor"].unique().tolist()
        raise ValueError(f"Unrecognized branch_predictor values: {bad_values}")

    costs = (
        0.4 * df["rob_size"]
        + 0.25 * (df["lq_entries"] + df["sq_entries"])
        + 0.12 * pipe
        + 0.18 * issue
        + 3.5e-9 * cache_bytes.astype(float)
        + 0.5 * bp_cost
        + 0.35 * df["max_icache_fills"]
    )
    return costs.tolist()


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
