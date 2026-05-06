#!/usr/bin/env python3
"""
Pareto-optimal hardware configuration search for Peregrine O3CPU.

Jointly minimizes predicted CPI (via the ML model) and hardware cost (via the
cost model) over the discrete/categorical parameter grid. The search proceeds
in four sequential stages:
  1. Constraint pruning over an integer-encoded parameter space.
  2. Latin Hypercube global sampling + batched model/cost evaluation.
  3. NSGA-II evolutionary refinement, bifurcated on `branch_predictor`.
  4. Archive-wide Pareto extraction, decoding, sensitivity annotation, and
     validation-candidate flagging.

Feature preprocessing is precomputed into numpy LUTs by ``FeatureBuilder``,
matching the columns and transforms applied by ``train.pre_process_features``
so the rows seen by the model here match what the model was trained on.

Requires the Neuron PyTorch venv:
  source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch_xla
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler
import joblib

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from model import PeregrineMLModel

# ---------------------------------------------------------------------------
# Parameterization Grid & Default Parameter Config
# ---------------------------------------------------------------------------

PARAM_VALUES: dict[str, list[Any]] = {
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
# Config Embedding & Decoding Keys
# ---------------------------------------------------------------------------

# Parameters held at a constant value and injected at decode time rather than
# searched over.
FIXED_PARAMS: dict[str, Any] = {"simd_unit_issue_width": 1}

# Ordered list of parameters participating in the search + lookup helpers.
SEARCH_PARAM_ORDER: list[str] = [p for p in PARAM_VALUES if p not in FIXED_PARAMS]
PARAM_N_VALUES: np.ndarray = np.array(
    [len(PARAM_VALUES[p]) for p in SEARCH_PARAM_ORDER], dtype=np.int64
)
PARAM_COL_IDX: dict[str, int] = {p: i for i, p in enumerate(SEARCH_PARAM_ORDER)}
N_SEARCH_PARAMS: int = len(SEARCH_PARAM_ORDER)

# ---------------------------------------------------------------------------
# Parameter Validity Constraint Constants
# ---------------------------------------------------------------------------

# ROB must hold at least ROB_HEADROOM cycles' worth of in-flight instructions
# relative to commit bandwidth. Structural stalls dominate otherwise.
ROB_HEADROOM = 8
# Total issue width across FU types must not wildly exceed rename bandwidth.
ISSUE_WIDTH_RENAME_RATIO = 2.0

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
# Feature assembly — numpy-only LUTs mirroring train.pre_process_features.
# Precomputes every per-bundle and per-param feature contribution at startup
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


class FeatureBuilder:
    """Precomputed numpy feature assembler.

    Mirrors ``train.pre_process_features`` exactly, but avoids all per-call
    pandas work by baking every column's contribution into lookup tables at
    construction time. The hot path is pure numpy gather + broadcast + z-score.

    Layout of the produced feature matrix (per row):
      - columns indexed by ``scaler.feature_names_in_``
      - config-dependent columns vary across cfg axis
      - bundle-dependent columns (prog_*, cache_*, bp_misprediction_rate) vary
        across bundle axis
    """

    def __init__(self, bundles: list[TraceBundle], scaler: StandardScaler) -> None:
        feature_names: list[str] = [str(c) for c in scaler.feature_names_in_]
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.n_bundles = len(bundles)
        col_idx = {name: i for i, name in enumerate(feature_names)}

        mean = np.asarray(scaler.mean_, dtype=np.float32)
        scale = np.asarray(scaler.scale_, dtype=np.float32)

        # ---------- Per-bundle static features (prog_* + cache_* + bp_rate) ----------
        # Shape: (n_l1i, n_l1d, n_l2, n_bundles, n_bp, n_features).
        # We precompute the full (memory-config × bundle × bp) tensor of
        # bundle-dependent contributions; the cfg-dependent columns are zero
        # here and filled separately in transform().
        l1i_sizes = PARAM_VALUES["l1i_size"]
        l1d_sizes = PARAM_VALUES["l1d_size"]
        l2_sizes = PARAM_VALUES["l2_size"]
        bp_vals = PARAM_VALUES["branch_predictor"]
        n_l1i, n_l1d, n_l2 = len(l1i_sizes), len(l1d_sizes), len(l2_sizes)
        n_bp = len(bp_vals)

        bundle_feats = np.zeros(
            (n_l1i, n_l1d, n_l2, self.n_bundles, n_bp, self.n_features),
            dtype=np.float32,
        )
        # Older checkpoints named this column `misprediction_rate`; newer ones
        # use `bp_misprediction_rate`. Accept either.
        if "bp_misprediction_rate" in col_idx:
            bp_rate_col = col_idx["bp_misprediction_rate"]
        elif "misprediction_rate" in col_idx:
            bp_rate_col = col_idx["misprediction_rate"]
        else:
            raise ValueError(
                "Scaler is missing a branch-predictor misprediction rate column "
                "(expected 'bp_misprediction_rate' or 'misprediction_rate')."
            )
        for bi, tb in enumerate(bundles):
            # prog_* columns: same for all (mem-cfg, bp) combinations.
            for k, v in tb.program.items():
                if k in col_idx:
                    bundle_feats[:, :, :, bi, :, col_idx[k]] = float(v)
            # bp_misprediction_rate varies per (bundle, bp).
            for bp_i, bp_name in enumerate(bp_vals):
                bundle_feats[:, :, :, bi, bp_i, bp_rate_col] = float(
                    tb.bp_rates[str(bp_name).lower()]
                )
            # cache_* columns: depend on (mem-cfg, bundle).
            for li_i, l1i in enumerate(l1i_sizes):
                for ld_i, l1d in enumerate(l1d_sizes):
                    for l2_i, l2 in enumerate(l2_sizes):
                        clat = _cache_latency_row(
                            tb,
                            _parse_size_to_kb(l1i),
                            _parse_size_to_kb(l1d),
                            _parse_size_to_kb(l2),
                        )
                        for k in CACHE_LATENCY_COLS:
                            name = f"cache_{k}"
                            if name in col_idx:
                                bundle_feats[li_i, ld_i, l2_i, bi, :, col_idx[name]] = float(clat[k])
        self._bundle_feats = bundle_feats

        # ---------- Per-search-param contribution LUTs ----------
        # For each search param we build a (n_values, n_features) table of the
        # contribution that parameter makes to the feature row. Config features
        # are the sum of these LUTs over the param axis.
        self._param_luts: list[np.ndarray] = []
        for p in SEARCH_PARAM_ORDER:
            vals = PARAM_VALUES[p]
            lut = np.zeros((len(vals), self.n_features), dtype=np.float32)
            if p == "branch_predictor":
                # Handled via bundle_feats (bp axis) — contributes nothing here.
                pass
            elif p in ("l1i_size", "l1d_size", "l2_size"):
                # Numeric byte-size column (pre_process_features converts KiB/MiB → bytes).
                if p in col_idx:
                    for vi, v in enumerate(vals):
                        lut[vi, col_idx[p]] = float(_size_str_to_bytes(v))
            elif p == "stride_prefetcher_degree":
                # One-hot expansion: stride_prefetcher_degree_{value}.
                for vi, v in enumerate(vals):
                    name = f"stride_prefetcher_degree_{v}"
                    if name in col_idx:
                        lut[vi, col_idx[name]] = 1.0
            elif p in ("lq_entries", "sq_entries", "rob_size"):
                # Numeric column + reciprocal column.
                if p in col_idx:
                    for vi, v in enumerate(vals):
                        lut[vi, col_idx[p]] = float(v)
                recip = f"reciprocal_{p}"
                if recip in col_idx:
                    for vi, v in enumerate(vals):
                        lut[vi, col_idx[recip]] = 1.0 / float(v)
            else:
                # Plain numeric search parameter.
                if p in col_idx:
                    for vi, v in enumerate(vals):
                        lut[vi, col_idx[p]] = float(v)
            self._param_luts.append(lut)

        # ---------- Fixed-param contribution (simd_unit_issue_width etc.) ----------
        fixed_contrib = np.zeros(self.n_features, dtype=np.float32)
        for p, v in FIXED_PARAMS.items():
            if p in col_idx:
                fixed_contrib[col_idx[p]] = float(v)
        self._fixed_contrib = fixed_contrib

        # Precompute column indices we need fast access to.
        self._l1i_col = PARAM_COL_IDX["l1i_size"]
        self._l1d_col = PARAM_COL_IDX["l1d_size"]
        self._l2_col = PARAM_COL_IDX["l2_size"]
        self._bp_col = PARAM_COL_IDX["branch_predictor"]

        # Mean/scale for inline z-scoring.
        self._mean = mean
        self._scale = scale

    def transform(self, indices: np.ndarray) -> np.ndarray:
        """Assemble scaled features for ``indices`` (shape: n_cfgs, n_params).

        Returns a ``(n_cfgs * n_bundles, n_features)`` float32 array, cfg-major.
        """
        if indices.ndim != 2:
            raise ValueError(f"indices must be 2D, got shape {indices.shape}")
        n_cfgs = indices.shape[0]
        if n_cfgs == 0:
            return np.empty((0, self.n_features), dtype=np.float32)

        # Config-dependent row: sum each param's LUT contribution + fixed params.
        cfg_rows = np.broadcast_to(self._fixed_contrib, (n_cfgs, self.n_features)).copy()
        for j, lut in enumerate(self._param_luts):
            cfg_rows += lut[indices[:, j]]

        # Gather bundle-dependent rows keyed by (l1i, l1d, l2, bp) for each cfg.
        l1i_i = indices[:, self._l1i_col]
        l1d_i = indices[:, self._l1d_col]
        l2_i = indices[:, self._l2_col]
        bp_i = indices[:, self._bp_col]
        # Shape: (n_cfgs, n_bundles, n_features).
        bundle_rows = self._bundle_feats[l1i_i, l1d_i, l2_i, :, bp_i, :]

        # Broadcast cfg_rows over the bundle axis and sum in the bundle rows.
        out = bundle_rows + cfg_rows[:, None, :]
        out = out.reshape(n_cfgs * self.n_bundles, self.n_features)
        out -= self._mean
        out /= self._scale
        return out


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


@dataclass
class EvalTimings:
    """Cumulative timers capturing where model-evaluation time is spent."""
    feature_build_s: float = 0.0
    model_forward_s: float = 0.0
    cost_s: float = 0.0
    n_cfgs_evaluated: int = 0
    n_model_calls: int = 0

    def add(self, other: "EvalTimings") -> None:
        self.feature_build_s += other.feature_build_s
        self.model_forward_s += other.model_forward_s
        self.cost_s += other.cost_s
        self.n_cfgs_evaluated += other.n_cfgs_evaluated
        self.n_model_calls += other.n_model_calls

    def report(self, prefix: str) -> str:
        return (
            f"{prefix} features={self.feature_build_s:.2f}s "
            f"model={self.model_forward_s:.2f}s "
            f"cost={self.cost_s:.2f}s "
            f"cfgs={self.n_cfgs_evaluated} calls={self.n_model_calls}"
        )


def evaluate_cpis_from_indices(
    indices: np.ndarray,
    model: PeregrineMLModel,
    builder: "FeatureBuilder",
    device: str,
    timings: EvalTimings | None = None,
) -> np.ndarray:
    """Mean CPI across traces for each row of ``indices``."""
    t0 = time.perf_counter()
    features = builder.transform(indices)
    t1 = time.perf_counter()
    x = torch.from_numpy(features).to(device)
    with torch.no_grad():
        y = model(x).reshape(indices.shape[0], builder.n_bundles)
        mean_per_cfg = y.mean(dim=1)
    torch_xla.sync()
    out = mean_per_cfg.detach().float().cpu().numpy().astype(np.float64)
    t2 = time.perf_counter()
    if timings is not None:
        timings.feature_build_s += (t1 - t0)
        timings.model_forward_s += (t2 - t1)
        timings.n_cfgs_evaluated += int(indices.shape[0])
        timings.n_model_calls += 1
    return out


def evaluate_configs(
    model: PeregrineMLModel,
    builder: "FeatureBuilder",
    cfgs: list[dict[str, Any]],
    device: str,
    timings: EvalTimings | None = None,
) -> list[float]:
    """Mean CPI across traces, per HW config."""
    if not cfgs:
        return []
    indices = np.stack([encode_cfg(cfg) for cfg in cfgs], axis=0)
    return evaluate_cpis_from_indices(indices, model, builder, device, timings).tolist()


def evaluate_indices(
    indices: np.ndarray,
    model: PeregrineMLModel,
    builder: "FeatureBuilder",
    device: str,
    batch_size: int,
    timings: EvalTimings | None = None,
) -> np.ndarray:
    """Objective matrix for a batch of encoded configs. Returns (N, 2) of (CPI, cost)."""
    n = indices.shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)

    cpis = np.empty(n, dtype=np.float64)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        cpis[i:j] = evaluate_cpis_from_indices(indices[i:j], model, builder, device, timings)

    t_cost = time.perf_counter()
    cfgs = decode_indices(indices)
    costs = np.asarray(hardware_cost(cfgs), dtype=np.float64)
    if timings is not None:
        timings.cost_s += (time.perf_counter() - t_cost)
    return np.stack([cpis, costs], axis=1)


# ---------------------------------------------------------------------------
# Stage 1: encoding + validity predicate
# ---------------------------------------------------------------------------

def encode_cfg(cfg: dict[str, Any]) -> np.ndarray:
    idx = np.empty(N_SEARCH_PARAMS, dtype=np.int64)
    for i, p in enumerate(SEARCH_PARAM_ORDER):
        idx[i] = PARAM_VALUES[p].index(cfg[p])
    return idx


def decode_indices(indices: np.ndarray) -> list[dict[str, Any]]:
    if indices.ndim != 2:
        raise ValueError(f"indices must be 2D, got shape {indices.shape}")
    cfgs: list[dict[str, Any]] = []
    for row in indices:
        cfg: dict[str, Any] = dict(FIXED_PARAMS)
        for i, p in enumerate(SEARCH_PARAM_ORDER):
            cfg[p] = PARAM_VALUES[p][int(row[i])]
        cfgs.append(cfg)
    return cfgs


def _param_int_values(param: str) -> np.ndarray:
    vals = PARAM_VALUES[param]
    if not all(isinstance(v, int) for v in vals):
        raise ValueError(f"Param {param} is not integer-valued")
    return np.array(vals, dtype=np.int64)


def _vals_for(indices: np.ndarray, param: str) -> np.ndarray:
    """Decoded integer values for `param` over all rows in `indices`."""
    return _param_int_values(param)[indices[:, PARAM_COL_IDX[param]]]


def is_valid(indices: np.ndarray) -> np.ndarray:
    """Vectorized validity check over (N, n_params) integer-index batch."""
    if indices.ndim != 2:
        raise ValueError(f"indices must be 2D, got shape {indices.shape}")

    fetch = _vals_for(indices, "fetch_width")
    decode = _vals_for(indices, "decode_width")
    rename = _vals_for(indices, "rename_width")
    commit = _vals_for(indices, "commit_width")
    rob = _vals_for(indices, "rob_size")
    lq = _vals_for(indices, "lq_entries")
    sq = _vals_for(indices, "sq_entries")
    rp = _vals_for(indices, "read_port_issue_width")
    rw = _vals_for(indices, "rdwr_port_issue_width")
    ir = _vals_for(indices, "int_reg_issue_width")
    im = _vals_for(indices, "int_mult_div_issue_width")
    fr = _vals_for(indices, "fp_reg_issue_width")
    fm = _vals_for(indices, "fp_mult_div_issue_width")

    mem_ports = rp + rw
    total_issue = ir + im + fr + fm + rp + rw

    return (
        (decode <= fetch)
        & (rename <= decode)
        & (commit <= rename)
        & (rob >= ROB_HEADROOM * commit)
        & (lq >= mem_ports)
        & (sq >= mem_ports)
        & (total_issue <= ISSUE_WIDTH_RENAME_RATIO * rename)
    )


def dedup_rows(X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        return X
    return np.unique(X, axis=0)


# ---------------------------------------------------------------------------
# Stage 2: LHS global sampling
# ---------------------------------------------------------------------------

def lhs_sample(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube draw snapped to the discrete parameter grid."""
    sampler = qmc.LatinHypercube(d=N_SEARCH_PARAMS, seed=rng)
    u = sampler.random(n_samples)  # (n_samples, d) in [0, 1)
    indices = np.floor(u * PARAM_N_VALUES).astype(np.int64)
    # Guard against the (vanishingly rare) case of u == 1.0 after rounding.
    indices = np.minimum(indices, PARAM_N_VALUES - 1)
    return indices


def pareto_front_2d(F: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated front for a 2-objective minimization problem.

    O(N log N) via lexicographic sort + prefix-min sweep. Equivalent to
    pymoo's ``NonDominatedSorting(only_non_dominated_front=True)`` for M=2,
    but tractable at N >> 10^4 where the O(N^2) default stalls and blows memory.
    """
    n = F.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    # Sort by (f0 asc, f1 asc). For j < i in sorted order, f0[j] <= f0[i], so
    # i is non-dominated iff its f1 is strictly less than the running min of
    # earlier f1s. Duplicates collapse to their first occurrence.
    order = np.lexsort((F[:, 1], F[:, 0]))
    f1 = F[order, 1]
    keep = np.empty(n, dtype=bool)
    keep[0] = True
    keep[1:] = f1[1:] < np.minimum.accumulate(f1)[:-1]
    return np.sort(order[keep])


def _extract_fronts_until(F: np.ndarray, n_target: int) -> list[np.ndarray]:
    """Return a list of row-index arrays forming successive non-dominated fronts
    until we have collected at least ``n_target`` rows."""
    n = F.shape[0]
    remaining = np.arange(n)
    layers: list[np.ndarray] = []
    collected = 0
    while collected < n_target and remaining.size > 0:
        front = pareto_front_2d(F[remaining])
        layer = remaining[front]
        layers.append(layer)
        collected += layer.size
        mask = np.ones(remaining.size, dtype=bool)
        mask[front] = False
        remaining = remaining[mask]
    return layers


def stage2_global_sample(
    n_samples: int,
    model: PeregrineMLModel,
    builder: FeatureBuilder,
    device: str,
    rng: np.random.Generator,
    batch_size: int,
    near_pareto_frac: float,
) -> dict[str, Any]:
    """Stage 2: sample → filter → evaluate → extract seed population."""
    stage_t0 = time.perf_counter()
    print(f"[Stage 2] LHS sampling {n_samples} candidates over {N_SEARCH_PARAMS} parameters")
    t0 = time.perf_counter()
    X = lhs_sample(n_samples, rng)
    n_raw = X.shape[0]

    # Raw LHS draws only satisfy the constraint predicate rarely (<1%), so
    # repair every draw to the nearest valid index vector. Done per-BP so the
    # repair operator (which pins branch_predictor) preserves the LHS
    # sampling distribution over that axis.
    bp_col = PARAM_COL_IDX["branch_predictor"]
    repaired_parts: list[np.ndarray] = []
    for bp_idx in range(len(PARAM_VALUES["branch_predictor"])):
        mask = X[:, bp_col] == bp_idx
        if not mask.any():
            continue
        repaired_parts.append(
            HWRepair(bp_fixed_idx=bp_idx)._do(None, X[mask].astype(float))
        )
    X = np.concatenate(repaired_parts, axis=0) if repaired_parts else X[:0]
    n_valid = X.shape[0]
    X = dedup_rows(X)
    n_unique = X.shape[0]
    sample_s = time.perf_counter() - t0
    print(f"[Stage 2] {n_raw} drawn → {n_valid} repaired → {n_unique} unique after dedup ({sample_s:.2f}s)")

    if n_unique == 0:
        raise RuntimeError("Stage 2 produced no valid configurations; loosen constraints or raise --lhs-samples")

    timings = EvalTimings()
    t0 = time.perf_counter()
    F = evaluate_indices(X, model, builder, device, batch_size, timings)
    eval_s = time.perf_counter() - t0
    print(f"[Stage 2] Evaluated {n_unique} configs in {eval_s:.2f}s "
          f"(features={timings.feature_build_s:.2f}s model={timings.model_forward_s:.2f}s "
          f"cost={timings.cost_s:.2f}s calls={timings.n_model_calls})")

    t0 = time.perf_counter()
    pareto_rows = pareto_front_2d(F)
    # Near-Pareto inclusion: peel successive non-dominated layers until we have
    # at least ``near_pareto_frac`` of the evaluated population. Preserves
    # genetic diversity for Stage 3 while still biasing toward the frontier.
    n_near_target = max(len(pareto_rows), int(n_unique * near_pareto_frac))
    layers = _extract_fronts_until(F, n_near_target)
    seed_rows = np.concatenate(layers, axis=0)
    seed_X = X[seed_rows]
    seed_F = F[seed_rows]
    sort_s = time.perf_counter() - t0
    print(f"[Stage 2] Initial Pareto front: {len(pareto_rows)} points; "
          f"seed population (Pareto + near-Pareto): {len(seed_X)} points ({sort_s:.2f}s)")
    print(f"[Stage 2] Total: {time.perf_counter() - stage_t0:.2f}s")

    return {
        "seed_X": seed_X,
        "seed_F": seed_F,
        "archive_X": X,
        "archive_F": F,
        "pareto_X": X[pareto_rows],
        "pareto_F": F[pareto_rows],
        "stats": {
            "n_samples_lhs": int(n_raw),
            "n_valid_lhs": int(n_valid),
            "n_unique_lhs": int(n_unique),
            "n_pareto_lhs": int(len(pareto_rows)),
            "n_seed": int(len(seed_X)),
            "sample_s": round(sample_s, 2),
            "eval_s": round(eval_s, 2),
            "sort_s": round(sort_s, 2),
            "feature_build_s": round(timings.feature_build_s, 2),
            "model_forward_s": round(timings.model_forward_s, 2),
        },
    }


def random_valid_configs(
    n: int, rng: np.random.Generator, fix_bp_idx: int | None = None
) -> np.ndarray:
    """Draw ``n`` valid configs by LHS + repair. Optionally fixes the BP index."""
    X = lhs_sample(n, rng).astype(float)
    bp_idx = fix_bp_idx if fix_bp_idx is not None else 0
    repaired = HWRepair(bp_fixed_idx=bp_idx)._do(None, X)
    if fix_bp_idx is None:
        # HWRepair pins the BP column, so re-diversify when the caller didn't
        # explicitly ask for a fixed predictor.
        repaired[:, PARAM_COL_IDX["branch_predictor"]] = rng.integers(
            0, len(PARAM_VALUES["branch_predictor"]), size=len(repaired)
        )
    # Final safety check — repair is exhaustive, but guard against regressions.
    if not is_valid(repaired).all():
        raise RuntimeError("random_valid_configs produced invalid rows after repair")
    return repaired


# ---------------------------------------------------------------------------
# Stage 3: NSGA-II refinement
# ---------------------------------------------------------------------------

class HWRepair(Repair):
    """Round floats to ints, fix branch_predictor to the bifurcation value, and
    clamp pipeline/queue parameters so the constraint predicate holds."""

    def __init__(self, bp_fixed_idx: int) -> None:
        super().__init__()
        self.bp_fixed_idx = bp_fixed_idx

    def _do(self, problem: Problem, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X = np.round(X).astype(np.int64)
        # Clamp to per-parameter bounds.
        X = np.minimum(np.maximum(X, 0), (PARAM_N_VALUES - 1).reshape(1, -1))
        # Pin branch predictor for this bifurcation.
        X[:, PARAM_COL_IDX["branch_predictor"]] = self.bp_fixed_idx

        # Ensure fetch/decode/rename are high enough to satisfy the eventual
        # issue-width coherence constraint (total_issue ≤ ratio * rename).
        # Applied before ordering so the clamp below can't silently violate it.
        fetch_col = PARAM_COL_IDX["fetch_width"]
        decode_col = PARAM_COL_IDX["decode_width"]
        rename_col = PARAM_COL_IDX["rename_width"]
        commit_col = PARAM_COL_IDX["commit_width"]
        fu_cols = [
            PARAM_COL_IDX[p]
            for p in (
                "int_reg_issue_width",
                "int_mult_div_issue_width",
                "fp_reg_issue_width",
                "fp_mult_div_issue_width",
                "read_port_issue_width",
                "rdwr_port_issue_width",
            )
        ]
        min_rename_val = int(np.ceil(len(fu_cols) / ISSUE_WIDTH_RENAME_RATIO))
        rename_grid = _param_int_values("rename_width")
        min_rename_idx = int(np.searchsorted(rename_grid, min_rename_val))
        min_rename_idx = min(min_rename_idx, len(rename_grid) - 1)
        X[:, fetch_col] = np.maximum(X[:, fetch_col], min_rename_idx)
        X[:, decode_col] = np.maximum(X[:, decode_col], min_rename_idx)
        X[:, rename_col] = np.maximum(X[:, rename_col], min_rename_idx)

        # Pipeline width ordering: commit ≤ rename ≤ decode ≤ fetch.
        # Since fetch/decode/rename/commit share identical integer ranges
        # (1..12), clamping by index preserves the constraint.
        X[:, decode_col] = np.minimum(X[:, decode_col], X[:, fetch_col])
        X[:, rename_col] = np.minimum(X[:, rename_col], X[:, decode_col])
        X[:, commit_col] = np.minimum(X[:, commit_col], X[:, rename_col])

        # ROB sizing: rob_value ≥ ROB_HEADROOM * commit_value. rob index i maps
        # to value i+1 for our grid, so min index is (min_value - 1).
        commit_vals = _param_int_values("commit_width")[X[:, commit_col]]
        rob_vals = _param_int_values("rob_size")
        rob_col = PARAM_COL_IDX["rob_size"]
        rob_min_idx = np.clip(ROB_HEADROOM * commit_vals - 1, 0, len(rob_vals) - 1)
        X[:, rob_col] = np.maximum(X[:, rob_col], rob_min_idx)

        # Issue-width coherence: trim the largest FU widths until the total
        # fits within ISSUE_WIDTH_RENAME_RATIO * rename. Done before LSQ
        # balance since mem-port FU counts factor into the LSQ bound. FU
        # indices map to values idx+1 (all FU grids start at 1); total issue
        # = sum(indices) + len(fu_cols).
        rename_vals = _param_int_values("rename_width")[X[:, rename_col]]
        budget = np.floor(ISSUE_WIDTH_RENAME_RATIO * rename_vals).astype(np.int64)
        fu_sub = X[:, fu_cols]
        for _ in range(fu_sub.shape[1] * 8):  # bounded — each pass reduces sum by 1 per row
            over = (fu_sub.sum(axis=1) + len(fu_cols)) - budget
            mask = over > 0
            if not mask.any():
                break
            rows = np.nonzero(mask)[0]
            argmax = np.argmax(fu_sub[rows], axis=1)
            fu_sub[rows, argmax] = np.maximum(fu_sub[rows, argmax] - 1, 0)
        X[:, fu_cols] = fu_sub

        # LSQ balance: lq / sq each ≥ (read_port + rdwr_port).
        rp_col = PARAM_COL_IDX["read_port_issue_width"]
        rw_col = PARAM_COL_IDX["rdwr_port_issue_width"]
        rp_vals = _param_int_values("read_port_issue_width")[X[:, rp_col]]
        rw_vals = _param_int_values("rdwr_port_issue_width")[X[:, rw_col]]
        mem_ports = rp_vals + rw_vals
        lq_col = PARAM_COL_IDX["lq_entries"]
        sq_col = PARAM_COL_IDX["sq_entries"]
        lq_n = len(_param_int_values("lq_entries"))
        sq_n = len(_param_int_values("sq_entries"))
        lq_min_idx = np.clip(mem_ports - 1, 0, lq_n - 1)
        sq_min_idx = np.clip(mem_ports - 1, 0, sq_n - 1)
        X[:, lq_col] = np.maximum(X[:, lq_col], lq_min_idx)
        X[:, sq_col] = np.maximum(X[:, sq_col], sq_min_idx)
        return X


class HWConfigProblem(Problem):
    """Two-objective (CPI, cost) minimization over the encoded parameter grid."""

    def __init__(
        self,
        eval_fn: Callable[[np.ndarray], np.ndarray] | None,
    ) -> None:
        super().__init__(
            n_var=N_SEARCH_PARAMS,
            n_obj=2,
            xl=np.zeros(N_SEARCH_PARAMS, dtype=float),
            xu=(PARAM_N_VALUES - 1).astype(float),
            vtype=int,
        )
        self._eval_fn = eval_fn
        self.archive_X: list[np.ndarray] = []
        self.archive_F: list[np.ndarray] = []

    def _evaluate(self, X: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        X_int = np.round(X).astype(np.int64)
        F = self._eval_fn(X_int)
        out["F"] = F
        self.archive_X.append(X_int.copy())
        self.archive_F.append(F.copy())


def _prepare_bifurcation_seed(
    seed_X: np.ndarray, bp_idx: int, pop_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Subset the Stage 2 seed to configs matching ``bp_idx`` and top up to
    ``pop_size`` with fresh valid random draws if needed."""
    bp_col = PARAM_COL_IDX["branch_predictor"]
    mask = seed_X[:, bp_col] == bp_idx
    sub = seed_X[mask]
    if len(sub) > pop_size:
        keep = rng.choice(len(sub), size=pop_size, replace=False)
        sub = sub[keep]
    elif len(sub) < pop_size:
        needed = pop_size - len(sub)
        extra = random_valid_configs(needed, rng, fix_bp_idx=bp_idx)
        sub = np.concatenate([sub, extra], axis=0)
    return sub.astype(np.int64)


def stage3_nsga2(
    seed_X: np.ndarray,
    model: PeregrineMLModel,
    builder: FeatureBuilder,
    device: str,
    pop_size: int,
    n_generations: int,
    batch_size: int,
    rng: np.random.Generator,
    verbose: bool,
) -> dict[str, Any]:
    """Stage 3: bifurcated NSGA-II (one run per branch_predictor value)."""
    stage_t0 = time.perf_counter()

    mutation_prob = 1.0 / N_SEARCH_PARAMS
    archive_X: list[np.ndarray] = []
    archive_F: list[np.ndarray] = []
    per_bp_stats: dict[str, dict[str, Any]] = {}

    for bp_name in PARAM_VALUES["branch_predictor"]:
        bp_idx = PARAM_VALUES["branch_predictor"].index(bp_name)
        print(f"[Stage 3] NSGA-II bifurcation: branch_predictor={bp_name!r}")
        seed = _prepare_bifurcation_seed(seed_X, bp_idx, pop_size, rng)

        repair = HWRepair(bp_fixed_idx=bp_idx)
        problem = HWConfigProblem(eval_fn=None)  # attached below so we can capture timings
        bp_timings = EvalTimings()
        problem._eval_fn = lambda X_int, _t=bp_timings: evaluate_indices(  # noqa: E731
            X_int, model, builder, device, batch_size, _t
        )
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=seed.astype(float),
            crossover=SBX(prob=0.9, eta=15, vtype=float, repair=repair),
            mutation=PM(prob=mutation_prob, eta=20, vtype=float, repair=repair),
            eliminate_duplicates=True,
        )
        t0 = time.perf_counter()
        minimize(
            problem,
            algorithm,
            ("n_gen", n_generations),
            seed=int(rng.integers(0, 2**31 - 1)),
            verbose=verbose,
        )
        dur = time.perf_counter() - t0

        bp_X = np.concatenate(problem.archive_X, axis=0) if problem.archive_X else np.empty((0, N_SEARCH_PARAMS), dtype=np.int64)
        bp_F = np.concatenate(problem.archive_F, axis=0) if problem.archive_F else np.empty((0, 2), dtype=np.float64)
        archive_X.append(bp_X)
        archive_F.append(bp_F)
        ga_overhead_s = max(0.0, dur - bp_timings.feature_build_s - bp_timings.model_forward_s - bp_timings.cost_s)
        per_bp_stats[bp_name] = {
            "n_evaluated": int(len(bp_X)),
            "n_generations": int(n_generations),
            "pop_size": int(pop_size),
            "duration_s": round(dur, 2),
            "feature_build_s": round(bp_timings.feature_build_s, 2),
            "model_forward_s": round(bp_timings.model_forward_s, 2),
            "cost_s": round(bp_timings.cost_s, 2),
            "ga_overhead_s": round(ga_overhead_s, 2),
            "n_model_calls": bp_timings.n_model_calls,
        }
        print(
            f"[Stage 3] {bp_name!r} evaluated {len(bp_X)} configs in {dur:.2f}s "
            f"(features={bp_timings.feature_build_s:.2f}s model={bp_timings.model_forward_s:.2f}s "
            f"cost={bp_timings.cost_s:.2f}s ga={ga_overhead_s:.2f}s calls={bp_timings.n_model_calls})"
        )

    print(f"[Stage 3] Total: {time.perf_counter() - stage_t0:.2f}s")
    return {
        "archive_X": np.concatenate(archive_X, axis=0),
        "archive_F": np.concatenate(archive_F, axis=0),
        "stats": per_bp_stats,
    }


# ---------------------------------------------------------------------------
# Stage 4: final Pareto extraction, sensitivity, validation flagging
# ---------------------------------------------------------------------------

def merge_archive(*archives_X: np.ndarray) -> np.ndarray:
    return np.concatenate([a for a in archives_X if a.size > 0], axis=0)


def final_pareto(archive_X: np.ndarray, archive_F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate the archive, drop any infeasible slip-throughs, and extract
    the strict Pareto front."""
    if archive_X.size == 0:
        return archive_X, archive_F
    _, keep = np.unique(archive_X, axis=0, return_index=True)
    keep.sort()
    X = archive_X[keep]
    F = archive_F[keep]
    valid = is_valid(X)
    X = X[valid]
    F = F[valid]
    pareto_rows = pareto_front_2d(F)
    return X[pareto_rows], F[pareto_rows]


def compute_sensitivity(
    pareto_X: np.ndarray,
    pareto_F: np.ndarray,
    model: PeregrineMLModel,
    builder: FeatureBuilder,
    device: str,
    batch_size: int,
) -> tuple[list[dict[str, dict[str, float]]], int]:
    """For each Pareto config and each searchable parameter, perturb its index
    by ±1 (within bounds, respecting validity) and report the worst-case
    (max-|Δ|) change in CPI and cost. Empty dict for params at boundary or
    otherwise unperturbable without breaking feasibility."""
    if pareto_X.size == 0:
        return [], 0

    perturb_rows: list[np.ndarray] = []
    meta: list[tuple[int, int]] = []  # (pareto_idx, param_col)
    for i in range(len(pareto_X)):
        for j in range(N_SEARCH_PARAMS):
            for delta in (-1, +1):
                new_val = pareto_X[i, j] + delta
                if not (0 <= new_val < PARAM_N_VALUES[j]):
                    continue
                pert = pareto_X[i].copy()
                pert[j] = new_val
                if not bool(is_valid(pert.reshape(1, -1))[0]):
                    continue
                perturb_rows.append(pert)
                meta.append((i, j))

    if not perturb_rows:
        return [{} for _ in range(len(pareto_X))], 0

    perturb_X = np.stack(perturb_rows, axis=0)
    F_pert = evaluate_indices(perturb_X, model, builder, device, batch_size)
    n_sensitivity_configs = int(perturb_X.shape[0])

    buckets: list[dict[str, dict[str, list[float]]]] = [
        {p: {"dcpi": [], "dcost": []} for p in SEARCH_PARAM_ORDER}
        for _ in range(len(pareto_X))
    ]
    for k, (i, j) in enumerate(meta):
        param = SEARCH_PARAM_ORDER[j]
        buckets[i][param]["dcpi"].append(float(F_pert[k, 0] - pareto_F[i, 0]))
        buckets[i][param]["dcost"].append(float(F_pert[k, 1] - pareto_F[i, 1]))

    reduced: list[dict[str, dict[str, float]]] = []
    for b in buckets:
        entry: dict[str, dict[str, float]] = {}
        for p, d in b.items():
            if d["dcpi"]:
                entry[p] = {
                    "max_abs_dcpi": float(np.max(np.abs(d["dcpi"]))),
                    "max_abs_dcost": float(np.max(np.abs(d["dcost"]))),
                }
        reduced.append(entry)
    return reduced, n_sensitivity_configs


def flag_validation_candidates(pareto_F: np.ndarray, k: int) -> np.ndarray:
    """Flag Pareto points worth spending gem5 simulation budget on: the CPI
    optimum, the cost optimum, and up to ``k - 2`` evenly-spaced points along
    the cost axis."""
    n = len(pareto_F)
    flags = np.zeros(n, dtype=bool)
    if n == 0:
        return flags
    flags[int(np.argmin(pareto_F[:, 0]))] = True
    flags[int(np.argmin(pareto_F[:, 1]))] = True
    n_extra = max(0, k - 2)
    if n_extra > 0 and n >= 2:
        cost_order = np.argsort(pareto_F[:, 1])
        positions = np.linspace(0, n - 1, n_extra + 2)[1:-1].astype(int)
        for p in positions:
            flags[int(cost_order[p])] = True
    return flags


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def build_output(
    pareto_X: np.ndarray,
    pareto_F: np.ndarray,
    sensitivities: list[dict[str, dict[str, float]]],
    validation_flags: np.ndarray,
    stats: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    cfgs = decode_indices(pareto_X)
    front: list[dict[str, Any]] = []
    for i, cfg in enumerate(cfgs):
        front.append(
            {
                "predicted_cpi": float(pareto_F[i, 0]),
                "predicted_cost": float(pareto_F[i, 1]),
                "validation_candidate": bool(validation_flags[i]),
                "sensitivity": sensitivities[i] if i < len(sensitivities) else {},
                "config": cfg,
            }
        )
    return {"baseline": baseline, "pareto_front": front, "stats": stats}


def print_tradeoff_table(front: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
    print()
    print(f"{'#':>4} {'CPI':>10} {'Cost':>12}  {'Val?':>4}  BP     L1D/L1I/L2        Widths(F/D/R/C)")
    bc = baseline["config"]
    widths = f"{bc['fetch_width']}/{bc['decode_width']}/{bc['rename_width']}/{bc['commit_width']}"
    caches = f"{bc['l1d_size']}/{bc['l1i_size']}/{bc['l2_size']}"
    print(
        f"{'base':>4} {baseline['predicted_cpi']:>10.4f} {baseline['predicted_cost']:>12.2f}  "
        f"{'':>4}  {bc['branch_predictor']:<6} {caches:<18} {widths}"
    )
    for i, pt in enumerate(front):
        c = pt["config"]
        widths = f"{c['fetch_width']}/{c['decode_width']}/{c['rename_width']}/{c['commit_width']}"
        caches = f"{c['l1d_size']}/{c['l1i_size']}/{c['l2_size']}"
        print(
            f"{i:>4} {pt['predicted_cpi']:>10.4f} {pt['predicted_cost']:>12.2f}  "
            f"{'*' if pt['validation_candidate'] else '':>4}  "
            f"{c['branch_predictor']:<6} {caches:<18} {widths}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pareto-optimal hardware configuration search.")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Directory containing checkpoint.pt and scaler.joblib.")
    p.add_argument("--traces-dir", type=Path, required=True,
                   help="Parent of trace dirs, or one trace dir containing ronamol/.")
    p.add_argument("--output", type=Path, default=Path("pareto_front.json"),
                   help="Where to write the JSON Pareto front.")
    p.add_argument("--lhs-samples", type=int, default=256*1024,
                   help="Number of Latin Hypercube samples drawn in Stage 2.")
    p.add_argument("--near-pareto-frac", type=float, default=0.2,
                   help="Fraction of Stage 2 population to peel as the seed pool for Stage 3.")
    p.add_argument("--nsga-pop-size", type=int, default=400,
                   help="Population size per branch_predictor bifurcation in Stage 3.")
    p.add_argument("--nsga-generations", type=int, default=120,
                   help="Number of NSGA-II generations per bifurcation.")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Model-forward batch size (configs per chunk).")
    p.add_argument("--validation-k", type=int, default=5,
                   help="Number of Pareto points to tag as gem5 validation candidates.")
    p.add_argument("--skip-sensitivity", action="store_true",
                   help="Skip per-parameter sensitivity annotation on the final front.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-generation NSGA-II output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    total_t0 = time.perf_counter()

    t0 = time.perf_counter()
    ckpt_path, scaler_path = resolve_artifact_paths(args.checkpoint)
    bundles = [load_trace_bundle(p) for p in discover_trace_dirs(args.traces_dir)]
    print(f"Loaded {len(bundles)} trace bundle(s) ({time.perf_counter() - t0:.2f}s)")

    t0 = time.perf_counter()
    scaler: StandardScaler = joblib.load(scaler_path)
    if not hasattr(scaler, "feature_names_in_"):
        raise ValueError("Scaler must have been fit with a pandas DataFrame (feature_names_in_).")
    n_features = len(scaler.feature_names_in_)
    print(f"Loaded scaler ({n_features} features, {time.perf_counter() - t0:.2f}s)")

    t0 = time.perf_counter()
    builder = FeatureBuilder(bundles, scaler)
    print(f"Built feature LUTs ({time.perf_counter() - t0:.2f}s)")

    t0 = time.perf_counter()
    device = "xla"
    model = load_model(ckpt_path, input_size=n_features, device=device)
    print(f"Loaded model on {device!r} ({time.perf_counter() - t0:.2f}s)")

    # Warm up XLA compilation with a single baseline config.
    t0 = time.perf_counter()
    warmup_timings = EvalTimings()
    default_cpi = evaluate_configs(
        model, builder, [dict(DEFAULT_CONFIG)], device, warmup_timings
    )
    default_cost = hardware_cost([DEFAULT_CONFIG])[0]
    print(
        f"Baseline config: predicted CPI={default_cpi[0]:.4f}, cost={default_cost:.2f} "
        f"(warmup {time.perf_counter() - t0:.2f}s)"
    )
    n_warmup_configs = warmup_timings.n_cfgs_evaluated

    # -------------------- Stage 2 --------------------
    stage2 = stage2_global_sample(
        n_samples=args.lhs_samples,
        model=model,
        builder=builder,
        device=device,
        rng=rng,
        batch_size=args.batch_size,
        near_pareto_frac=args.near_pareto_frac,
    )

    # -------------------- Stage 3 --------------------
    stage3 = stage3_nsga2(
        seed_X=stage2["seed_X"],
        model=model,
        builder=builder,
        device=device,
        pop_size=args.nsga_pop_size,
        n_generations=args.nsga_generations,
        batch_size=args.batch_size,
        rng=rng,
        verbose=not args.quiet,
    )

    # -------------------- Stage 4 --------------------
    stage4_t0 = time.perf_counter()
    t0 = time.perf_counter()
    full_X = merge_archive(stage2["archive_X"], stage3["archive_X"])
    full_F = merge_archive(stage2["archive_F"], stage3["archive_F"])
    pareto_X, pareto_F = final_pareto(full_X, full_F)
    # Sort ascending by cost so the tradeoff curve reads left-to-right.
    order = np.argsort(pareto_F[:, 1])
    pareto_X = pareto_X[order]
    pareto_F = pareto_F[order]
    print(
        f"[Stage 4] Final Pareto front: {len(pareto_X)} points "
        f"(from archive of {len(full_X)}, {time.perf_counter() - t0:.2f}s)"
    )

    if args.skip_sensitivity:
        sensitivities: list[dict[str, dict[str, float]]] = [{} for _ in range(len(pareto_X))]
        n_sensitivity_configs = 0
    else:
        t0 = time.perf_counter()
        print("[Stage 4] Computing per-parameter sensitivity")
        sensitivities, n_sensitivity_configs = compute_sensitivity(
            pareto_X, pareto_F, model, builder, device, args.batch_size
        )
        print(f"[Stage 4] Sensitivity computed in {time.perf_counter() - t0:.2f}s")

    flags = flag_validation_candidates(pareto_F, k=args.validation_k)
    print(f"[Stage 4] Total: {time.perf_counter() - stage4_t0:.2f}s")

    n_bundles = builder.n_bundles
    n_stage2_configs = int(stage2["stats"]["n_unique_lhs"])
    n_stage3_configs = sum(int(s["n_evaluated"]) for s in stage3["stats"].values())
    n_total_configs = (
        n_warmup_configs + n_stage2_configs + n_stage3_configs + n_sensitivity_configs
    )
    n_total_inference_points = n_total_configs * n_bundles
    print(
        f"Total inference: {n_total_configs} configs × {n_bundles} program points "
        f"= {n_total_inference_points} (warmup={n_warmup_configs}, stage2={n_stage2_configs}, "
        f"stage3={n_stage3_configs}, sensitivity={n_sensitivity_configs})"
    )

    stats = {
        "stage2": stage2["stats"],
        "stage3": stage3["stats"],
        "n_archive_total": int(len(full_X)),
        "n_pareto_final": int(len(pareto_X)),
        "n_validation_candidates": int(int(flags.sum())),
        "n_program_points": int(n_bundles),
        "n_inference_configs": {
            "warmup": int(n_warmup_configs),
            "stage2": int(n_stage2_configs),
            "stage3": int(n_stage3_configs),
            "sensitivity": int(n_sensitivity_configs),
            "total": int(n_total_configs),
        },
        "n_total_inference_points": int(n_total_inference_points),
        "total_duration_s": round(time.perf_counter() - total_t0, 2),
    }
    baseline = {
        "predicted_cpi": float(default_cpi[0]),
        "predicted_cost": float(default_cost),
        "config": dict(DEFAULT_CONFIG),
    }
    output = build_output(pareto_X, pareto_F, sensitivities, flags, stats, baseline)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print_tradeoff_table(output["pareto_front"], output["baseline"])
    print(f"\nWrote {len(output['pareto_front'])} Pareto points to {args.output}")
    print(f"Total runtime: {time.perf_counter() - total_t0:.2f}s")


if __name__ == "__main__":
    main()
