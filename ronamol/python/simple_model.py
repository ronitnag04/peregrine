"""
Ronamol simple analytical model (fresh start).

Computes a compact, bottleneck-aligned analytical feature vector from an
evantrace CSV trace, plus (optionally) per-cache-config latency summaries.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

try:
    from evantrace.parser import Parser
except ModuleNotFoundError:  # allow running directly without PYTHONPATH
    _HERE = Path(__file__).resolve()
    _PEREGRINE_ROOT = _HERE.parents[2]
    sys.path.insert(0, str(_PEREGRINE_ROOT))
    from evantrace.parser import Parser  # type: ignore


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _percentiles(x: np.ndarray, ps: Iterable[float]) -> Dict[str, float]:
    x = np.asarray(x)
    if x.size == 0:
        return {f"p{int(p)}": 0.0 for p in ps}
    vals = np.percentile(x, list(ps))
    return {f"p{int(p)}": float(v) for p, v in zip(ps, vals)}


def _find_sidecar(trace_csv: Path, candidates: list[str]) -> Optional[Path]:
    """
    Return the first existing file among:
      - trace_dir/<candidate>
      - trace_dir/<stem><candidate_suffix>
    """
    trace_dir = trace_csv.parent
    stem = trace_csv.stem
    for c in candidates:
        p = trace_dir / c
        if p.exists():
            return p
        if c.startswith("_"):
            p2 = trace_dir / f"{stem}{c}"
            if p2.exists():
                return p2
    return None


@dataclass
class ProgramFeatures:
    # --- Instruction mix ---
    frac_int_alu: float
    frac_int_mult_div: float
    frac_fp_alu: float
    frac_fp_mult_div: float
    frac_simd: float
    frac_load: float
    frac_store: float
    frac_branch: float
    frac_other: float

    # --- Frontend / control-flow ---
    branch_rate: float
    mean_basic_block_size: float
    p50_basic_block_size: float
    p95_basic_block_size: float
    frac_sync_instructions: float
    frac_indirect_branches: float
    frac_direct_cond_branches: float
    frac_direct_uncond_branches: float

    # --- ILP / dependency structure ---
    mean_reg_dep_distance: float
    p50_reg_dep_distance: float
    p95_reg_dep_distance: float
    frac_independent_last_16: float
    crit_path_density_10: float
    mean_reg_fan_out: float

    # --- Memory access patterns / hazards ---
    memory_instruction_fraction: float
    store_to_load_ratio: float
    unique_load_pages: float
    unique_store_pages: float
    frac_mem_dependent: float
    frac_memory_ordering_hazards: float
    load_stride_regularity: float

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}


def compute_program_features(
    trace_csv: str | Path,
    dep_window: int = 16,
    crit_threshold: int = 10,
    stride_min_samples: int = 16,
) -> ProgramFeatures:
    trace_csv = Path(trace_csv)
    parser = Parser(str(trace_csv))

    total = 0
    n_load = 0
    n_store = 0
    n_branch = 0
    n_sync = 0
    n_int_alu = 0
    n_int_md = 0
    n_fp_alu = 0
    n_fp_md = 0
    n_simd = 0
    n_other = 0

    n_indirect = 0
    n_direct_cond = 0
    n_direct_uncond = 0

    bb_sizes: list[int] = []
    cur_bb = 0

    last_seen_ip: dict[int, int] = {}
    reg_dep_dists: list[int] = []
    indep_count = 0
    crit_count = 0
    producer_fanout: dict[int, int] = {}

    load_pages: set[int] = set()
    store_pages: set[int] = set()
    mem_ops = 0
    mem_dep_ops = 0
    mem_order_hazard_ops = 0

    last_load_addr_by_ip: dict[int, int] = {}
    stride_stats_by_ip: dict[int, dict[int, int]] = {}
    load_samples_by_ip: dict[int, int] = {}

    for idx, inst in enumerate(parser.iter_instructions()):
        total += 1
        cur_bb += 1

        bucket = inst.fu_group
        if bucket == "read_port":
            n_load += 1
        elif bucket == "rdwr_port":
            n_store += 1
        elif bucket == "int_mult_div":
            n_int_md += 1
        elif bucket == "fp_alu":
            n_fp_alu += 1
        elif bucket == "fp_mult_div":
            n_fp_md += 1
        elif bucket == "int_alu":
            n_int_alu += 1
        elif bucket == "simd_unit":
            n_simd += 1
        elif bucket == "other":
            n_other += 1
        else:
            raise ValueError(f"Unknown FU group: {bucket}")

        if inst.inst_sync:
            n_sync += 1

        if inst.branch_type is not None:
            n_branch += 1
            bt = inst.branch_type.name if hasattr(inst.branch_type, "name") else str(inst.branch_type)
            if bt == "indirect":
                n_indirect += 1
            elif bt == "direct_conditional":
                n_direct_cond += 1
            elif bt == "direct_unconditional":
                n_direct_uncond += 1
            if inst.branch_taken:
                bb_sizes.append(cur_bb)
                cur_bb = 0

        # --- deps (approx via producer IP last-seen dynamic index) ---
        max_dist = None
        deps = getattr(inst, "reg_dependent_ips", []) or []
        if deps:
            for dep_ip_u64 in deps:
                dep_ip = int(dep_ip_u64)
                prev = last_seen_ip.get(dep_ip)
                if prev is not None:
                    d = idx - prev
                    reg_dep_dists.append(d)
                    max_dist = d if max_dist is None else max(max_dist, d)
                producer_fanout[dep_ip] = producer_fanout.get(dep_ip, 0) + 1

        if max_dist is None:
            indep_count += 1
        else:
            if max_dist > dep_window:
                indep_count += 1
            if max_dist > crit_threshold:
                crit_count += 1

        last_seen_ip[int(inst.inst_ptr)] = idx

        # --- memory ---
        is_load = len(inst.read_addrs) > 0
        is_store = len(inst.write_addrs) > 0
        if is_load or is_store:
            mem_ops += 1
            if inst.mem_dependent_ips:
                mem_dep_ops += 1
                if len(inst.mem_dependent_ips) > 1:
                    mem_order_hazard_ops += 1

        if is_load:
            a0 = int(inst.read_addrs[0])
            load_pages.add(a0 >> 12)
            ip = int(inst.inst_ptr)
            load_samples_by_ip[ip] = load_samples_by_ip.get(ip, 0) + 1
            prev_addr = last_load_addr_by_ip.get(ip)
            if prev_addr is not None:
                delta = a0 - prev_addr
                dct = stride_stats_by_ip.get(ip)
                if dct is None:
                    dct = {}
                    stride_stats_by_ip[ip] = dct
                dct[delta] = dct.get(delta, 0) + 1
            last_load_addr_by_ip[ip] = a0

        if is_store:
            a0 = int(inst.write_addrs[0])
            store_pages.add(a0 >> 12)

    if cur_bb > 0:
        bb_sizes.append(cur_bb)

    stride_reg_scores: list[float] = []
    for ip, dct in stride_stats_by_ip.items():
        samples = load_samples_by_ip.get(ip, 0)
        if samples < stride_min_samples:
            continue
        total_deltas = sum(dct.values())
        if total_deltas == 0:
            continue
        best = max(dct.values())
        stride_reg_scores.append(best / total_deltas)
    load_stride_regularity = float(np.mean(stride_reg_scores)) if stride_reg_scores else 0.0

    bb_arr = np.asarray(bb_sizes, dtype=np.int32)
    dep_arr = np.asarray(reg_dep_dists, dtype=np.int32)

    bb_ps = _percentiles(bb_arr, [50, 95])
    dep_ps = _percentiles(dep_arr, [50, 95])

    mean_bb = float(bb_arr.mean()) if bb_arr.size else 0.0
    mean_dep = float(dep_arr.mean()) if dep_arr.size else 0.0
    branch_rate = _safe_div(n_branch, total)

    mean_fanout = _safe_div(sum(producer_fanout.values()), len(producer_fanout))
    memory_instruction_fraction = _safe_div(n_load + n_store, total)

    return ProgramFeatures(
        frac_int_alu=_safe_div(n_int_alu, total),
        frac_int_mult_div=_safe_div(n_int_md, total),
        frac_fp_alu=_safe_div(n_fp_alu, total),
        frac_fp_mult_div=_safe_div(n_fp_md, total),
        frac_simd=_safe_div(n_simd, total),
        frac_load=_safe_div(n_load, total),
        frac_store=_safe_div(n_store, total),
        frac_branch=_safe_div(n_branch, total),
        frac_other=_safe_div(n_other, total),
        branch_rate=branch_rate,
        mean_basic_block_size=mean_bb,
        p50_basic_block_size=float(bb_ps["p50"]),
        p95_basic_block_size=float(bb_ps["p95"]),
        frac_sync_instructions=_safe_div(n_sync, total),
        frac_indirect_branches=_safe_div(n_indirect, n_branch),
        frac_direct_cond_branches=_safe_div(n_direct_cond, n_branch),
        frac_direct_uncond_branches=_safe_div(n_direct_uncond, n_branch),
        mean_reg_dep_distance=mean_dep,
        p50_reg_dep_distance=float(dep_ps["p50"]),
        p95_reg_dep_distance=float(dep_ps["p95"]),
        frac_independent_last_16=_safe_div(indep_count, total),
        crit_path_density_10=_safe_div(crit_count, total),
        mean_reg_fan_out=float(mean_fanout),
        memory_instruction_fraction=float(memory_instruction_fraction),
        store_to_load_ratio=_safe_div(n_store, n_load),
        unique_load_pages=float(len(load_pages)),
        unique_store_pages=float(len(store_pages)),
        frac_mem_dependent=_safe_div(mem_dep_ops, mem_ops),
        frac_memory_ordering_hazards=_safe_div(mem_order_hazard_ops, mem_ops),
        load_stride_regularity=float(load_stride_regularity),
    )


def summarize_cache_latencies(
    trace_csv: str | Path,
    latencies_npy: str | Path | None = None,
    configs_json: str | Path | None = None,
    percentiles: tuple[int, ...] = (50, 75, 95),
) -> list[dict[str, Any]]:
    trace_csv = Path(trace_csv)

    if latencies_npy is None:
        lat_path = _find_sidecar(trace_csv, ["trace_latencies.npy", "_latencies.npy"])
    else:
        lat_path = Path(latencies_npy)
    if lat_path is None or not lat_path.exists():
        raise FileNotFoundError(
            f"Could not find latencies .npy for trace {trace_csv}. "
            "Expected trace_latencies.npy or <stem>_latencies.npy (or pass --latencies-npy)."
        )

    if configs_json is None:
        cfg_path = _find_sidecar(trace_csv, ["trace_configs.json", "_configs.json"])
    else:
        cfg_path = Path(configs_json)
    if cfg_path is None or not cfg_path.exists():
        raise FileNotFoundError(
            f"Could not find configs .json for trace {trace_csv}. "
            "Expected trace_configs.json or <stem>_configs.json (or pass --configs-json)."
        )

    meta = json.loads(cfg_path.read_text())
    configs = meta.get("configs", [])

    mmap = np.load(lat_path, mmap_mode="r")  # shape (C, N, 2)
    if mmap.ndim != 3 or mmap.shape[2] != 2:
        raise ValueError(f"{lat_path}: expected shape (C, N, 2), got {mmap.shape}")

    rows: list[dict[str, Any]] = []
    C = int(mmap.shape[0])
    ps = list(percentiles)
    for i in range(C):
        fetch = np.asarray(mmap[i, :, 0], dtype=np.float32)
        exe = np.asarray(mmap[i, :, 1], dtype=np.float32)
        r: dict[str, Any] = {"config_idx": i}
        if i < len(configs):
            l1i, l1d, l2 = configs[i]
            r.update({"l1i_kb": int(l1i), "l1d_kb": int(l1d), "l2_kb": int(l2)})
        r["fetch_mean"] = float(fetch.mean()) if fetch.size else 0.0
        r["exec_mean"] = float(exe.mean()) if exe.size else 0.0
        for p, v in zip(ps, np.percentile(fetch, ps) if fetch.size else [0.0] * len(ps)):
            r[f"fetch_p{int(p)}"] = float(v)
        for p, v in zip(ps, np.percentile(exe, ps) if exe.size else [0.0] * len(ps)):
            r[f"exec_p{int(p)}"] = float(v)
        rows.append(r)
    return rows

