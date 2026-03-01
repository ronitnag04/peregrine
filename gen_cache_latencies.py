"""
gen_cache_latencies.py

Runs the cache simulation for all combinations of cache parameters and writes
ONE .npy file per trace containing per-instruction latencies for every config.

Output shape: (N_configs, N_instructions, 2) uint16
  axis 0: config index (product of L1_KB × L1_KB × L2_KB)
  axis 1: instruction index
  axis 2: [0] = fetch_latency, [1] = exec_latency

Config index formula (matches anamol C++ lookup):
  idx = l1i_rank*20 + l1d_rank*4 + l2_rank
  where *_rank = position in the sorted grid list.

BP simulation is independent of cache configs and is handled separately
by gen_bp_rates.py.

Usage: python gen_cache_latencies.py <trace.csv>
Output: <trace_dir>/<trace_stem>_latencies.npy   (same directory as input trace)
        <trace_dir>/<trace_stem>_configs.json
"""

import sys
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import numpy as np

from evantrace.parser import Parser
from evantrace.caches import Cache
from evantrace.cache_sim import CacheSim

# ── per-process shared state (set once via initializer) ──────────────────────
_instructions = None


def _init_worker(instr_list):
    global _instructions
    _instructions = instr_list


def _worker(args):
    """Run one cache config in a worker process. Returns (config_index, uint16 array)."""
    idx, l1i_kb, l1d_kb, l2_kb = args
    latencies = run_sim(_instructions, l1i_kb, l1d_kb, l2_kb)
    # shape (N, 2): col 0 = fetch_latency, col 1 = exec_latency
    return idx, np.array(latencies, dtype=np.uint16)


# Parameter grids (KB), matching params_gen.h
L1_KB = [16, 32, 64, 128, 256]  # L1I and L1D
L2_KB = [512, 1024, 2048, 4096]  # L2

# Cache fixed parameters
L1_ASSOC = 8
L2_ASSOC = 16
L1_READ_LAT = 4
L2_READ_LAT = 12


def run_sim(instructions, l1i_kb: int, l1d_kb: int, l2_kb: int):
    """Simulate one cache config. Returns list of (fetch_latency, exec_latency)."""
    l2cache = Cache(
        associativity=L2_ASSOC,
        total_size=l2_kb * 1024,
        read_latency=L2_READ_LAT,
    )
    icache = Cache(
        associativity=L1_ASSOC,
        total_size=l1i_kb * 1024,
        read_latency=L1_READ_LAT,
        parent=l2cache,
    )
    dcache = Cache(
        associativity=L1_ASSOC,
        total_size=l1d_kb * 1024,
        read_latency=L1_READ_LAT,
        parent=l2cache,
    )
    CacheSim(
        trace=instructions,
        icache=icache,
        dcache=dcache,
        l2cache=l2cache,
    ).run()
    return [(instr.fetch_latency, instr.exec_latency) for instr in instructions]


def main():
    if len(sys.argv) != 2:
        print("Usage: python gen_cache_latencies.py <trace.csv>")
        sys.exit(1)

    trace_filename = sys.argv[1]
    print(f"Parsing: {trace_filename}")
    instructions = Parser(trace_filename).parse()
    if not instructions:
        print("Error: no instructions parsed — check trace file.")
        sys.exit(1)
    print(f"  {len(instructions)} instructions")

    configs = list(product(L1_KB, L1_KB, L2_KB))
    n = len(configs)
    n_workers = os.cpu_count() or 1
    print(f"Running {n} cache configurations on {n_workers} workers...")

    trace_dir = os.path.dirname(os.path.abspath(trace_filename))
    stem = os.path.splitext(os.path.basename(trace_filename))[0]
    out_npy = os.path.join(trace_dir, f"{stem}_latencies.npy")
    out_json = os.path.join(trace_dir, f"{stem}_configs.json")
    tmp_path = os.path.join(trace_dir, f".{stem}_latencies_tmp.npy")

    N = len(instructions)
    tasks = [(i, l1i, l1d, l2) for i, (l1i, l1d, l2) in enumerate(configs)]

    renamed = False
    mmap = None
    try:
        # Memmap shape: (N_configs, N_instructions, 2) — configs-first for contiguous
        # per-config reads in C++.  uint16 fits latencies up to 65535 cycles.
        mmap = np.lib.format.open_memmap(
            tmp_path, mode="w+", dtype=np.uint16, shape=(n, N, 2)
        )
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(instructions,),
        ) as executor:
            futures = {executor.submit(_worker, task): task[0] for task in tasks}
            done = 0
            for future in as_completed(futures):
                idx, arr = future.result()  # arr: uint16 array (N, 2)
                mmap[idx, :, :] = arr  # contiguous write; arr freed immediately
                done += 1
                l1i, l1d, l2 = configs[idx]
                print(f"  [{done:3d}/{n}] {l1i}_{l1d}_{l2}")

        # Flush and close before rename
        mmap = None

        os.rename(tmp_path, out_npy)
        renamed = True

        with open(out_json, "w") as f:
            json.dump(
                {
                    "configs": [list(c) for c in configs],
                    "fields": ["l1i_kb", "l1d_kb", "l2_kb"],
                    "shape": [n, N, 2],
                    "dtype": "uint16",
                },
                f,
                indent=2,
            )

        print(f"Done.  {n} configs × {N} instructions × 2 → {out_npy}")
        print(f"       config map → {out_json}")

    finally:
        mmap = None  # release memmap handle before any file ops
        if not renamed and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
