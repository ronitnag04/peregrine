"""
gen_br_rates.py

Runs the branch predictor simulation for all combinations of branch predictor parameters and 
writes ONE .npy file per trace containing total misprediction rate for every config.

Output shape: (N_configs) float
  axis 0: config index

Usage: python gen_br_rates.py <trace.csv>
Output: <trace_dir>/<trace_stem>_br_rates.npy
        <trace_dir>/<trace_stem>_configs.json
"""

import sys
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import numpy as np

from evantrace.parser import Parser
from evantrace.bp_sim import BPSim
from evantrace.branch_predictor import BranchPredictor, LocalBranchPredictor, TAGEBranchPredictor

# ── per-process shared state (set once via initializer) ──────────────────────
_instructions = None


def _init_worker(instr_list):
    global _instructions
    _instructions = instr_list


def _worker(args):
    """Run one branch predictor config in a worker process. Returns (config_index, float)."""
    idx, bp_id = args
    misprediction_rate = run_bp_sim(_instructions, bp_id)
    return idx, misprediction_rate

def run_bp_sim(instructions, bp_id: int):
    if bp_id == 0:
        bp = LocalBranchPredictor(local_predictor_size=2048, local_ctr_bits=2)
    else:
        bp = TAGEBranchPredictor()
    bpsim = BPSim(
        trace=instructions,
        branch_predictor=bp
    )
    bpsim.run()
    return bp.get_misprediction_rate()


def main():
    if len(sys.argv) != 2:
        print("Usage: python gen_br_rates.py <trace.csv>")
        sys.exit(1)

    trace_filename = sys.argv[1]
    print(f"Parsing: {trace_filename}")
    instructions = Parser(trace_filename).parse()
    if not instructions:
        print("Error: no instructions parsed — check trace file.")
        sys.exit(1)
    print(f"  {len(instructions)} instructions")

    configs = list(product(range(2)))
    n = len(configs)
    n_workers = os.cpu_count() or 1
    print(f"Running {n} branch predictor configurations on {n_workers} workers...")

    trace_dir = os.path.dirname(os.path.abspath(trace_filename))
    stem = os.path.splitext(os.path.basename(trace_filename))[0]
    out_npy = os.path.join(trace_dir, f"{stem}_br_rates.npy")
    out_json = os.path.join(trace_dir, f"{stem}_br_configs.json")
    tmp_path = os.path.join(trace_dir, f".{stem}_br_rates_tmp.npy")

    tasks = [(i, bp_id) for i, bp_id in enumerate(configs)]

    renamed = False
    mmap = None
    try:
        # Memmap shape: (N_configs) — one misprediction rate per config.
        mmap = np.lib.format.open_memmap(
            tmp_path, mode="w+", dtype=np.float32, shape=(n,)
        )
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(instructions,),
        ) as executor:
            futures = {executor.submit(_worker, task): task[0] for task in tasks}
            done = 0
            for future in as_completed(futures):
                idx, misprediction_rate = future.result()
                mmap[idx] = misprediction_rate
                done += 1
                bp_id = configs[idx]
                print(f"  [{done:3d}/{n}] {bp_id}")

        # Flush and close before rename
        mmap = None

        os.rename(tmp_path, out_npy)
        renamed = True

        with open(out_json, "w") as f:
            json.dump(
                {
                    "configs": [list(c) for c in configs],
                    "fields": ["bp_id"],
                    "shape": [n],
                    "dtype": "float32",
                },
                f,
                indent=2,
            )

        print(f"Done.  {n} branch predictor configs → {out_npy}")
        print(f"       config map → {out_json}")

    finally:
        mmap = None  # release memmap handle before any file ops
        if not renamed and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
