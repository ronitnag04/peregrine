"""
gen_bp_rates.py

Runs branch predictor simulation for a trace and writes a JSON file with
the misprediction rate for each predictor type.

Output: <trace_dir>/<trace_stem>_bp.json
  {"local": <float>, "tage": <float>}  (rates in [0, 1])

Usage: python gen_bp_rates.py <trace.csv>
"""

import sys
import json
import os

from evantrace.parser import Parser
from evantrace.bp_sim import BPSim
from evantrace.branch_predictor import LocalBranchPredictor, TAGEBranchPredictor

def main():
    if len(sys.argv) != 2:
        print("Usage: python gen_bp_rates.py <trace.csv>")
        sys.exit(1)

    trace_filename = sys.argv[1]
    print(f"Parsing: {trace_filename}")
    instructions = Parser(trace_filename).parse()
    if not instructions:
        print("Error: no instructions parsed — check trace file.")
        sys.exit(1)
    print(f"  {len(instructions)} instructions")

    trace_dir = os.path.dirname(os.path.abspath(trace_filename))
    stem = os.path.splitext(os.path.basename(trace_filename))[0]
    out_bp = os.path.join(trace_dir, f"{stem}_bp.json")

    bp_results = {}
    for bp_name in ["local", "tage"]:
        if bp_name == "local":
            bp = LocalBranchPredictor()
        else:
            bp = TAGEBranchPredictor()
        BPSim(trace=instructions, branch_predictor=bp).run()
        rate = bp.get_misprediction_rate()
        bp_results[bp_name] = rate
        print(f"  {bp_name}: {rate:.4f} misprediction rate")

    with open(out_bp, "w") as f:
        json.dump(bp_results, f, indent=2)

    print(f"Done. → {out_bp}")


if __name__ == "__main__":
    main()
