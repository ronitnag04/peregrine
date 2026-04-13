"""
gen_bp_rate.py

Runs branch predictor simulation for a trace and branch predictor type.

Usage: python gen_bp_rate.py \
            --trace <trace.csv> \
            --branch-predictor <bp_type>

Outputs:
    <trace_dir>/bp_rates/<bp_type>_bp_rate.npy
"""

import os
import argparse

import numpy as np

from evantrace.parser import Parser
from evantrace.bp_sim import BPSim
from evantrace.branch_predictor import LocalBranchPredictor, TAGEBranchPredictor

def run_sim(trace_filename: str, branch_predictor_type: str):
    parser = Parser(trace_filename)
    
    if branch_predictor_type == "local":
        bp = LocalBranchPredictor()
    elif branch_predictor_type == "tage":
        bp = TAGEBranchPredictor()
    else:
        raise ValueError(f"Invalid branch predictor type: {branch_predictor_type}")
    parser = Parser(trace_filename)
    trace_iter = parser.iter_instructions()
    BPSim(trace=trace_iter, branch_predictor=bp).run()
    rate = bp.get_misprediction_rate()
    return rate


def main():
    parser = argparse.ArgumentParser(description="Run branch predictor simulation for a trace and branch predictor type.")
    parser.add_argument("--trace", type=str, required=True, help="Path to the trace file.")
    parser.add_argument("--branch-predictor", type=str, required=True, choices=["local", "tage"], help="Branch predictor type.")
    args = parser.parse_args()

    if not os.path.exists(args.trace):
        raise FileNotFoundError(f"Trace file not found: {args.trace}")

    trace_dir = os.path.dirname(args.trace)
    output_dir = os.path.join(trace_dir, "bp_rates")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.branch_predictor}_bp_rate.npy")
    rate = run_sim(args.trace, args.branch_predictor)
    rate_array = np.array(rate, dtype=np.float32)
    np.save(output_file, rate_array)


if __name__ == "__main__":
    main()
