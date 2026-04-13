"""
gen_cache_latency.py

Runs the cache simulation for a benchmark and cache configuration, and writes the results to a .npy file.

Usage: python gen_cache_latencies.py \
            --trace <trace.csv> \
            --l1i-size <l1i_kb> \
            --l1d-size <l1d_kb> \
            --l2-size <l2_kb> \

Outputs:
    <trace_dir>/cache_latencies/l1i_<l1i_kb>_l1d_<l1d_kb>_l2_<l2_kb>_cache_latencies.npy
"""

import os
import argparse

import numpy as np

from evantrace.parser import Parser
from evantrace.caches import Cache
from evantrace.cache_sim import CacheSim

L1_KB = [16, 32, 64, 128, 256]  # L1I and L1D
L2_KB = [512, 1024, 2048, 4096]  # L2

# Cache fixed parameters: match peregrine-gem5 L1/L2 defaults (tag+data cycles).
# gem5 L1ICache/L1DCache: tag_latency=1, data_latency=1 → 2 cycles. L2Cache: tag=10, data=10 → 20.
# Paper: L1=4, L2=10, RAM=200. If we rerun gem5 to match paper, revert to those.
L1_ASSOC = 8
L2_ASSOC = 16
L1_READ_LAT = 2   # gem5 L1 hit: tag_latency + data_latency = 1 + 1
L2_READ_LAT = 20  # gem5 L2 hit: tag_latency + data_latency = 10 + 10


def run_sim(trace_filename: str, l1i_kb: int, l1d_kb: int, l2_kb: int):
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

    parser = Parser(trace_filename)
    trace_iter = parser.iter_instructions()
    latencies = list(
        CacheSim(
            trace=trace_iter,
            icache=icache,
            dcache=dcache,
            l2cache=l2cache,
        ).run()
    )
    return latencies  # shape (N, 2): col 0 = fetch_latency, col 1 = exec_latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--l1i-size", type=int, default=32, help="L1I size in KB", choices=L1_KB)
    parser.add_argument("--l1d-size", type=int, default=32, help="L1D size in KB", choices=L1_KB)
    parser.add_argument("--l2-size", type=int, default=256, help="L2 size in KB", choices=L2_KB)
    args = parser.parse_args()  

    if not os.path.exists(args.trace):
        raise FileNotFoundError(f"Trace file not found: {args.trace}")

    trace_dir = os.path.dirname(args.trace)
    output_dir = os.path.join(trace_dir, "cache_latencies")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"l1i_{args.l1i_size}_l1d_{args.l1d_size}_l2_{args.l2_size}_cache_latencies.npy")

    latencies = run_sim(args.trace, args.l1i_size, args.l1d_size, args.l2_size)
    latencies_array = np.array(latencies, dtype=np.uint16)
    np.save(output_file, latencies_array)


if __name__ == "__main__":
    main()
