"""
Consolidate per-trace training data into a single training matrix.

Reads all .pkl files from the training directory (skipping all_traces.pkl),
adds a 'trace_name' column to each, then concatenates row-wise into a single
DataFrame and saves it.

Usage:
  python consolidate_training.py -i training/ -o training/all_traces.pkl
"""

import argparse
import pandas as pd
from pathlib import Path


def consolidate(input_dir: str, output_path: str) -> pd.DataFrame:
    input_path = Path(input_dir)
    out = Path(output_path)

    pkls = sorted(p for p in input_path.glob("*.pkl") if p.name != out.name)
    if not pkls:
        raise RuntimeError(f"No .pkl files found in {input_path}")

    print(f"Found {len(pkls)} trace file(s):")
    dfs = []
    for pkl in pkls:
        df = pd.read_pickle(pkl)
        df["trace_name"] = pkl.stem
        dfs.append(df)
        print(f"  {pkl.name}: {df.shape[0]} rows x {df.shape[1]-1} features")

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_pickle(out)

    print(f"\n✓ Consolidated matrix: {combined.shape[0]} rows x {combined.shape[1]} columns")
    print(f"✓ Saved to: {out}")
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate per-trace training .pkl files into one matrix"
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, default="training",
        help="Directory containing per-trace .pkl files (default: training)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="training/all_traces.pkl",
        help="Output path for the consolidated matrix (default: training/all_traces.pkl)"
    )
    args = parser.parse_args()
    consolidate(args.input_dir, args.output)


if __name__ == "__main__":
    main()
