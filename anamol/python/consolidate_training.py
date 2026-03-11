"""
Consolidate per-trace training data into a single CSV.

Streams all .csv files from the training directory (skipping the output file)
and writes rows directly to the output.
Uses constant memory (one row at a time).

Usage:
  python consolidate_training.py -i training/ -o training/all_benchmarks.csv
"""

import argparse
import csv
import pandas as pd
from pathlib import Path


def consolidate_csv_files(input_dir: str, output_path: str) -> None:
    input_path = Path(input_dir)
    out = Path(output_path)

    csvs = sorted(p for p in input_path.glob("*.csv") if p.name != out.name)
    if not csvs:
        raise RuntimeError(f"No .csv files found in {input_path}")

    print(f"Found {len(csvs)} trace file(s):")
    total_rows = 0
    header_written = False
    reference_header = None

    with open(out, "w", newline="") as outf:
        writer = None
        for csv_path in csvs:
            with open(csv_path, newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                if reference_header is None:
                    reference_header = header
                elif header != reference_header:
                    raise ValueError(
                        f"Header in {csv_path.name} does not match first file. "
                        f"First file: {reference_header}. Got: {header}."
                    )
                if not header_written:
                    writer = csv.writer(outf)
                    writer.writerow(header)
                    header_written = True
                file_rows = 0
                for row in reader:
                    if len(row) != len(header):
                        raise ValueError(
                            f"{csv_path.name}: row has {len(row)} columns, expected {len(header)} (header: {header})"
                        )
                    writer.writerow(row)
                    file_rows += 1
                total_rows += file_rows
            print(f"  {csv_path.name}: {file_rows} rows")

    n_cols = len(reference_header) if reference_header else 0
    print(f"\n✓ Consolidated: {total_rows} rows x {n_cols} columns")
    print(f"✓ Saved to: {out}")


def consolidate_pkl_files(input_dir: str, output_path: str):
    input_path = Path(input_dir)
    out = Path(output_path)

    pkls = sorted(p for p in input_path.glob("*.pkl") if p.name != out.name)
    if not pkls:
        raise RuntimeError(f"No .pkl files found in {input_path}")

    print(f"Found {len(pkls)} trace file(s):")
    dfs = []
    for pkl in pkls:
        df = pd.read_pickle(pkl)
        dfs.append(df)
        print(f"  {pkl.name}: {df.shape[0]} rows x {df.shape[1]} features")

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_pickle(out)


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate per-trace training .csv files into one matrix"
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, default="training",
        help="Directory containing per-trace .csv files (default: training)"
    )
    parser.add_argument(
        "-t", "--type", type=str, default="csv", choices=["csv", "pkl"],
        help="Type of files to consolidate (default: csv)"
    )
    args = parser.parse_args()
    if args.type == "csv":
        output_file = args.input_dir + "/all_benchmarks.csv"
        consolidate_csv_files(args.input_dir, output_file)
    elif args.type == "pkl":
        output_file = args.input_dir + "/all_benchmarks.pkl"
        consolidate_pkl_files(args.input_dir, output_file)
    else:
        parser.error(f"Invalid type: {args.type}")

    print(f"Consolidated {args.type} files saved to {output_file}")

if __name__ == "__main__":
    main()
