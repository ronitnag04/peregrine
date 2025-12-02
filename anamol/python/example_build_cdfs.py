import os
import glob
import numpy as np
import pandas as pd  # NEW


# Same list as in example_read.py
RESOURCE_FILES = [
    "thr_rob.npy",
    "thr_load_queue.npy",
    "thr_store_queue.npy",
    "thr_alu_issue.npy",
    "thr_fp_issue.npy",
    "thr_ls_issue.npy",
    "thr_load_ls_pipes_lower.npy",
    "thr_load_ls_pipes_upper.npy",
    "thr_icache_fills.npy",
    "thr_fetch_buffers.npy",
]

# Resources with two parameters (p0, p1); others have one parameter (p0 only)
DOUBLE_PARAM_RESOURCES = {
    "load_ls_pipes_lower",
    "load_ls_pipes_upper",
}

# Parameter name specifications per resource.
# Adjust names to match your actual parameters.
RESOURCE_PARAM_SPECS = {
    "rob": ["rob_size"],
    "load_queue": ["ldq_entries"],
    "store_queue": ["stq_entries"],
    "alu_issue": ["alu_width"],
    "fp_issue": ["fp_width"],
    "ls_issue": ["ls_width"],
    "load_ls_pipes_lower": ["p0", "p1"],  # TODO: replace with real names
    "load_ls_pipes_upper": ["p0", "p1"],  # TODO: replace with real names
    "icache_fills": ["icache_miss_buf"],
    "fetch_buffers": ["fetch_buf_size"],
}


def load_resource_file(path, res_name):
    """
    Load a single resource .npy file.

    Layout (current C++ writer, matching example_read.py):
      single-param resources:
        [p0, thr_0, ..., thr_{N-1}]
      double-param resources:
        [p0, p1, thr_0, ..., thr_{N-1}]

    Returns:
        params: (num_combos, 1 or 2) array of parameter values (p0[, p1])
        thr:    (num_combos, num_windows) array of throughputs
    """
    arr = np.load(path)  # shape: (num_combos, param_cols + num_windows)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2D array, got shape {arr.shape}")

    num_combos, total_cols = arr.shape

    double_params = res_name in DOUBLE_PARAM_RESOURCES
    param_cols = 2 if double_params else 1
    num_windows = total_cols - param_cols
    if num_windows <= 0:
        raise ValueError(
            f"{path}: invalid layout, total_cols={total_cols}, "
            f"param_cols={param_cols}"
        )

    params = arr[:, :param_cols]
    thr = arr[:, param_cols:]
    return params, thr


def load_all_throughputs(output_dir="output"):
    """
    Load all known throughput npy files from 'output_dir'.

    Returns:
        dict: {resource_name: (params, thr)}
              where:
                params: (num_combos, 1 or 2)
                thr:    (num_combos, num_windows)
    """
    results = {}
    for fname in RESOURCE_FILES:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            continue
        res_name = os.path.splitext(fname)[0].replace("thr_", "")
        params, thr = load_resource_file(path, res_name)
        results[res_name] = (params, thr)
    return results


def empirical_cdf_percentiles(samples: np.ndarray, num_points: int = 50) -> np.ndarray:
    """
    Compute a percentile-based CDF encoding.

    Args:
        samples: 1D numpy array of values (e.g., all throughput values for a resource)
        num_points: number of percentile points (default 50 for 50 percentiles)

    Returns:
        1D numpy array of shape (num_points,) containing the value at each percentile.
        Percentiles are at 100 * i / (num_points - 1), i = 0..num_points-1
        (so for num_points=50: 0%, ~2.04%, ..., 100%).
    """
    if samples.size == 0:
        raise ValueError("Cannot compute CDF on empty sample set")

    ps = np.linspace(0, 100, num_points)
    return np.percentile(samples, ps)


def build_throughput_cdfs(
    input_dir="output",
    output_dir="output_cdf",
    verbose=True,
    save_parquet=True,
    save_csv=False,
):
    """
    Load all throughput arrays, build CDFs as described in the README,
    and save them as .npy vectors.

    Additionally, build a pandas DataFrame with one row per
    (resource, parameter combination), including explicit parameter names.

    For each resource r:
      - flatten thr into a 1D array over all param combos and windows
      - compute 50 fixed percentiles over the original distribution
      - compute 50 fixed percentiles over a size-weighted distribution
      - append the mean value
      - save to: output_cdf/cdf_<resource>.npy

    Final feature vector per resource has length 50 + 50 + 1 = 101.
    """
    os.makedirs(output_dir, exist_ok=True)
    res_data = load_all_throughputs(input_dir)

    if not res_data:
        if verbose:
            print(f"No throughput files found in {input_dir}")
        return

    # Rows for the per-parameter DataFrame
    df_rows = []

    for res_name, (params, thr) in res_data.items():
        # thr: (num_combos, num_windows)
        flat_thr = thr.reshape(-1)

        # 1) 50 percentiles over the original empirical distribution
        cdf_raw = empirical_cdf_percentiles(flat_thr, num_points=50)

        # 2) 50 percentiles over a "size-weighted" distribution.
        # In Concorde, size-weighted CDFs weight windows by dynamic size/work.
        # Here, we approximate that by weighting each sample by its throughput
        # itself (higher-throughput windows contribute proportionally more).
        #
        # Build a weighted sample by repeating each value proportional to its
        # relative weight, then take percentiles on that expanded set.
        # To keep it finite, we scale to a fixed total weight.
        weights = flat_thr.copy()
        # Ensure non-negative weights and avoid all-zero
        weights = np.clip(weights, a_min=0.0, a_max=None)
        if np.all(weights == 0):
            # Fallback: no informative weighting, reuse raw distribution
            cdf_weighted = cdf_raw.copy()
        else:
            weights = weights / weights.sum()
            target_count = 10_000  # total "samples" in expanded set
            counts = np.round(weights * target_count).astype(int)
            # Guarantee at least 1 sample for non-zero weights
            counts[counts == 0] = 1
            expanded = np.repeat(flat_thr, counts)
            cdf_weighted = empirical_cdf_percentiles(expanded, num_points=50)

        # 3) Mean value of the original distribution
        mean_val = float(np.mean(flat_thr))

        # 4) Concatenate: [50 raw, 50 weighted, mean]
        cdf_vec = np.concatenate([cdf_raw, cdf_weighted, np.array([mean_val])])

        out_path = os.path.join(output_dir, f"cdf_{res_name}.npy")
        np.save(out_path, cdf_vec)

        # ---- Per-parameter rows for DataFrame ----
        num_combos, num_windows = thr.shape
        # Parameter names for this resource
        param_names = RESOURCE_PARAM_SPECS.get(res_name)
        if param_names is None:
            # Fallback to generic names
            n_params = params.shape[1]
            param_names = [f"p{i}" for i in range(n_params)]

        # For each parameter combo, build its own CDF vector (over windows)
        for i in range(num_combos):
            # samples for this parameter combo across windows
            samples = thr[i, :]

            combo_cdf_raw = empirical_cdf_percentiles(samples, num_points=50)

            w = np.clip(samples.copy(), a_min=0.0, a_max=None)
            if np.all(w == 0):
                combo_cdf_weighted = combo_cdf_raw.copy()
            else:
                w = w / w.sum()
                target_count = 2_000  # smaller since per-combo
                c = np.round(w * target_count).astype(int)
                c[c == 0] = 1
                expanded_i = np.repeat(samples, c)
                combo_cdf_weighted = empirical_cdf_percentiles(
                    expanded_i, num_points=50
                )

            combo_mean = float(np.mean(samples))

            row = {"resource": res_name}
            # Attach parameters by name
            for j, pname in enumerate(param_names):
                if j < params.shape[1]:
                    row[pname] = float(params[i, j])
                else:
                    row[pname] = np.nan

            # Attach CDF features
            for k in range(50):
                row[f"cdf_raw_{k}"] = float(combo_cdf_raw[k])
                row[f"cdf_weighted_{k}"] = float(combo_cdf_weighted[k])
            row["mean"] = combo_mean

            df_rows.append(row)

        if verbose:
            num_combos, num_windows = thr.shape
            print("=" * 60)
            print(f"Resource: {res_name}")
            print(f"  Input shape:         combos={num_combos}, windows={num_windows}")
            print(f"  Total samples:       {flat_thr.size}")
            print(f"  CDF feature length:  {cdf_vec.size} (50 + 50 + 1)")
            print(f"  CDF saved to:        {out_path}")
            print(
                f"  raw[0], raw[25], raw[49]: "
                f"{cdf_raw[0]:.4f}, {cdf_raw[25]:.4f}, {cdf_raw[49]:.4f}"
            )
            print(
                f"  wgt[0], wgt[25], wgt[49]: "
                f"{cdf_weighted[0]:.4f}, {cdf_weighted[25]:.4f}, {cdf_weighted[49]:.4f}"
            )
            print(f"  mean: {mean_val:.4f}")

    # ---- Build and save the DataFrame ----
    if df_rows:
        df = pd.DataFrame(df_rows)
        df_path_parquet = os.path.join(output_dir, "all_cdfs.parquet")
        if save_parquet:
            df.to_parquet(df_path_parquet, index=False)
            if verbose:
                print(f"Per-parameter CDF DataFrame written to {df_path_parquet}")

        if save_csv:
            df_path_csv = os.path.join(output_dir, "all_cdfs.csv")
            df.to_csv(df_path_csv, index=False)
            if verbose:
                print(f"Per-parameter CDF DataFrame written to {df_path_csv}")

    if verbose:
        print("=" * 60)
        print(f"All CDFs written to {output_dir}")


def list_input_files(input_dir="output"):
    files = glob.glob(os.path.join(input_dir, "thr_*.npy"))
    if not files:
        print(f"No files matching thr_*.npy in {input_dir}")
    else:
        print("Found throughput files:")
        for f in sorted(files):
            print("  ", f)


def list_output_files(output_dir="output_cdf"):
    files = glob.glob(os.path.join(output_dir, "cdf_*.npy"))
    if not files:
        print(f"No files matching cdf_*.npy in {output_dir}")
    else:
        print("Found CDF files:")
        for f in sorted(files):
            print("  ", f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Read throughput .npy files (thr_*.npy) and build CDF encodings "
            "as described in the README (101-percentile vectors per resource)."
        )
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="output",
        help="Directory containing thr_*.npy (default: output)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output_cdf",
        help="Directory to write cdf_*.npy (default: output_cdf)",
    )
    parser.add_argument(
        "--list-input",
        action="store_true",
        help="List available thr_*.npy files and exit",
    )
    parser.add_argument(
        "--list-output",
        action="store_true",
        help="List available cdf_*.npy files and exit",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress verbose summary prints",
    )
    args = parser.parse_args()

    if args.list_input:
        list_input_files(args.input_dir)
    elif args.list_output:
        list_output_files(args.output_dir)
    else:
        build_throughput_cdfs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )
