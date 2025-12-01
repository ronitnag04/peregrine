# %%
import os
import glob
import numpy as np
import pandas as pd

# %%
# Configuration

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
RESOURCE_PARAM_SPECS = {
    "rob": ["rob_size"],
    "load_queue": ["ldq_entries"],
    "store_queue": ["stq_entries"],
    "alu_issue": ["alu_width"],
    "fp_issue": ["fp_width"],
    "ls_issue": ["ls_width"],
    "load_ls_pipes_lower": ["p0", "p1"],  # TODO: replace with real names if known
    "load_ls_pipes_upper": ["p0", "p1"],  # TODO: replace with real names if known
    "icache_fills": ["icache_miss_buf"],
    "fetch_buffers": ["fetch_buf_size"],
}

# %%
# I/O helpers: read throughput npy files


def load_resource_file(path, res_name):
    """
    Load a single resource .npy file.

    Layout (current C++ writer):
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


# %%
# CDF utilities


def empirical_cdf_percentiles(samples: np.ndarray, num_points: int = 50) -> np.ndarray:
    """
    Compute a percentile-based CDF encoding.

    Args:
        samples: 1D numpy array of values
        num_points: number of percentile points

    Returns:
        1D numpy array of shape (num_points,)
    """
    if samples.size == 0:
        raise ValueError("Cannot compute CDF on empty sample set")

    ps = np.linspace(0, 100, num_points)
    return np.percentile(samples, ps)


def compute_cdf_features(samples: np.ndarray, num_points: int = 50):
    """
    Compute [raw CDF (num_points), weighted CDF (num_points), mean] for samples.
    Weighting is proportional to sample value, clipped at zero.
    """
    samples = np.asarray(samples).reshape(-1)
    cdf_raw = empirical_cdf_percentiles(samples, num_points=num_points)

    # size-weighted CDF (approximation: weight by throughput magnitude)
    w = np.clip(samples.copy(), a_min=0.0, a_max=None)
    if np.all(w == 0):
        cdf_weighted = cdf_raw.copy()
    else:
        w = w / w.sum()
        target_count = 10_000
        counts = np.round(w * target_count).astype(int)
        counts[counts == 0] = 1
        expanded = np.repeat(samples, counts)
        cdf_weighted = empirical_cdf_percentiles(expanded, num_points=num_points)

    mean_val = float(np.mean(samples))
    return cdf_raw, cdf_weighted, mean_val


# %%
# Build per-resource CDF vectors and per-parameter DataFrames (one per resource)


def build_cdfs_and_dataframes_per_resource(
    input_dir="output",
    num_points: int = 50,
    verbose: bool = True,
):
    """
    Load thr_*.npy files, build CDF encodings, and assemble
    one DataFrame per resource.

    Returns:
        dict: {resource_name: DataFrame}
    """
    res_data = load_all_throughputs(input_dir)

    if not res_data:
        if verbose:
            print(f"No throughput files found in {input_dir}")
        return {}

    dfs = {}

    for res_name, (params, thr) in res_data.items():
        num_combos, num_windows = thr.shape
        flat_thr = thr.reshape(-1)

        if verbose:
            print("=" * 60)
            print(f"Resource: {res_name}")
            print(f"  Input shape:         combos={num_combos}, windows={num_windows}")
            print(f"  Total samples:       {flat_thr.size}")

        param_names = RESOURCE_PARAM_SPECS.get(res_name)
        if param_names is None:
            n_params = params.shape[1]
            param_names = [f"p{i}" for i in range(n_params)]

        df_rows = []

        for i in range(num_combos):
            samples = thr[i, :]

            combo_raw, combo_weighted, combo_mean = compute_cdf_features(
                samples, num_points=num_points
            )

            row = {}
            # Attach parameters by name
            for j, pname in enumerate(param_names):
                if j < params.shape[1]:
                    row[pname] = float(params[i, j])
                else:
                    row[pname] = np.nan

            # Attach CDF features: all raw first, then all weighted
            for k in range(num_points):
                row[f"cdf_raw_{k}"] = float(combo_raw[k])
            for k in range(num_points):
                row[f"cdf_weighted_{k}"] = float(combo_weighted[k])

            row["mean"] = combo_mean

            df_rows.append(row)

        dfs[res_name] = pd.DataFrame(df_rows)

    if verbose:
        print("=" * 60)
        print(f"Built DataFrames for resources: {list(dfs.keys())}")

    return dfs


# %%
# Interactive usage: read NPY files, build CDFs, and store in 10 DataFrames

# Directory that contains the thr_*.npy files (relative to this script)
input_dir = "../output"

# Number of percentile points to use in the CDF representation
num_points = 50

# Build one DataFrame per resource
dfs = build_cdfs_and_dataframes_per_resource(
    input_dir=input_dir,
    num_points=num_points,
    verbose=True,
)

# Unpack into separate variables if desired
df_rob = dfs.get("rob")
df_load_queue = dfs.get("load_queue")
df_store_queue = dfs.get("store_queue")
df_alu_issue = dfs.get("alu_issue")
df_fp_issue = dfs.get("fp_issue")
df_ls_issue = dfs.get("ls_issue")
df_load_ls_lower = dfs.get("load_ls_pipes_lower")
df_load_ls_upper = dfs.get("load_ls_pipes_upper")
df_icache_fills = dfs.get("icache_fills")
df_fetch_buffers = dfs.get("fetch_buffers")

# Example inspection
df_rob.head() if df_rob is not None else None
# %%
