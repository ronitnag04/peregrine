import os
import models
import numpy as np


def load_resource_file(path, res_name):
    """
    Load a single resource .npy file.

    Layout (current C++ writer):
      [num_params, p0, (p1)?, thr_0, ..., thr_{N-1}]

    Returns:
        params: (num_combos, 1 or 2) array of parameter values (p0[, p1])
        thr:    (num_combos, num_windows) array of throughputs
    """
    # normalize resource identifier to string key
    res_key = models.resource_key(res_name)

    arr = np.load(path)  # shape: (num_combos, param_cols + num_windows)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2D array, got shape {arr.shape}")

    num_combos, total_cols = arr.shape

    # First column is param count (1 or 2). Detect whether file contains a p1
    # column by checking any row's param count.
    param_count_col = arr[:, 0].astype(int)
    file_has_p1 = bool((param_count_col >= 2).any())

    # Determine how many param columns are present in the file (1 or 2 params plus the count col).
    param_cols_in_file = 3 if file_has_p1 else 2
    num_windows = total_cols - param_cols_in_file
    if num_windows <= 0:
        raise ValueError(
            f"{path}: invalid layout, total_cols={total_cols}, "
            f"param_cols_in_file={param_cols_in_file}"
        )

    # Extract params and throughputs. params start at column 1.
    params = arr[:, 1:param_cols_in_file]  # shape (num_combos, 1) or (num_combos, 2)
    thr = arr[:, param_cols_in_file:]

    # If the resource is expected to have two params but the file only has one
    # (possible for backward compatibility), pad a zero p1 column so callers
    # always get shape (N,2) for double-param resources.
    if res_key in models.DOUBLE_PARAM_RESOURCES and params.shape[1] == 1:
        pad = np.zeros((num_combos, 1), dtype=params.dtype)
        params = np.concatenate([params, pad], axis=1)

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
    for fname in models.RESOURCE_FILES:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            continue
        # derive string resource key from filename (keep backwards-compatible keys)
        res_name = os.path.splitext(fname)[0].replace("thr_", "")
        params, thr = load_resource_file(path, res_name)
        results[res_name] = (params, thr)
    return results


def empirical_cdf_percentiles(samples: np.ndarray, num_points: int = 50) -> np.ndarray:
    samples = np.asarray(samples).reshape(-1)
    if samples.size == 0:
        raise ValueError("Cannot compute CDF on empty sample set")

    # avoid extreme instability at 0% or 100%
    ps = np.linspace(1, 99, num_points)
    return np.percentile(samples, ps)


def compute_cdf_features(samples: np.ndarray, num_points: int = 50):
    samples = np.asarray(samples).reshape(-1)
    cdf_raw = empirical_cdf_percentiles(samples, num_points=num_points)

    # size-weight the distribution: weight = max(sample, 0)
    w = np.clip(samples, 0, None)

    if w.sum() == 0:
        # all samples <= 0 → fallback to raw distribution
        cdf_weighted = cdf_raw.copy()
    else:
        w = w / w.sum()
        target_count = 10_000

        counts = np.round(w * target_count).astype(int)

        # Do NOT force zero-weights to become 1
        mask = counts > 0
        expanded = np.repeat(samples[mask], counts[mask])

        cdf_weighted = empirical_cdf_percentiles(expanded, num_points=num_points)

    mean_val = float(np.mean(samples))
    return cdf_raw, cdf_weighted, mean_val
