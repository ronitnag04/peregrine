import os
import models
import numpy as np

# Percentile levels used for all CDF feature vectors (avoid 0/100 extremes).
PERCENTILE_POINTS = np.linspace(1, 99, 50)

# Dimension of each per-resource feature vector: raw CDF + weighted CDF + mean.
FEATURES_PER_RESOURCE = 2 * len(PERCENTILE_POINTS) + 1  # = 101


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

    return np.percentile(samples, PERCENTILE_POINTS[:num_points])


def compute_cdf_features(samples: np.ndarray, num_points: int = 50):
    samples = np.asarray(samples).reshape(-1)
    cdf_raw = empirical_cdf_percentiles(samples, num_points=num_points)

    # Size-weighted CDF: weight each sample by its throughput value (max 0).
    # Computed exactly via sorted cumulative weights + interpolation.
    w = np.clip(samples, 0, None)

    if w.sum() == 0:
        # All samples <= 0 — fall back to the raw distribution.
        cdf_weighted = cdf_raw.copy()
    else:
        w = w / w.sum()
        sorted_idx = np.argsort(samples)
        cum_weights = np.cumsum(w[sorted_idx])
        ps = PERCENTILE_POINTS[:num_points] / 100
        cdf_weighted = np.interp(ps, cum_weights, samples[sorted_idx])

    return cdf_raw, cdf_weighted, float(np.mean(samples))
