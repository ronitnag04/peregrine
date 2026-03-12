import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for script usage
import matplotlib.pyplot as plt
import numpy as np


def _read_program_features_json(path: Path) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_feature_mapping(obj: object) -> Dict[str, float]:
    """
    Convert a JSON object into a flat mapping of feature_name -> float value.

    Accepted forms:
    - dict[str, number]: used directly
    - dict with a top-level "features" dict: {"features": {...}}
    - list[number]: converted to {"f0": v0, "f1": v1, ...}
    """
    if isinstance(obj, dict):
        if "features" in obj and isinstance(obj["features"], dict):
            obj = obj["features"]
        if all(isinstance(k, str) for k in obj.keys()):
            out: Dict[str, float] = {}
            for k, v in obj.items():
                if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(
                    float(v)
                ):
                    out[k] = float(v)
                else:
                    raise ValueError(f"Feature {k!r} is not a finite number: {v!r}")
            if not out:
                raise ValueError("No numeric features found in JSON object")
            return out

    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(
                float(v)
            ):
                out[f"f{i}"] = float(v)
            else:
                raise ValueError(f"Feature index {i} is not a finite number: {v!r}")
        if not out:
            raise ValueError("Feature list is empty")
        return out

    raise ValueError(
        "Unsupported JSON format for program features. "
        "Expected dict[str, number], {'features': dict[str, number]}, or list[number]."
    )


def _collect_feature_vectors(traces_dir: Path) -> Tuple[List[str], np.ndarray]:
    """
    Load per-benchmark program feature vectors from:

      <traces_dir>/<benchmark>/ronamol/program_features.json

    and return them stacked as a matrix.

    Returns
    -------
    benchmark_names : list[str]
        Benchmark directory names in the same order as the rows in `features`.
    features : np.ndarray
        Array of shape (n_benchmarks, n_features).
    """
    if not traces_dir.is_dir():
        raise NotADirectoryError(f"{traces_dir} is not a directory")

    benchmark_dirs = sorted([p for p in traces_dir.iterdir() if p.is_dir()])
    if not benchmark_dirs:
        raise RuntimeError(f"No benchmark subdirectories found in {traces_dir}")

    per_benchmark: List[Tuple[str, Dict[str, float]]] = []
    for bench_dir in benchmark_dirs:
        features_path = bench_dir / "ronamol" / "program_features.json"
        if not features_path.is_file():
            continue
        raw = _read_program_features_json(features_path)
        fmap = _coerce_feature_mapping(raw)
        per_benchmark.append((bench_dir.name, fmap))

    if not per_benchmark:
        raise RuntimeError(
            f"Found 0 benchmarks containing ronamol/program_features.json under {traces_dir}"
        )

    # Use the intersection of keys across all benchmarks, sorted for a stable order.
    key_sets = [set(fmap.keys()) for _, fmap in per_benchmark]
    common_keys = sorted(set.intersection(*key_sets))
    if not common_keys:
        raise ValueError(
            "No common feature keys across benchmarks. "
            "Ensure all program_features.json files share the same feature names."
        )

    benchmark_names: List[str] = []
    rows: List[np.ndarray] = []
    for name, fmap in per_benchmark:
        benchmark_names.append(name)
        rows.append(np.array([fmap[k] for k in common_keys], dtype=np.float64))

    features = np.stack(rows, axis=0)

    # L2-normalize each feature vector (row-wise) so that ||v_i||_2 = 1.
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Encountered a zero-norm feature vector during normalization")
    features_normalized = features / norms

    return benchmark_names, features_normalized


def compute_pairwise_distance_matrix(traces_dir: str) -> np.ndarray:
    """
    Compute the pairwise Euclidean distance matrix of analytical feature vectors.

    Each per-benchmark feature vector is first L2-normalized (unit length).

    Parameters
    ----------
    traces_dir : str
        Directory containing benchmark subdirectories. Each benchmark must contain:
        ronamol/program_features.json

    Returns
    -------
    distances : np.ndarray
        2D array of shape (N, N) with distances between benchmarks found under
        `traces_dir`.
    """
    benchmark_names, features = _collect_feature_vectors(Path(traces_dir))

    n_benchmarks = features.shape[0]
    if n_benchmarks < 2:
        raise ValueError(
            f"Need at least 2 benchmarks for a distance matrix, found {n_benchmarks} "
            f"in {traces_dir}: {benchmark_names}"
        )

    # features has shape (N, d). Use broadcasting to get all pairwise distances.
    diffs = features[:, None, :] - features[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    return distances


def _parse_size(size_str: str) -> float:
    """
    Parse cache size strings like '32KiB', '4MiB' into a numeric value (bytes).
    """
    size_str = size_str.strip()
    if size_str.endswith("KiB"):
        return float(size_str[:-3]) * 1024.0
    if size_str.endswith("MiB"):
        return float(size_str[:-3]) * 1024.0 * 1024.0
    # Fallback: plain number
    return float(size_str)


def _linear_regression_r2(X: np.ndarray, y: np.ndarray) -> float:
    """
    Fit a linear regression y ~ X (with intercept) and return R^2.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if X.shape[0] < X.shape[1] + 1:
        raise ValueError(
            f"Not enough samples ({X.shape[0]}) for linear regression with {X.shape[1]} features"
        )

    # Add intercept term.
    X_design = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ beta
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _compute_regression_r2(
    benchmark_names: List[str],
    features: np.ndarray,
    sweep_results_path: Path,
) -> Tuple[float, float, int, float, float, float]:
    """
    Compute R^2 for:
      1) CPI ~ analytical feature vectors (from program_features.json)
      2) CPI ~ microarchitectural configuration values (from sweep_results.csv)

    Only rows whose benchmark appears in `benchmark_names` are used.

    Also compute a simple variance decomposition of CPI over the included sweep rows:
    - total variance (SS_tot)
    - between-benchmark variance (SS_between), using per-benchmark mean CPI
    - within-benchmark variance (SS_within = SS_tot - SS_between)

    The between-benchmark fraction SS_between / SS_tot is an upper bound on how much
    CPI variation can be explained by any benchmark-constant signal (including the
    analytical feature vectors used here).
    """
    bench_to_idx = {name: i for i, name in enumerate(benchmark_names)}

    with open(sweep_results_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        try:
            cpi_idx = header.index("cpi")
            bench_idx = header.index("benchmark")
        except ValueError as e:
            raise ValueError(
                f"Expected 'cpi' and 'benchmark' columns in {sweep_results_path}"
            ) from e

        # Identify configuration columns (everything except cpi and benchmark).
        cfg_indices = [
            i for i in range(len(header)) if i not in (cpi_idx, bench_idx)
        ]

        y_vals: List[float] = []
        bench_labels: List[str] = []
        X_feat_rows: List[np.ndarray] = []
        X_cfg_rows: List[List[float]] = []

        for row in reader:
            bench = row[bench_idx]
            if bench not in bench_to_idx:
                continue

            try:
                cpi_val = float(row[cpi_idx])
            except ValueError:
                continue

            # Analytical features (already L2-normalized).
            feat_vec = features[bench_to_idx[bench]]

            # Config values.
            cfg_vals: List[float] = []
            for i in cfg_indices:
                col = header[i]
                val = row[i]
                if col == "branch_predictor":
                    # Encode as binary: local=0.0, tage=1.0
                    cfg_vals.append(1.0 if val == "tage" else 0.0)
                elif col in {"l1d_size", "l1i_size", "l2_size"}:
                    cfg_vals.append(_parse_size(val))
                else:
                    try:
                        cfg_vals.append(float(val))
                    except ValueError:
                        # Skip rows with non-numeric config values.
                        cfg_vals = []
                        break

            if not cfg_vals:
                continue

            y_vals.append(cpi_val)
            bench_labels.append(bench)
            X_feat_rows.append(feat_vec)
            X_cfg_rows.append(cfg_vals)

    if not y_vals:
        raise RuntimeError(
            f"No sweep_results rows matched benchmarks {benchmark_names} "
            f"in {sweep_results_path}"
        )

    y = np.array(y_vals, dtype=np.float64)
    X_feat = np.vstack(X_feat_rows).astype(np.float64)
    X_cfg = np.vstack(X_cfg_rows).astype(np.float64)

    r2_analytical = _linear_regression_r2(X_feat, y)
    r2_config = _linear_regression_r2(X_cfg, y)

    # Variance decomposition across sweep rows.
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    ss_between = 0.0
    if ss_tot > 0.0:
        # Compute per-benchmark mean CPI, weighted by rows per benchmark.
        by_bench: Dict[str, List[float]] = {}
        for b, v in zip(bench_labels, y_vals):
            by_bench.setdefault(b, []).append(v)
        for vals in by_bench.values():
            n_b = float(len(vals))
            mu_b = float(np.mean(vals))
            ss_between += n_b * (mu_b - y_mean) ** 2
    ss_within = ss_tot - ss_between

    return r2_analytical, r2_config, y.shape[0], ss_tot, ss_between, ss_within


def _compute_param_response_curves(
    benchmark_names: List[str],
    sweep_results_path: Path,
) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """
    For each configuration parameter, and for each benchmark, construct a CPI
    response curve:

        param_response[param][benchmark] = [(param_value_0, cpi_0), ...]

    where the points come from OFAT-style sweeps: CPI vs that single parameter,
    with all other parameters held constant.
    """
    bench_set = set(benchmark_names)

    with open(sweep_results_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        try:
            cpi_idx = header.index("cpi")
            bench_idx = header.index("benchmark")
        except ValueError as e:
            raise ValueError(
                f"Expected 'cpi' and 'benchmark' columns in {sweep_results_path}"
            ) from e

        cfg_indices = [
            i for i in range(len(header)) if i not in (cpi_idx, bench_idx)
        ]

        # Keep only rows from benchmarks of interest.
        rows = [row for row in reader if row[bench_idx] in bench_set]

    # param_name -> benchmark_name -> list[(param_value, cpi)]
    param_response: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

    for p_idx in cfg_indices:
        p_name = header[p_idx]
        # (benchmark, other_cfg_tuple) -> list[(param_value, cpi)]
        groups: Dict[Tuple[str, Tuple[str, ...]], List[Tuple[float, float]]] = {}

        other_indices = [i for i in cfg_indices if i != p_idx]

        for row in rows:
            bench = row[bench_idx]

            try:
                cpi_val = float(row[cpi_idx])
            except ValueError:
                continue

            # Build grouping key for "all other params held equal".
            other_cfg = tuple(row[i] for i in other_indices)

            # Numeric value for this parameter (used for sorting / alignment).
            val_raw = row[p_idx]
            col = p_name
            try:
                if col == "branch_predictor":
                    # Encode as binary: local=0.0, tage=1.0
                    p_val = 1.0 if val_raw == "tage" else 0.0
                elif col in {"l1d_size", "l1i_size", "l2_size"}:
                    p_val = _parse_size(val_raw)
                else:
                    p_val = float(val_raw)
            except ValueError:
                continue

            key = (bench, other_cfg)
            groups.setdefault(key, []).append((p_val, cpi_val))

        # For each benchmark, pick the largest sweep (most distinct param values).
        bench_curves: Dict[str, List[Tuple[float, float]]] = {}
        for (bench, _other_cfg), pts in groups.items():
            if len(pts) < 2:
                continue
            if bench not in bench_curves or len(pts) > len(bench_curves[bench]):
                bench_curves[bench] = pts

        if not bench_curves:
            continue

        # Sort each curve by parameter value.
        for bench, pts in bench_curves.items():
            bench_curves[bench] = sorted(pts, key=lambda x: x[0])

        param_response[p_name] = bench_curves

    return param_response


def _compute_param_response_distances(
    benchmark_names: List[str],
    param_response: Dict[str, Dict[str, List[Tuple[float, float]]]],
) -> Dict[str, np.ndarray]:
    """
    For each parameter, compute an N x N matrix where:

        response_dist[i, j, param] = || cpi_curve[i, param] - cpi_curve[j, param] ||_2

    Curves are aligned on the intersection of parameter values. If a pair of
    benchmarks has fewer than 1 common parameter value in its curves, the
    distance entry is left as NaN.
    """
    n = len(benchmark_names)
    name_to_idx = {name: i for i, name in enumerate(benchmark_names)}

    param_dists: Dict[str, np.ndarray] = {}

    for param, curves in param_response.items():
        dist = np.full((n, n), np.nan, dtype=np.float64)
        # Distance from a benchmark to itself is always 0.
        for i in range(n):
            dist[i, i] = 0.0

        for i_name in benchmark_names:
            if i_name not in curves:
                continue
            curve_i = curves[i_name]
            dict_i = {v: c for v, c in curve_i}
            for j_name in benchmark_names:
                if j_name not in curves:
                    continue
                if i_name == j_name:
                    continue
                curve_j = curves[j_name]
                dict_j = {v: c for v, c in curve_j}

                common_vals = sorted(set(dict_i.keys()) & set(dict_j.keys()))
                if not common_vals:
                    continue

                diffs = [
                    dict_i[val] - dict_j[val]
                    for val in common_vals
                ]
                d = float(np.linalg.norm(diffs))
                i = name_to_idx[i_name]
                j = name_to_idx[j_name]
                dist[i, j] = d

        param_dists[param] = dist

    return param_dists


def _plot_matrix(
    matrix: np.ndarray,
    benchmark_names: List[str],
    title: str,
    output_path: Path,
    cmap: str = "viridis",
) -> None:
    """
    Plot a square distance matrix with benchmark labels on both axes.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Benchmark")

    ticks = np.arange(len(benchmark_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(benchmark_names, rotation=45, ha="right")
    ax.set_yticklabels(benchmark_names)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Distance")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_regression_r2(
    r2_analytical: float,
    r2_config: float,
    output_path: Path,
) -> None:
    """
    Plot a simple bar chart comparing R^2 values.
    """
    labels = ["Analytical features", "Config values"]
    values = [r2_analytical, r2_config]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=["tab:blue", "tab:orange"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("R²")
    ax.set_title("CPI prediction R² comparison")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:.3f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Analyze RonAMoL per-benchmark features and (optionally) sweep results.\n"
            "\n"
            "Always computed:\n"
            "  - Pairwise Euclidean distance matrix between benchmarks using the\n"
            "    L2-normalized analytical feature vectors in:\n"
            "      <benchmark>/ronamol/program_features.json\n"
            "    Outputs:\n"
            "      * feature_distance_matrix.csv\n"
            "      * feature_distance_matrix.png\n"
            "      * benchmarks.txt (row/col order for matrices)\n"
            "\n"
            "When --ofat-sweep-results is provided (sweep_results.csv):\n"
            "  - Linear regression R² (does NOT require OFAT):\n"
            "      * CPI ~ analytical features\n"
            "      * CPI ~ microarchitectural configuration values\n"
            "    Outputs:\n"
            "      * regression_summary.txt\n"
            "      * regression_r2_comparison.png\n"
            "  - Per-parameter CPI response-distance matrices (requires OFAT-style rows):\n"
            "      * Build a CPI-vs-parameter curve per benchmark from rows where all\n"
            "        other parameters are held constant\n"
            "      * Align curves on shared parameter values\n"
            "      * Compute pairwise L2 distances between curves\n"
            "    Outputs (for each swept parameter <param>):\n"
            "      * response_distance_<param>.csv\n"
            "      * response_distance_<param>.png\n"
            "\n"
            "All outputs are written under --analysis-output-dir."
        )
    )
    parser.add_argument(
        "-i",
        "--traces-dir",
        type=str,
        required=True,
        help=(
            "Directory containing benchmark subdirectories. Each benchmark directory must "
            "contain ronamol/program_features.json. The analysis runs on all benchmarks "
            "found (N >= 2), unless excluded via --drop-benchmark."
        ),
    )
    parser.add_argument(
        "-s",
        "--ofat-sweep-results",
        "--sweep-results",
        type=str,
        default=None,
        dest="ofat_sweep_results",
        help=(
            "Optional path to sweep_results.csv.\n"
            "\n"
            "Enables:\n"
            "  - Linear regression R²:\n"
            "      * CPI ~ analytical features (from program_features.json)\n"
            "      * CPI ~ configuration values (CSV columns other than benchmark/cpi)\n"
            "  - Per-parameter CPI response-distance matrices (CSV + heatmap PNG)\n"
            "      (requires OFAT-style rows: one varying parameter, others held constant)\n"
            "\n"
            "Preferred flag name: --ofat-sweep-results\n"
            "Compatibility alias: --sweep-results"
        ),
    )
    parser.add_argument(
        "-A",
        "--analysis-output-dir",
        type=str,
        default="analysis_outputs",
        help=(
            "Output directory for generated artifacts.\n"
            "\n"
            "Always written:\n"
            "  - benchmarks.txt (row/col order)\n"
            "  - feature_distance_matrix.csv\n"
            "  - feature_distance_matrix.png\n"
            "\n"
            "When --ofat-sweep-results is provided:\n"
            "  - regression_summary.txt\n"
            "  - regression_r2_comparison.png\n"
            "  - response_distance_<param>.csv\n"
            "  - response_distance_<param>.png\n"
            "\n"
            "Default: analysis_outputs"
        ),
    )
    parser.add_argument(
        "--drop-benchmark",
        action="append",
        default=[],
        help=(
            "Benchmark directory name to exclude from all analyses and outputs. "
            "May be specified multiple times."
        ),
    )

    args = parser.parse_args()
    benchmark_names, features = _collect_feature_vectors(Path(args.traces_dir))

    # Optionally drop selected benchmarks.
    if args.drop_benchmark:
        drop_set = set(args.drop_benchmark)
        unknown = sorted(drop_set - set(benchmark_names))
        if unknown:
            raise ValueError(
                f"--drop-benchmark specified unknown benchmark(s): {unknown}. "
                f"Available benchmarks: {benchmark_names}"
            )
        keep_mask = [name not in drop_set for name in benchmark_names]
        if not any(keep_mask):
            raise ValueError("All benchmarks were dropped; nothing left to analyze.")
        benchmark_names = [n for n, keep in zip(benchmark_names, keep_mask) if keep]
        features = features[keep_mask, :]

    if features.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 benchmarks for distance analysis, found {features.shape[0]}"
        )

    diffs = features[:, None, :] - features[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)

    analysis_dir = Path(args.analysis_output_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Save benchmark order.
    benchmarks_path = analysis_dir / "benchmarks.txt"
    with open(benchmarks_path, "w", encoding="utf-8") as f:
        for i, name in enumerate(benchmark_names):
            f.write(f"{i},{name}\n")

    # Save feature distance matrix.
    feature_dist_path = analysis_dir / "feature_distance_matrix.csv"
    np.savetxt(feature_dist_path, distances, delimiter=",", fmt="%.10g")

    # Plot feature distance matrix.
    feature_dist_plot = analysis_dir / "feature_distance_matrix.png"
    _plot_matrix(
        distances,
        benchmark_names,
        title="Feature distance matrix (analytical vectors)",
        output_path=feature_dist_plot,
    )

    print("Completed: analytical feature distance matrix")
    print(f"  Saved benchmark order to:        {benchmarks_path}")
    print(f"  Saved distance matrix CSV to:    {feature_dist_path}")
    print(f"  Saved distance matrix heatmap to:{feature_dist_plot}")

    if args.ofat_sweep_results:
        sweep_path = Path(args.ofat_sweep_results)
        (
            r2_analytical,
            r2_config,
            n_samples,
            ss_tot,
            ss_between,
            ss_within,
        ) = _compute_regression_r2(benchmark_names, features, sweep_path)

        # Save regression summary.
        reg_summary_path = analysis_dir / "regression_summary.txt"
        between_frac = (ss_between / ss_tot) if ss_tot > 0.0 else float("nan")
        within_frac = (ss_within / ss_tot) if ss_tot > 0.0 else float("nan")
        with open(reg_summary_path, "w", encoding="utf-8") as f:
            f.write(
                f"Samples used: {n_samples}\n"
                f"Benchmarks: {', '.join(benchmark_names)}\n"
                f"R^2 (analytical features vs CPI): {r2_analytical:.6f}\n"
                f"R^2 (config values vs CPI):      {r2_config:.6f}\n"
                "\n"
                "CPI variance decomposition over included sweep rows:\n"
                f"SS_tot:     {ss_tot:.6g}\n"
                f"SS_between: {ss_between:.6g}  (between-benchmark)\n"
                f"SS_within:  {ss_within:.6g}  (within-benchmark / config-to-config)\n"
                f"Frac_between (SS_between/SS_tot): {between_frac:.6f}\n"
                f"Frac_within  (SS_within/SS_tot):  {within_frac:.6f}\n"
            )

        # Plot regression R^2 comparison.
        reg_plot_path = analysis_dir / "regression_r2_comparison.png"
        _plot_regression_r2(r2_analytical, r2_config, reg_plot_path)
        print("\nCompleted: CPI regression R² analyses")
        print(f"  Saved regression summary to:     {reg_summary_path}")
        print(f"  Saved R² comparison plot to:     {reg_plot_path}")

        # Per-parameter CPI response-distance matrices.
        param_response = _compute_param_response_curves(benchmark_names, sweep_path)
        param_dists = _compute_param_response_distances(benchmark_names, param_response)

        def _sanitize_param_name(name: str) -> str:
            cleaned = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
            return cleaned or "param"

        if param_dists:
            print("\nCompleted: per-parameter CPI response-distance matrices")
            print(f"  Benchmark order for matrices:   {benchmarks_path}")

        for param, dist in sorted(param_dists.items()):
            # Save per-parameter response distance matrix.
            sanitized = _sanitize_param_name(param)
            param_path = analysis_dir / f"response_distance_{sanitized}.csv"
            np.savetxt(param_path, dist, delimiter=",", fmt="%.10g")

            # Plot per-parameter response distance matrix.
            param_plot = analysis_dir / f"response_distance_{sanitized}.png"
            _plot_matrix(
                dist,
                benchmark_names,
                title=f"CPI response distance matrix ({param})",
                output_path=param_plot,
            )
            print(f"  {param}")
            print(f"    Saved CSV to:                 {param_path}")
            print(f"    Saved heatmap to:             {param_plot}")


if __name__ == "__main__":
    main()

