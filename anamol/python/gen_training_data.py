"""
Generate training data for microarchitecture performance prediction.

This script combines multiple feature types into a unified training dataset:
  1. Per-resource throughput features (from lookup table)
  2. Pipeline stall features (from trace analysis)
  3. Config scalar features (including misprediction_percent from config)
  4. (Future: Additional feature types can be added here)

Feature encoding (101 dimensions per distribution):
  - 50 percentiles of raw distribution (p1, p3, ..., p99)
  - 50 percentiles of size-weighted distribution
  - 1 mean value

Command-line usage:
  # Generate training data for a specific config and trace (pickle format)
  python gen_training_data.py \\
      --lookup throughput_lookup.pkl \\
      --config-json '{"rob_size": 256, "load_queue_size": 128}' \\
      --trace traces/my_trace.csv \\
      --window-size 400 \\
      -o training_data.pkl

  # Generate training data for multiple random configs (CSV format)
  python gen_training_data.py \\
      --lookup throughput_lookup.pkl \\
      --random-configs 1000 \\
      --trace traces/my_trace.csv \\
      --window-size 400 \\
      -o training_data.csv

Programmatic usage:
  # Single configuration
  from gen_training_data import generate_training_sample
  from build_throughput_lookup import ThroughputLookupTable
  import models

  lookup = ThroughputLookupTable.load("throughput_lookup.pkl")
  config = models.Config(rob_size=256, load_queue_size=128)
  training_data = generate_training_sample(
      config, lookup, "traces/my_trace.csv",
      window_size=400
  )

  # Multiple configurations
  from gen_training_data import generate_training_matrix

  configs = [models.Config(rob_size=128), models.Config(rob_size=256)]
  training_data = generate_training_matrix(
      configs, lookup, "traces/my_trace.csv",
      window_size=400
  )

  # Pipeline stalls only
  from gen_training_data import compute_pipeline_stall_features

  stall_features = compute_pipeline_stall_features(
      "traces/my_trace.csv", window_size=400
  )

  # Convenience wrapper (loads lookup table automatically)
  from gen_training_data import build_training_data

  training_data = build_training_data(
      lookup_path="throughput_lookup.pkl",
      trace_path="traces/my_trace.csv",
      configs=[models.Config(rob_size=256)],
      window_size=400,
  )
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

# Local imports
import registry
import utils
import models
from build_throughput_lookup import ThroughputLookupTable


# Public API
__all__ = [
    "compute_pipeline_stall_features",
    "compute_rob_latency_features",
    "generate_training_sample",
    "generate_training_matrix",
    "build_training_data",
    "sample_random_config",
]


def sample_random_config() -> models.Config:
    """
    Sample a random microarchitecture configuration following Concorde's approach.

    Independently samples each enabled parameter from its range in registry.yaml.
    This creates a massive microarchitecture space (~2x10^23 as noted in Concorde paper).

    Returns:
        Random Config with all parameters independently sampled
    """
    config_dict = {}
    for p in registry.ENABLED_PARAMS:
        if isinstance(p.step, list):
            config_dict[p.name] = int(np.random.choice(p.step))
        else:
            config_dict[p.name] = int(np.random.randint(p.min_val, p.max_val + 1))
    return models.Config(**config_dict)


def get_config_scalar_features(config: models.Config) -> Dict[str, Any]:
    """
    Extract scalar and one-hot encoded features from a Config object.

    Returns dictionary with 25 columns:
    - 21 scalar microarchitecture parameters
    - 2 one-hot encoded branch predictor columns (bp_is_simple, bp_is_tage)
    - 2 one-hot encoded prefetcher columns (prefetcher_off, prefetcher_on)
    """
    # Scalar columns: one per enabled param, name matches Config field name.
    features = {
        p.name: getattr(config, p.name)
        for p in registry.ENABLED_PARAMS
    }
    # One-hot encodings for categorical params
    features["bp_is_simple"] = 1 if config.branch_predictor == 0 else 0
    features["bp_is_tage"] = 1 if config.branch_predictor == 1 else 0
    features["prefetcher_off"] = 1 if config.l1d_stride_prefetch == 0 else 0
    features["prefetcher_on"] = 1 if config.l1d_stride_prefetch != 0 else 0
    return features


def _add_cdf_features_to_dict(
    feature_dict: dict, prefix: str, latencies: np.ndarray, num_points: int = 50
) -> None:
    """
    Helper to compute CDF features and add them to feature_dict with given prefix.

    Adds 101 features: 50 raw percentiles + 50 weighted percentiles + 1 mean.
    """
    cdf_raw, cdf_weighted, mean_val = utils.compute_cdf_features(
        latencies, num_points=num_points
    )

    for j, p in enumerate(utils.PERCENTILE_POINTS[:num_points]):
        feature_dict[f"{prefix}_raw_p{int(p)}"] = cdf_raw[j]
    for j, p in enumerate(utils.PERCENTILE_POINTS[:num_points]):
        feature_dict[f"{prefix}_weighted_p{int(p)}"] = cdf_weighted[j]
    feature_dict[f"{prefix}_mean"] = mean_val


def compute_rob_latency_features(output_dir: str = "output") -> pd.DataFrame:
    """
    Compute ROB latency features from pre-computed .npy files.

    This function reads the latency analysis results generated by the C++ code and
    computes CDF features (101 dimensions each) for:
    - Overall ROB throughput (11 values, one per ROB size)
    - Issue Latency (11 x 101 = 1111 features)
    - Commit Latency (11 x 101 = 1111 features)
    - Exec Latency (101 features, averaged across ROB sizes)

    Total: 11 + 1111 + 1111 + 101 = 2334 features

    Args:
        output_dir: Directory containing the latency .npy files

    Returns:
        DataFrame with 1 row and 2334 columns

    Example:
        >>> from gen_training_data import compute_rob_latency_features
        >>> latency_features = compute_rob_latency_features("output")
        >>> print(latency_features.shape)  # (1, 2334)
    """
    from pathlib import Path

    output_path = Path(output_dir)

    # Load overall throughput: shape (11, 2) [rob_size, throughput]
    thr_path = output_path / "rob_latency_overall_thr.npy"
    if not thr_path.exists():
        raise FileNotFoundError(f"Latency file not found: {thr_path}")

    overall_thr = np.load(thr_path)  # shape (11, 2)
    rob_sizes = overall_thr[:, 0].astype(int)
    throughputs = overall_thr[:, 1]

    # Load latency distributions: shape (11, k) where k = num_instructions
    issue_latencies = np.load(output_path / "rob_latency_issue.npy")
    commit_latencies = np.load(output_path / "rob_latency_commit.npy")
    exec_latencies = np.load(output_path / "rob_latency_exec.npy")

    print(f"\nComputing ROB latency features from: {output_dir}")
    print(f"  ROB sizes: {rob_sizes.tolist()}")
    print(f"  Number of instructions: {issue_latencies.shape[1]:,}")

    feature_dict = {}

    # 1. Overall throughput for each ROB size (11 values)
    for i, rob_size in enumerate(rob_sizes):
        feature_dict[f"rob{rob_size}_overall_thr"] = throughputs[i]

    # 2. Issue Latency CDF features for each ROB size (11 x 101 = 1111 features)
    for i, rob_size in enumerate(rob_sizes):
        _add_cdf_features_to_dict(
            feature_dict, f"rob{rob_size}_issue", issue_latencies[i]
        )

    # 3. Commit Latency CDF features for each ROB size (11 x 101 = 1111 features)
    for i, rob_size in enumerate(rob_sizes):
        _add_cdf_features_to_dict(
            feature_dict, f"rob{rob_size}_commit", commit_latencies[i]
        )

    # 4. Exec Latency CDF features (average across all ROB sizes)
    #    Since exec latency should be similar across ROB sizes, we average them
    exec_latencies_avg = exec_latencies.mean(axis=0)
    _add_cdf_features_to_dict(feature_dict, "exec", exec_latencies_avg)

    # Convert to DataFrame
    features_df = pd.DataFrame([feature_dict])

    total_features = len(feature_dict)
    expected_features = 11 + 11 * 101 + 11 * 101 + 101  # 2334
    print(
        f"  ROB latency features: {total_features} columns "
        f"(expected {expected_features})"
    )

    if total_features != expected_features:
        print(
            f"  WARNING: Feature count mismatch! "
            f"Got {total_features}, expected {expected_features}"
        )

    return features_df


def build_training_data(
    lookup_path: str,
    trace_path: str,
    configs: list[models.Config],
    window_size: int = 400,
    include_latency_features: bool = True,
    output_dir: str = "output",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to build training data from scratch (loads lookup table).

    This is a high-level wrapper that loads the lookup table and generates training data
    in a single call. Useful for programmatic use without manual setup.

    Args:
        lookup_path: Path to pre-built throughput lookup table (.pkl file)
        trace_path: Path to trace CSV file for pipeline stall analysis
        configs: List of microarchitecture configurations to generate features for
        window_size: Window size for pipeline stall analysis (default: 400)
        include_latency_features: Whether to include ROB latency features (default: True)
        output_dir: Directory containing latency .npy files (default: "output")
        output_path: Optional path to save training data (.pkl or .csv). If None, returns DataFrame only.

    Returns:
        DataFrame with training data (N configs x M features)

    Example:
        >>> import models
        >>> from gen_training_data import build_training_data
        >>>
        >>> # Generate training data for multiple configs
        >>> configs = [
        ...     models.Config(rob_size=128, load_queue_size=64),
        ...     models.Config(rob_size=256, load_queue_size=128),
        ... ]
        >>> training_data = build_training_data(
        ...     lookup_path="throughput_lookup.pkl",
        ...     trace_path="traces/my_trace.csv",
        ...     configs=configs,
        ...     window_size=400,
        ...     output_path="training_data.pkl",
        ... )
    """
    # Load lookup table
    print(f"Loading lookup table from: {lookup_path}")
    lookup_table = ThroughputLookupTable.load(lookup_path)

    training_data = generate_training_matrix(
        configs,
        lookup_table,
        trace_path,
        window_size,
        include_latency_features,
        output_dir,
    )

    # Save if requested
    if output_path:
        output_path_obj = Path(output_path).resolve()
        if output_path_obj.suffix.lower() == ".pkl":
            training_data.to_pickle(output_path_obj)
            print(f"\n✓ Training data saved to: {output_path_obj} (pickle format)")
        elif output_path_obj.suffix.lower() == ".csv":
            training_data.to_csv(output_path_obj, index=False)
            print(f"\n✓ Training data saved to: {output_path_obj} (CSV format)")
        else:
            training_data.to_pickle(output_path_obj)
            print(
                f"\n✓ Training data saved to: {output_path_obj} (pickle format, unknown extension)"
            )

    return training_data


def compute_pipeline_stall_features(trace_file: str, window_size: int) -> pd.DataFrame:
    """
    Compute pipeline stall features from a trace file.

    Analyzes the distribution of stall-inducing instructions across sliding windows:
    - ISB (Instruction Sync Barrier)
    - DIRECT_COND (direct conditional branches)
    - DIRECT_UNCOND (direct unconditional branches)
    - INDIRECT (indirect branches)

    Args:
        trace_file: Path to CSV trace file with columns 'Instruction Sync' and 'Branch Type'
        window_size: Number of consecutive instructions per window (must match C++ model)

    Returns:
        DataFrame with 1 row and 4x101=404 columns (4 stall types, 101 features each)

    Example:
        >>> from gen_training_data import compute_pipeline_stall_features
        >>> stall_features = compute_pipeline_stall_features(
        ...     "traces/collatz_trace.csv", window_size=400
        ... )
        >>> print(stall_features.shape)  # (1, 404)
    """
    print(f"Computing pipeline stall features from: {trace_file}")
    print(f"  Window size: {window_size} instructions")

    # Read trace file
    df = pd.read_csv(trace_file)
    num_instructions = len(df)
    num_windows = (num_instructions + window_size - 1) // window_size

    print(f"  Total instructions: {num_instructions:,}")
    print(f"  Number of windows: {num_windows:,}")

    # Boolean masks for each stall type
    raw_masks = {
        "ISB": (df["Instruction Sync"] == True).to_numpy(),
        "DIRECT_COND": (df["Branch Type"] == "direct_conditional").to_numpy(),
        "DIRECT_UNCOND": (df["Branch Type"] == "direct_unconditional").to_numpy(),
        "INDIRECT": (df["Branch Type"] == "indirect").to_numpy(),
    }

    # Count occurrences per window via reshape+sum (vectorized, no Python loop).
    n_full = (num_instructions // window_size) * window_size
    stall_counts = {}
    for name, mask in raw_masks.items():
        counts = mask[:n_full].reshape(-1, window_size).sum(axis=1).astype(float)
        if n_full < num_instructions:  # partial last window
            counts = np.append(counts, float(mask[n_full:].sum()))
        stall_counts[name] = counts

    # Compute features for each stall type
    feature_dict = {}

    for stall_name, counts in stall_counts.items():
        total_count = int(counts.sum())
        print(
            f"  {stall_name}: {total_count:,} total, "
            f"mean {counts.mean():.2f} per window"
        )

        if counts.sum() == 0:
            # If no instructions of this type, use zeros
            print(f"    WARNING: No {stall_name} instructions found, using zeros")
            cdf_raw = np.zeros(50)
            cdf_weighted = np.zeros(50)
            mean_val = 0.0
        else:
            # Compute CDF features from the distribution of counts
            cdf_raw, cdf_weighted, mean_val = utils.compute_cdf_features(
                counts, num_points=50
            )

        # Add to feature dictionary with proper column names
        for i, p in enumerate(utils.PERCENTILE_POINTS):
            feature_dict[f"{stall_name}_raw_p{int(p)}"] = cdf_raw[i]
        for i, p in enumerate(utils.PERCENTILE_POINTS):
            feature_dict[f"{stall_name}_weighted_p{int(p)}"] = cdf_weighted[i]
        feature_dict[f"{stall_name}_mean"] = mean_val

    # Convert to DataFrame
    features_df = pd.DataFrame([feature_dict])

    print(f"  Pipeline stall features: {features_df.shape[1]} columns")
    return features_df


def generate_training_sample(
    config: models.Config,
    lookup_table: ThroughputLookupTable,
    trace_file: str,
    window_size: int,
    include_latency_features: bool = True,
    output_dir: str = "output",
) -> pd.DataFrame:
    """
    Generate a single training sample (one row) combining all features.

    Args:
        config: Microarchitecture configuration
        lookup_table: Pre-built throughput lookup table
        trace_file: Path to trace CSV file
        window_size: Window size for pipeline stall analysis
        include_latency_features: Whether to include ROB latency features (default: True)
        output_dir: Directory containing latency .npy files (default: "output")

    Returns:
        DataFrame with 1 row containing all features:
        - Throughput features (N resources x 101 each)
        - Pipeline stall features (4 stall types x 101 each)
        - ROB latency features (2334 features, if include_latency_features=True)
        - Config scalar features including misprediction_percent (25 features)

    Example:
        >>> import models
        >>> from gen_training_data import generate_training_sample
        >>> from build_throughput_lookup import ThroughputLookupTable
        >>>
        >>> lookup = ThroughputLookupTable.load("throughput_lookup.pkl")
        >>> config = models.Config(rob_size=256, load_queue_size=128)
        >>> sample = generate_training_sample(
        ...     config, lookup, "traces/my_trace.csv", window_size=400
        ... )
        >>> print(sample.shape)  # (1, N_features)
    """
    return generate_training_matrix(
        [config], lookup_table, trace_file, window_size,
        include_latency_features, output_dir,
    )


def generate_training_matrix(
    configs: list[models.Config],
    lookup_table: ThroughputLookupTable,
    trace_file: str,
    window_size: int,
    include_latency_features: bool = True,
    output_dir: str = "output",
) -> pd.DataFrame:
    """
    Generate a training matrix for multiple configurations.

    Args:
        configs: List of microarchitecture configurations
        lookup_table: Pre-built throughput lookup table
        trace_file: Path to trace CSV file
        window_size: Window size for pipeline stall analysis
        include_latency_features: Whether to include ROB latency features (default: True)
        output_dir: Directory containing latency .npy files (default: "output")

    Returns:
        DataFrame with N rows (one per config) and all feature columns

    Example:
        >>> import models
        >>> from gen_training_data import generate_training_matrix
        >>> from build_throughput_lookup import ThroughputLookupTable
        >>>
        >>> lookup = ThroughputLookupTable.load("throughput_lookup.pkl")
        >>> configs = [
        ...     models.Config(rob_size=128, load_queue_size=64),
        ...     models.Config(rob_size=256, load_queue_size=128),
        ...     models.Config(rob_size=512, load_queue_size=256),
        ... ]
        >>> matrix = generate_training_matrix(
        ...     configs, lookup, "traces/my_trace.csv", window_size=400
        ... )
        >>> print(matrix.shape)  # (3, N_features)
    """
    print(f"\nGenerating training matrix for {len(configs)} configurations...")

    # Compute pipeline stall features once (same for all configs)
    stall_features = compute_pipeline_stall_features(trace_file, window_size)

    # Get throughput features for all configs
    throughput_rows = []
    for i, config in enumerate(configs):
        thr_features = lookup_table.get_config_features(config, as_dataframe=True)
        throughput_rows.append(thr_features)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(configs)} configurations...")

    throughput_df = pd.concat(throughput_rows, ignore_index=True)

    # Replicate stall features for each config
    stall_df = pd.concat([stall_features] * len(configs), ignore_index=True)

    # Combine throughput and stall features
    training_data = pd.concat([throughput_df, stall_df], axis=1)

    # Add ROB latency features if requested (same for all configs)
    if include_latency_features:
        latency_features = compute_rob_latency_features(output_dir)
        latency_df = pd.concat([latency_features] * len(configs), ignore_index=True)
        training_data = pd.concat([training_data, latency_df], axis=1)

    # Add config scalar features for each configuration (25 columns per config)
    # Note: misprediction_percent is included in these features and can vary per config
    config_features_list = []
    for config in configs:
        config_features_list.append(get_config_scalar_features(config))
    config_features_df = pd.DataFrame(config_features_list)
    training_data = pd.concat([training_data, config_features_df], axis=1)

    return training_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data by combining throughput and pipeline stall features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Single configuration (pickle format)
  python gen_training_data.py \\
      --lookup throughput_lookup.pkl \\
      --config-json '{"rob_size": 256, "load_queue_size": 128}' \\
      --trace traces/my_trace.csv \\
      --window-size 400 \\
      -o sample.pkl

  # Multiple random configurations (CSV format)
  python gen_training_data.py \\
      --lookup throughput_lookup.pkl \\
      --random-configs 1000 \\
      --trace traces/my_trace.csv \\
      --window-size 400 \\
      -o training_data.csv
        """,
    )

    # Required arguments
    parser.add_argument(
        "--lookup",
        type=str,
        required=True,
        help="Path to pre-built throughput lookup table (.pkl file)",
    )
    parser.add_argument(
        "--trace",
        type=str,
        required=True,
        help="Path to trace CSV file for pipeline stall analysis",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=400,
        help="Window size for pipeline stall analysis (must match C++ model, default: 400)",
    )

    # Configuration specification (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config-json",
        type=str,
        help="Single config as JSON string (e.g., '{\"rob_size\": 256}')",
    )
    config_group.add_argument(
        "--random-configs",
        type=int,
        help="Generate N random configurations",
    )

    # Output
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file for training data (.pkl for pickle, .csv for CSV)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load lookup table
    print("=" * 70)
    print("LOADING LOOKUP TABLE")
    print("=" * 70)
    lookup_table = ThroughputLookupTable.load(args.lookup)

    # Generate configurations
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)

    configs = []
    if args.config_json:
        # Single config from JSON
        config_dict = json.loads(args.config_json)
        config = models.Config(**config_dict)
        configs = [config]
        print(f"Single configuration: {config_dict}")
    else:
        # Multiple random configs (Concorde's approach: independently sample each parameter)
        print(f"Generating {args.random_configs} random configurations...")
        print("  Sampling all 20 parameters independently from valid ranges")
        print("  Microarchitecture space size: ~2x10^23 combinations")

        for i in range(args.random_configs):
            configs.append(sample_random_config())

        print(f"  Generated {len(configs)} configurations")

    # Generate training data
    print("\n" + "=" * 70)
    print("GENERATING TRAINING DATA")
    print("=" * 70)
    print(f"Trace file: {args.trace}")
    print(f"Window size: {args.window_size}")
    print()

    training_data = generate_training_matrix(
        configs,
        lookup_table,
        args.trace,
        args.window_size,
    )

    # Save to file (format based on extension)
    output_path = Path(args.output).resolve()

    if output_path.suffix.lower() == ".pkl":
        training_data.to_pickle(output_path)
        format_name = "pickle"
    elif output_path.suffix.lower() == ".csv":
        training_data.to_csv(output_path, index=False)
        format_name = "CSV"
    else:
        # Default to pickle for unknown extensions
        training_data.to_pickle(output_path)
        format_name = "pickle"
        print(
            f"  Warning: Unknown extension '{output_path.suffix}', using pickle format"
        )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"✓ Training data saved to: {output_path}")
    print(f"  Format: {format_name}")
    print(
        f"  Shape: {training_data.shape[0]} samples x {training_data.shape[1]} features"
    )
    print()


if __name__ == "__main__":
    main()
