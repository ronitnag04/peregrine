import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from model import PeregrineMLModel

# XLA imports
import torch_xla.core.xla_model as xm
import torch_xla

# Default names for adversarial C benchmarks (see peregrine/adversarial_benchmarks/README.md)
ADVERSARIAL_BENCHMARKS = [
    "adversarial_branches",
    "icache_blast",
    "many_pages_streaming",
    "pow2_stride_benign",
    "pow2_stride_thrash",
    "ptrchase_rand",
    "serial_mul_chain",
    "stlf_misalign",
]

SPEC_BENCHMARKS = [
    "505.mcf_r",
    "502.gcc_r",
    "520.omnetpp_r",
    "523.xalancbmk_r",
    "525.x264_r",
    "531.deepsjeng_r",
    "541.leela_r",
    "548.exchange2_r",
    "557.xz_r",
]

_DEFAULT_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "training_data",
    "ronamol_spec_v3_adversarial_v1.csv",
)


def load_dataset(dataset_path: str) -> pd.DataFrame:
    # Standard header: cpi,benchmark,... (no leading index column)
    return pd.read_csv(dataset_path)


def expand_train_benchmark_tokens(
    raw: str,
    *,
    ood_benchmark: str | None,
) -> list[str]:
    out: list[str] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        if t == "spec":
            out.extend(SPEC_BENCHMARKS)
        elif t == "adversarial_all":
            out.extend(ADVERSARIAL_BENCHMARKS)
        elif t == "adversarial_except_ood":
            if not ood_benchmark:
                raise ValueError("adversarial_except_ood requires --ood-benchmark")
            out.extend([b for b in ADVERSARIAL_BENCHMARKS if b != ood_benchmark])
        else:
            out.append(t)
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for b in out:
        if b not in seen:
            seen.add(b)
            deduped.append(b)
    return deduped


def train_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        torch_xla.sync()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_percent_error = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            torch_xla.sync()
            total_loss += loss.item()
            # Compute mean absolute percent error for this batch.
            # Add small epsilon to denominator to avoid division by zero.
            eps = 1e-8
            batch_percent_error = (
                (torch.abs(outputs - targets) / (torch.abs(targets) + eps)).mean() * 100.0
            )
            total_percent_error += batch_percent_error.item()
            num_batches += 1

    denom = max(num_batches, 1)
    avg_loss = total_loss / denom
    avg_percent_error = total_percent_error / denom
    return float(avg_loss), float(avg_percent_error)


def pre_process_features(features: pd.DataFrame) -> pd.DataFrame:
    # Drop metric columns (Y in the X -> Y regression problem)
    features = features.drop(columns=["cpi"])

    # Drop columns encoding program region
    features = features.drop(columns=["benchmark", "checkpoint", "fast_forward"])

    # Split stride_prefetch into 1-hot encoded columns
    stride_prefetch_dummies = pd.get_dummies(features["stride_prefetcher_degree"], prefix="stride_prefetcher_degree")
    features = features.drop(columns=["stride_prefetcher_degree"])
    features = pd.concat([features, stride_prefetch_dummies], axis=1)

    # Drop branch_predictor (captured by misprediction rate column)
    features = features.drop(columns=["branch_predictor"])

    memsize_colummns = ["l1d_size", "l1i_size", "l2_size"]
    for col in memsize_colummns:
        features[col] = features[col].apply(lambda x: 
            int(x.replace("KiB", "")) * 1024 if "KiB" in x 
            else int(x.replace("MiB", "")) * 1024**2 if "MiB" in x 
            else int(x))

    # Reciprocal values for queue sizes
    queue_size_columns = ["lq_entries", "sq_entries", "rob_size"]
    for col in queue_size_columns:
        reciprocal_col_name = f"reciprocal_{col}"
        features[reciprocal_col_name] = 1.0 / features[col]

    return features


def build_train_test(
    dataset: pd.DataFrame,
    args: argparse.Namespace,
    train_benchmarks_expanded: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (train_df, test_df, split_metadata)."""
    meta: dict = {"experiment": args.experiment}

    if args.experiment == "adversarial_only":
        if not args.ood_benchmark:
            raise ValueError("adversarial_only requires --ood-benchmark (single benchmark to fit)")
        pool = dataset[dataset["benchmark"] == args.ood_benchmark].copy()
        if len(pool) < 2:
            raise ValueError(f"Not enough rows for benchmark {args.ood_benchmark!r}: {len(pool)}")
        strat = None
        if args.stratify_cpi_bins > 0:
            try:
                strat = pd.qcut(
                    pool["cpi"],
                    q=min(args.stratify_cpi_bins, len(pool)),
                    duplicates="drop",
                )
            except ValueError:
                strat = None
        train_dataset, test_dataset = train_test_split(
            pool,
            test_size=args.in_benchmark_test_fraction,
            random_state=args.random_seed,
            shuffle=True,
            stratify=strat,
        )
        meta.update(
            {
                "ood_benchmark": args.ood_benchmark,
                "in_benchmark_test_fraction": args.in_benchmark_test_fraction,
                "random_seed": args.random_seed,
                "n_train": len(train_dataset),
                "n_test": len(test_dataset),
            }
        )
        return train_dataset, test_dataset, meta

    if args.experiment == "spec_holdout_adversarial":
        if not args.ood_benchmark:
            raise ValueError("spec_holdout_adversarial requires --ood-benchmark")
        # Hold out all rows of the OOD benchmark from training, then optionally onboard a slice.
        test_dataset = dataset[dataset["benchmark"] == args.ood_benchmark].copy()
        train_dataset = dataset[
            dataset["benchmark"].isin(train_benchmarks_expanded)
            & (dataset["benchmark"] != args.ood_benchmark)
        ].copy()

        ood_n = int(args.ood_train_size)
        if ood_n > 0:
            if len(test_dataset) < ood_n:
                raise ValueError(
                    f"--ood-train-size {ood_n} exceeds OOD pool size {len(test_dataset)} for {args.ood_benchmark!r}"
                )
            ood_train_dataset = test_dataset.sample(n=ood_n, random_state=args.random_seed)
            train_dataset = pd.concat([train_dataset, ood_train_dataset], axis=0)
            test_dataset = test_dataset.drop(ood_train_dataset.index)

        meta.update(
            {
                "ood_benchmark": args.ood_benchmark,
                "ood_train_size": ood_n,
                "train_benchmarks": train_benchmarks_expanded,
                "n_train": len(train_dataset),
                "n_test": len(test_dataset),
            }
        )
        if len(train_dataset) == 0:
            raise ValueError(
                "spec_holdout_adversarial produced an empty training set. "
                "Use --train-benchmarks that includes rows outside --ood-benchmark, "
                "or set --ood-train-size > 0."
            )
        if len(test_dataset) == 0:
            raise ValueError(
                "Test set is empty: check --ood-benchmark and --ood-train-size "
                "(the latter cannot consume the entire OOD pool)."
            )
        return train_dataset, test_dataset, meta

    raise ValueError(f"Unknown --experiment {args.experiment!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Peregrine ML on spec + adversarial data (see adversarial_benchmarks README). "
            "Default dataset is ronamol_spec_v3_adversarial_v1.csv."
        )
    )
    parser.add_argument(
        "-d",
        "--dataset-path",
        help="Path to the Peregrine dataset CSV (default: bundled spec+adversarial v1)",
        default=_DEFAULT_DATASET,
    )
    parser.add_argument(
        "--experiment",
        choices=["spec_holdout_adversarial", "adversarial_only"],
        default="spec_holdout_adversarial",
        help=(
            "spec_holdout_adversarial: train on --train-benchmarks excluding the OOD name, "
            "test on --ood-benchmark; optional --ood-train-size 'onboards' samples into training. "
            "adversarial_only: train/test split within a single --ood-benchmark."
        ),
    )
    parser.add_argument(
        "--train-benchmarks",
        type=str,
        help=(
            "Comma-separated training benchmarks. Tokens: 'spec', 'adversarial_all', "
            "'adversarial_except_ood', or explicit benchmark names. "
            "Used only for spec_holdout_adversarial."
        ),
        default="spec",
    )
    parser.add_argument(
        "--ood-benchmark",
        type=str,
        help="Held-out benchmark for evaluation (required)",
    )
    parser.add_argument(
        "--ood-train-size",
        type=int,
        help="Number of OOD points moved from the test pool into training (spec_holdout_adversarial only)",
        default=0,
    )
    parser.add_argument(
        "--in-benchmark-test-fraction",
        type=float,
        default=0.2,
        help="Test fraction for adversarial_only (random split within the OOD benchmark)",
    )
    parser.add_argument(
        "--stratify-cpi-bins",
        type=int,
        default=0,
        help="If >0, stratify adversarial_only split by CPI quantile bins (0 = disabled)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="RNG seed for sampling and adversarial_only split",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for checkpoint and metrics files",
        default="training_results",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)
    print(f"Dataset size: {dataset.shape[0]}")

    train_benchmarks_expanded = expand_train_benchmark_tokens(
        args.train_benchmarks,
        ood_benchmark=args.ood_benchmark,
    )
    print(f"Resolved train-benchmark tokens -> {len(train_benchmarks_expanded)} names (showing first 12): {train_benchmarks_expanded[:12]!r}")

    train_dataset, test_dataset, split_meta = build_train_test(dataset, args, train_benchmarks_expanded)

    print(f"Experiment: {args.experiment}")
    print(f"Final split: {len(train_dataset)} train, {len(test_dataset)} test")
    if len(train_dataset) + len(test_dataset) > 0:
        frac = len(test_dataset) / (len(train_dataset) + len(test_dataset))
        print(f"Test fraction of combined pool: {frac:.2%}")
    print(f"Train dataset shape: {train_dataset.shape}")
    print(f"Test dataset shape: {test_dataset.shape}")

    train_features = pre_process_features(train_dataset)
    test_features = pre_process_features(test_dataset)

    scaler = StandardScaler()
    train_features = train_features.copy().astype(float)
    test_features = test_features.copy().astype(float)
    train_features[train_features.columns] = scaler.fit_transform(train_features[train_features.columns])
    test_features[test_features.columns] = scaler.transform(test_features[test_features.columns])

    train_labels = train_dataset["cpi"]
    test_labels = test_dataset["cpi"]

    train_features_t = torch.from_numpy(train_features.to_numpy(copy=True)).float()
    train_labels_t = torch.from_numpy(train_labels.to_numpy(copy=True)).float().unsqueeze(1)
    test_features_t = torch.from_numpy(test_features.to_numpy(copy=True)).float()
    test_labels_t = torch.from_numpy(test_labels.to_numpy(copy=True)).float().unsqueeze(1)

    train_ds = TensorDataset(train_features_t, train_labels_t)
    test_ds = TensorDataset(test_features_t, test_labels_t)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    device = "xla"
    epochs = 500
    model = PeregrineMLModel(input_size=train_features.shape[1], hidden_dims=[256, 128], output_size=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.L1Loss()

    early_stop_patience = 50
    best_eval_loss = float("inf")
    best_percent_error = float("inf")
    epochs_without_improvement = 0

    print("----------- Start Training --------------")
    epochs_data: list[dict[str, float]] = []
    total_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        eval_loss, percent_error = evaluate(model, test_loader, loss_fn, device)
        epoch_duration = time.perf_counter() - epoch_start
        epochs_data.append(
            {
                "duration": epoch_duration,
                "train_loss": float(train_loss),
                "eval_loss": float(eval_loss),
                "percent_error": float(percent_error),
            }
        )
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Eval Loss: {eval_loss:.4f} | "
            f"Percent Error: {percent_error:.2f}% | "
            f"Time: {epoch_duration:.2f}s"
        )

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_percent_error = float(percent_error)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"Early stopping: no improvement for {early_stop_patience} epochs.")
                break

    print("------------ End Training ---------------")
    total_duration = time.perf_counter() - total_start

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pt")
    checkpoint = {"state_dict": model.state_dict()}
    xm.save(checkpoint, checkpoint_path)

    scaler_path = os.path.join(args.output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    run_config = {
        "dataset_path": os.path.abspath(args.dataset_path),
        "experiment": args.experiment,
        "train_benchmarks_arg": args.train_benchmarks,
        "ood_benchmark": args.ood_benchmark,
        "ood_train_size": args.ood_train_size,
        "in_benchmark_test_fraction": args.in_benchmark_test_fraction,
        "stratify_cpi_bins": args.stratify_cpi_bins,
        "random_seed": args.random_seed,
        "split": split_meta,
        "best_eval_loss": best_eval_loss,
        "best_percent_error": best_percent_error,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_duration": total_duration,
                "run_config": run_config,
                "epochs": epochs_data,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
