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


def load_dataset(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


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


def pre_process_features(features: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    # Drop metric columns (Y in the X -> Y regression problem)
    features = features.drop(columns=["cpi"])

    # Drop columns encoding program region
    benchmark_dummies = pd.get_dummies(features["benchmark"], prefix="benchmark")
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

    if args.drop_cache_cdf_features:
        features = features.drop(columns=[
            "cache_fetch_p50",
            "cache_fetch_p75",
            "cache_fetch_p95",
            "cache_exec_p50",
            "cache_exec_p75",
            "cache_exec_p95",
        ])

    if args.drop_stale_features:
        features = features.drop(columns=[
            "prog_frac_simd",
            "prog_frac_branch",
            "prog_frac_other",
        ])

    if args.drop_all_features:
        cols_to_drop = [col for col in features.columns if col.startswith("prog_") or col.startswith("cache_")]
        features = features.drop(columns=cols_to_drop)
        features = pd.concat([features, benchmark_dummies], axis=1)

    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Peregrine ML model using PyTorch and XLA with the AWS Neuron SDK.")
    parser.add_argument(
        "-d",
        "--dataset-path",
        help="Path to the Peregrine dataset csv file",
    )
    parser.add_argument(
        "--test-size",
        default=0.25,
        type=float,
        help="Fraction of the dataset to use for testing"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for checkpoint and metrics files",
        default="training_results",
    )
    parser.add_argument(
        "--drop-cache-cdf-features",
        help="Drop the 50th/75th/95th %% datapoints for the cache fetch/exec latency distribution",
        action="store_true"
    )
    parser.add_argument(
        "--drop-stale-features",
        help="Drop stale features from the dataset (prog_frac_simd, prog_frac_branch, prog_frac_other)",
        action="store_true"
    )
    parser.add_argument(
        "--drop-all-features",
        help="Drop all program and cache features, using benchmark as one-hot vector",
        action="store_true"
    )

    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)
    print(f'Dataset size: {dataset.shape[0]}')
    print(f"Using train/test split with test size {args.test_size}")
    train_dataset, test_dataset = train_test_split(dataset, test_size=args.test_size, random_state=42)

    print(f"Final split: {len(train_dataset)} train, {len(test_dataset)} test ({len(test_dataset)/(len(train_dataset)+len(test_dataset)):.2%} test)")
    print(f"Train dataset shape: {train_dataset.shape}")
    print(f"Test dataset shape: {test_dataset.shape}")

    train_features = pre_process_features(train_dataset, args)
    test_features = pre_process_features(test_dataset, args)
    print(f"Number of features after ablation: {train_features.shape[1]}")
    for col in train_features.columns:
        print(f"  {col}")

    # Standardize all features (program + config) with a shared StandardScaler
    scaler = StandardScaler()
    train_features = train_features.copy().astype(float)
    test_features = test_features.copy().astype(float)
    train_features[train_features.columns] = scaler.fit_transform(train_features[train_features.columns])
    test_features[train_features.columns] = scaler.transform(test_features[train_features.columns])

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
    epochs = 200
    model = PeregrineMLModel(input_size=train_features.shape[1], hidden_dims=[256, 128], output_size=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001) #, weight_decay=0.3)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 6000, 7000, 8000], gamma=0.5)
    loss_fn = nn.L1Loss()

    early_stop_patience = 10
    best_eval_loss = float("inf")
    epochs_without_improvement = 0

    print('----------- Start Training --------------')
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
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"Early stopping: no improvement for {early_stop_patience} epochs.")
                break

    print('------------ End Training ---------------')
    total_duration = time.perf_counter() - total_start

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.output_dir, f"checkpoint.pt")
    checkpoint = {'state_dict': model.state_dict()}
    xm.save(checkpoint, checkpoint_path)

    # Persist preprocessing artifacts so inference can reproduce training transforms.
    scaler_path = os.path.join(args.output_dir, f"scaler.joblib")
    joblib.dump(scaler, scaler_path)

    ablation_flags = {
        "drop_cache_cdf_features": bool(args.drop_cache_cdf_features),
        "drop_stale_features": bool(args.drop_stale_features),
        "drop_all_features": bool(args.drop_all_features),
    }
    best_epoch = min(epochs_data, key=lambda e: e["eval_loss"]) if epochs_data else None

    metrics_path = os.path.join(args.output_dir, f"metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_duration": total_duration,
                "ablation_flags": ablation_flags,
                "num_features": int(train_features.shape[1]),
                "best_eval_loss": float(best_epoch["eval_loss"]) if best_epoch else None,
                "best_percent_error": float(best_epoch["percent_error"]) if best_epoch else None,
                "epochs": epochs_data,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
