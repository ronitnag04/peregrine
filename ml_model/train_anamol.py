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


DROP_COLS = ("row_id", "stride_prefetcher_degree", "branch_predictor")
ID_COLS = ("benchmark", "checkpoint", "fast_forward")
LABEL_COL = "cpi"
MEMSIZE_COLS = ("l1d_size", "l1i_size", "l2_size")


def _convert_memsize(val) -> float:
    s = str(val)
    if "KiB" in s:
        return float(s.replace("KiB", "")) * 1024.0
    if "MiB" in s:
        return float(s.replace("MiB", "")) * 1024.0 * 1024.0
    return float(s)


def _cache_paths(cache_dir: str) -> dict[str, str]:
    return {
        "dir": cache_dir,
        "features": os.path.join(cache_dir, "features.npy"),
        "labels": os.path.join(cache_dir, "labels.npy"),
        "ids": os.path.join(cache_dir, "ids.pkl"),
        "columns": os.path.join(cache_dir, "columns.json"),
    }


def _cache_is_valid(cache_dir: str, csv_path: str | None = None) -> bool:
    paths = _cache_paths(cache_dir)
    required = [paths["features"], paths["labels"], paths["ids"], paths["columns"]]
    if not all(os.path.exists(p) for p in required):
        return False
    if csv_path is None:
        return True
    csv_mtime = os.path.getmtime(csv_path)
    cache_mtime = min(os.path.getmtime(p) for p in required)
    return cache_mtime >= csv_mtime


def build_cache(csv_path: str, cache_dir: str) -> None:
    """Parse the CSV once and write a compact float32 cache next to it.

    The cache contains the preprocessed feature matrix (dropped/encoded columns
    already removed, memsize strings converted to bytes) so subsequent runs can
    skip CSV parsing entirely.
    """
    paths = _cache_paths(cache_dir)
    os.makedirs(paths["dir"], exist_ok=True)

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    drop_set = set(DROP_COLS) | set(ID_COLS) | {LABEL_COL}
    feature_cols = [c for c in header if c not in drop_set]
    n_features = len(feature_cols)

    print(f"  [cache] reading labels + id columns ...")
    meta = pd.read_csv(
        csv_path,
        usecols=[LABEL_COL, *ID_COLS],
        dtype={LABEL_COL: np.float32},
    )
    n_rows = len(meta)

    np.save(paths["labels"], meta[LABEL_COL].to_numpy(dtype=np.float32, copy=False))
    meta[list(ID_COLS)].reset_index(drop=True).to_pickle(paths["ids"])
    del meta

    print(f"  [cache] building float32 feature matrix: {n_rows} x {n_features}")
    features = np.lib.format.open_memmap(
        paths["features"], mode="w+", dtype=np.float32, shape=(n_rows, n_features)
    )

    memsize_set = set(MEMSIZE_COLS)
    chunk_size = 32*1024
    offset = 0
    last_log = 0
    reader = pd.read_csv(
        csv_path,
        usecols=feature_cols,
        chunksize=chunk_size,
        dtype={c: str for c in MEMSIZE_COLS},
    )
    for chunk in reader:
        for col in MEMSIZE_COLS:
            chunk[col] = chunk[col].map(_convert_memsize)
        size = len(chunk)
        # Reorder to match feature_cols (read_csv preserves header order, but be explicit)
        features[offset:offset + size] = chunk[feature_cols].to_numpy(dtype=np.float32, copy=False)
        offset += size
        if offset - last_log >= 100_000 or offset == n_rows:
            print(f"  [cache]   {offset}/{n_rows} rows")
            last_log = offset

    features.flush()
    del features

    with open(paths["columns"], "w", encoding="utf-8") as f:
        json.dump(feature_cols, f)

    print(f"  [cache] wrote {paths['dir']}")


def load_dataset(dataset_path: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Return (features_memmap, labels, ids, feature_columns).

    Accepts either the source CSV (in which case a cache is built next to it
    as ``<csv>.cache/``) or a pre-built cache directory.

    features is a read-only float32 memmap; labels is a float32 ndarray;
    ids is a DataFrame with benchmark/checkpoint/fast_forward.
    """
    if os.path.isdir(dataset_path):
        cache_dir = dataset_path.rstrip("/")
        if not _cache_is_valid(cache_dir):
            raise FileNotFoundError(
                f"{cache_dir} is a directory but does not contain a complete cache "
                f"(expected features.npy, labels.npy, ids.pkl, columns.json)"
            )
    else:
        csv_path = dataset_path
        cache_dir = csv_path + ".cache"
        if not _cache_is_valid(cache_dir, csv_path):
            print(f"Building cache for {cache_dir} (one-time; ~14GB CSV)")
            build_cache(csv_path, cache_dir)
    paths = _cache_paths(cache_dir)
    features = np.load(paths["features"], mmap_mode="r")
    labels = np.load(paths["labels"])
    ids = pd.read_pickle(paths["ids"])
    with open(paths["columns"], encoding="utf-8") as f:
        feature_cols = json.load(f)
    return features, labels, ids, feature_cols


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


def write_predictions_csv(
    output_dir: str,
    suffix: str,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_ids: pd.DataFrame,
    model: nn.Module,
    device: str,
) -> tuple[float, float]:
    start_time = time.perf_counter()
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(test_features).to(device)
        predictions = model(inputs).detach().cpu().squeeze(1).numpy()
        torch_xla.sync()
    duration = time.perf_counter() - start_time

    predictions_df = pd.DataFrame(
        {
            "benchmark": test_ids["benchmark"].to_numpy(copy=False),
            "checkpoint": test_ids["checkpoint"].to_numpy(copy=False),
            "fast_forward": test_ids["fast_forward"].to_numpy(copy=False),
            "actual_cpi": test_labels,
            "predicted_cpi": predictions,
        }
    )
    predictions_path = os.path.join(output_dir, f"{suffix}_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    eps = 1e-8
    percent_error = float(
        (
            (predictions_df["predicted_cpi"] - predictions_df["actual_cpi"]).abs()
            / (predictions_df["actual_cpi"].abs() + eps)
        ).mean()
        * 100.0
    )
    return duration, percent_error


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
        "--add-test-predictions",
        action="store_true",
        help="Write a CSV with actual/predicted CPI plus benchmark, checkpoint, and fast_forward for the test dataset.",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    print(f"Loading dataset from {args.dataset_path}")
    features, labels, ids, feature_cols = load_dataset(args.dataset_path)
    n_rows = features.shape[0]
    print(f"Dataset size: {n_rows}")
    print(f"Feature dim: {features.shape[1]}")

    print(f"Using train/test split with test size {args.test_size}")
    all_idx = np.arange(n_rows)
    train_idx, test_idx = train_test_split(all_idx, test_size=args.test_size, random_state=42)

    # Fancy indexing on a memmap materializes just these rows as a contiguous
    # float32 ndarray — one copy in RAM, not four.
    train_features = features[train_idx]
    test_features = features[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    test_ids = ids.iloc[test_idx].reset_index(drop=True)

    print(f"Final split: {len(train_idx)} train, {len(test_idx)} test ({len(test_idx)/n_rows:.2%} test)")
    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")

    # Standardize in place to avoid doubling memory.
    scaler = StandardScaler(copy=False)
    scaler.fit(train_features)
    scaler.transform(train_features)
    scaler.transform(test_features)

    # Zero-copy handoff to torch (arrays are already contiguous float32).
    train_features_t = torch.from_numpy(train_features)
    train_labels_t = torch.from_numpy(train_labels).unsqueeze(1)
    test_features_t = torch.from_numpy(test_features)
    test_labels_t = torch.from_numpy(test_labels).unsqueeze(1)

    train_ds = TensorDataset(train_features_t, train_labels_t)
    test_ds = TensorDataset(test_features_t, test_labels_t)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    device = "xla"
    epochs = 1600
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

    test_predictions_duration = None
    test_predictions_percent_error = None

    if args.add_test_predictions:
        test_predictions_duration, test_predictions_percent_error = write_predictions_csv(
            args.output_dir,
            "test",
            test_features,
            test_labels,
            test_ids,
            model,
            device,
        )
        print(f"Test predictions time: {test_predictions_duration:.2f}s")
        print(f"Test predictions percent error: {test_predictions_percent_error:.2f}%")

    metrics_path = os.path.join(args.output_dir, f"metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_duration": total_duration,
                "test_predictions_duration": test_predictions_duration,
                "test_predictions_percent_error": test_predictions_percent_error,
                "epochs": epochs_data,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
