import argparse
import json
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

from model import PeregrineMLModel

# XLA imports
import torch_xla.core.xla_model as xm
import torch_xla


def load_dataset(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """One training epoch; returns mean L1 loss and mean absolute percent error on train batches."""
    model.train()
    total_loss = torch.zeros((), device=device)
    total_percent_error = torch.zeros((), device=device)
    num_batches = 0
    eps = torch.tensor(1e-8, device=device)
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        torch_xla.sync()
        total_loss += loss.detach()
        batch_percent_error = (
            (torch.abs(outputs - targets) / (torch.abs(targets) + eps)).mean() * 100.0
        )
        total_percent_error += batch_percent_error.detach()
        num_batches += 1
    denom = max(num_batches, 1)
    mean_loss = total_loss / denom
    mean_percent_error = total_percent_error / denom
    return mean_loss.item(), mean_percent_error.item()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Peregrine ML model using PyTorch and XLA with the AWS Neuron SDK (training only, no held-out evaluation).")
    parser.add_argument(
        "-d",
        "--dataset-path",
        help="Path to the Peregrine dataset csv file",
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

    train_features = pre_process_features(dataset)

    scaler = StandardScaler()
    train_features = train_features.copy().astype(float)
    train_features[train_features.columns] = scaler.fit_transform(train_features[train_features.columns])

    train_labels = dataset["cpi"]

    train_features_t = torch.from_numpy(train_features.to_numpy(copy=True)).float()
    train_labels_t = torch.from_numpy(train_labels.to_numpy(copy=True)).float().unsqueeze(1)

    train_ds = TensorDataset(train_features_t, train_labels_t)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    device = "xla"
    epochs = 200
    model = PeregrineMLModel(input_size=train_features.shape[1], hidden_dims=[256, 128], output_size=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)  # , weight_decay=0.3)
    loss_fn = nn.L1Loss()

    early_stop_patience = 10
    best_train_loss = float("inf")
    epochs_without_improvement = 0

    print("----------- Start Training --------------")
    epochs_data: list[dict[str, float]] = []
    total_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_percent_error = train_epoch(model, train_loader, loss_fn, optimizer, device)
        epoch_duration = time.perf_counter() - epoch_start
        epochs_data.append(
            {
                "duration": epoch_duration,
                "train_loss": float(train_loss),
                "train_percent_error": float(train_percent_error),
            }
        )
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Error: {train_percent_error:.2f}% | "
            f"Time: {epoch_duration:.2f}s"
        )

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"Early stopping: no improvement in train loss for {early_stop_patience} epochs.")
                break

    print("------------ End Training ---------------")
    total_duration = time.perf_counter() - total_start

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pt")
    checkpoint = {"state_dict": model.state_dict()}
    xm.save(checkpoint, checkpoint_path)

    # Persist preprocessing artifacts so inference can reproduce training transforms.
    scaler_path = os.path.join(args.output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_duration": total_duration,
                "epochs": epochs_data,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
