import argparse
import json
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from model import PeregrineMLModel

# XLA imports
import torch_xla.core.xla_model as xm
import torch_xla


def load_dataset(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path, index_col=0)


def train_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.view(inputs.size(0), -1)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        torch_xla.sync()
        total_loss += loss.detach().to("cpu")
    return total_loss / max(len(loader), 1)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_percent_error = 0.0
    num_batches = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.view(inputs.size(0), -1)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            torch_xla.sync() 
            total_loss += loss.detach().to("cpu")
            # Compute mean absolute percent error for this batch.
            # Add small epsilon to denominator to avoid division by zero.
            eps = 1e-8
            batch_percent_error = (
                (torch.abs(outputs - targets) / (torch.abs(targets) + eps)).mean() * 100.0
            )
            total_percent_error += batch_percent_error.detach().to("cpu")
            num_batches += 1

    denom = max(num_batches, 1)
    avg_loss = total_loss / denom
    avg_percent_error = total_percent_error / denom
    return float(avg_loss), float(avg_percent_error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Peregrine ML model using PyTorch and XLA with the AWS Neuron SDK.")
    parser.add_argument(
        "--dataset-path",
        help="Path to the Peregrine dataset csv file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = os.path.basename(args.dataset_path).split('.')[0]
    dataset = load_dataset(args.dataset_path)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.25, random_state=42)

    train_features = train_dataset.drop(columns=["cpi", "branch_predictor", "l1d_stride_prefetch"])
    train_labels = train_dataset["cpi"]
    test_features = test_dataset.drop(columns=["cpi", "branch_predictor", "l1d_stride_prefetch"])
    test_labels = test_dataset["cpi"]

    # Standardize features
    train_features = (train_features - train_features.mean()) / train_features.std()
    test_features = (test_features - test_features.mean()) / test_features.std()

    # Convert to float tensors and reshape labels to (n, 1) for MSELoss compatibility
    train_features_t = torch.from_numpy(train_features.values).float()
    train_labels_t = torch.from_numpy(train_labels.values).float().unsqueeze(1)
    test_features_t = torch.from_numpy(test_features.values).float()
    test_labels_t = torch.from_numpy(test_labels.values).float().unsqueeze(1)

    train_ds = TensorDataset(train_features_t, train_labels_t)
    test_ds = TensorDataset(test_features_t, test_labels_t)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    device = "xla"
    epochs = 1500
    model = PeregrineMLModel(input_size=train_features.shape[1], hidden_dims=[256, 128], output_size=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001) #, weight_decay=0.3)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 6000, 7000, 8000], gamma=0.5)
    loss_fn = nn.MSELoss()

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
            f"Percent Error: {percent_error:.2f}%"
        )
    print('------------ End Training ---------------')
    total_duration = time.perf_counter() - total_start

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", f"{dataset_name}_checkpoint.pt")
    checkpoint = {'state_dict': model.state_dict()}
    xm.save(checkpoint, checkpoint_path)

    os.makedirs("training_metrics", exist_ok=True)
    metrics_path = os.path.join("training_metrics", f"{dataset_name}_metrics.json")
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