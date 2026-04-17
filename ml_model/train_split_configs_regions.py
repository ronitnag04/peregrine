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


def pre_process_features(features: pd.DataFrame) -> pd.DataFrame:
    # Drop metric columns (Y in the X -> Y regression problem)
    features = features.drop(columns=["cpi"])

    # Drop columns encoding program region
    features = features.drop(columns=["benchmark", "ff_instructions"])

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
    parser = argparse.ArgumentParser(description="Train the Peregrine ML model using PyTorch and XLA with the AWS Neuron SDK.")
    parser.add_argument(
        "-d",
        "--dataset-path",
        help="Path to the Peregrine dataset csv file",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        help="Fraction of the dataset to use for testing"
    )
    parser.add_argument(
        "--test-benchmarks",
        help="Comma-separated list of benchmarks to use for testing",
        default="",
    )
    parser.add_argument(
        "--test-regions",
        type=float,
        help="Fraction of each benchmark's regions to use for testing",
    )
    parser.add_argument(
        "--test-configs",
        type=float,
        help="Fraction of configs to use for testing",
    )
    parser.add_argument(
        "--drop-benchmarks",
        help="Comma-separated list of benchmarks to drop entirely from training and testing",
        default="",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for checkpoint and metrics files",
        default=".",
    )

    args = parser.parse_args()

    if args.test_size is not None and (args.test_benchmarks or args.test_regions or args.test_configs):
        parser.error("Cannot specify both test_size and test_benchmarks, test_regions, or test_configs")

    if not (args.test_benchmarks or args.test_regions or args.test_configs) and args.test_size is None:
        parser.error("Must specify either test_size or test_benchmarks, test_regions, or test_configs")

    if args.test_benchmarks and args.test_regions:
        parser.error("Cannot specify both test_benchmarks and test_regions")

    return args


def main() -> None:
    args = parse_args()
    dataset_name = os.path.basename(args.dataset_path).split('.')[0]
    dataset = load_dataset(args.dataset_path)
    print(f'Dataset size: {dataset.shape[0]}')

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.drop_benchmarks:
        drop_list = [b.strip() for b in args.drop_benchmarks.split(",") if b.strip()]
        if drop_list:
            before = len(dataset)
            dataset = dataset[~dataset["benchmark"].isin(drop_list)]
            after = len(dataset)
            print(f"Dropped benchmarks {drop_list}: {before - after} rows removed, {after} remaining.")

    if args.test_size:
        print(f"Using train/test split with test size {args.test_size}")
        train_dataset, test_dataset = train_test_split(dataset, test_size=args.test_size, random_state=42)
    else:
        data_points = dataset.shape[0]

        train_dataset = dataset
        test_dataset = None

        if args.test_configs:
            config_columns = ['branch_predictor', 'commit_width', 'decode_width', 'fetch_width', 'fp_mult_div_issue_width', 'fp_reg_issue_width',
                                'int_mult_div_issue_width', 'int_reg_issue_width', 'l1d_size', 'l1i_size', 'l2_size', 'lq_entries', 
                                'max_icache_fills', 'rdwr_port_issue_width', 'read_port_issue_width', 'rename_width', 'rob_size', 
                                'simd_unit_issue_width', 'sq_entries', 'stride_prefetcher_degree']

            configs = dataset.groupby(config_columns).size().reset_index()[config_columns]

            configs_shuffled = configs.sample(frac=1, random_state=42).reset_index(drop=True)
            num_configs = len(configs_shuffled)
            split_idx = int(num_configs * args.test_configs)

            test_configs_df = configs_shuffled.iloc[:split_idx]
            
            print(f"Moving {len(test_configs_df)} configurations ({args.test_configs:.1%}) to test set")

            test_dataset = dataset.merge(test_configs_df, on=config_columns, how='inner')
            train_dataset = dataset.merge(test_configs_df, on=config_columns, how='outer', indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])

        if args.test_regions:
            print(f"Moving {args.test_regions:.1%} of regions from each benchmark to test set")
            if test_dataset is None:
                test_dataset = train_dataset
            for benchmark in dataset["benchmark"].unique():
                benchmark_regions = dataset[dataset["benchmark"] == benchmark]["ff_instructions"].unique()
                num_test_regions = int(args.test_regions * len(benchmark_regions))

                if num_test_regions > 0:
                    benchmark_test_regions = np.random.choice(benchmark_regions, size=num_test_regions, replace=False)
                    
                    benchmark_mask = train_dataset["benchmark"] == benchmark
                    region_mask = ~train_dataset["ff_instructions"].isin(benchmark_test_regions)
                    train_dataset = train_dataset[~benchmark_mask | region_mask]
                    
                    benchmark_mask = test_dataset["benchmark"] == benchmark
                    region_mask = test_dataset["ff_instructions"].isin(benchmark_test_regions)
                    test_dataset = test_dataset[~benchmark_mask | region_mask]
                else:
                    raise ValueError(f"num_test_regions is 0 for benchmark {benchmark}")
        elif args.test_benchmarks:
            test_benchmarks_names = [b.strip() for b in args.test_benchmarks.split(",") if b.strip()]
            print(f"Moving benchmarks {args.test_benchmarks} to test set")

            if test_dataset is None:
                test_dataset = train_dataset

            mask = train_dataset["benchmark"].isin(test_benchmarks_names)
            train_dataset = train_dataset[~mask]
            mask = test_dataset["benchmark"].isin(test_benchmarks_names)
            test_dataset = test_dataset[mask]

    print(f"Final split: {len(train_dataset)} train, {len(test_dataset)} test ({len(test_dataset)/(len(train_dataset)+len(test_dataset)):.2%} test)")
    print(f"Train dataset shape: {train_dataset.shape}")
    print(f"Test dataset shape: {test_dataset.shape}")

    train_features = pre_process_features(train_dataset)
    test_features = pre_process_features(test_dataset)

    # Standardize all features (program + config) with a shared StandardScaler
    scaler = StandardScaler()
    train_features = train_features.copy().astype(float)
    test_features = test_features.copy().astype(float)
    train_features[train_features.columns] = scaler.fit_transform(train_features[train_features.columns])
    test_features[train_features.columns] = scaler.transform(test_features[train_features.columns])

    train_labels = train_dataset["cpi"]
    test_labels = test_dataset["cpi"]

    train_features_t = torch.from_numpy(train_features.to_numpy(copy=True)).float()
    train_labels_t = torch.from_numpy(train_labels.to_numpy(copy=True, dtype=float)).float().unsqueeze(1)
    test_features_t = torch.from_numpy(test_features.to_numpy(copy=True)).float()
    test_labels_t = torch.from_numpy(test_labels.to_numpy(copy=True, dtype=float)).float().unsqueeze(1)

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

    checkpoint_path = os.path.join(args.output_dir, f"checkpoint.pt")
    checkpoint = {'state_dict': model.state_dict()}
    xm.save(checkpoint, checkpoint_path)

    # Persist preprocessing artifacts so inference can reproduce training transforms.
    scaler_path = os.path.join(args.output_dir, f"scaler.joblib")
    joblib.dump(scaler, scaler_path)

    metrics_path = os.path.join(args.output_dir, f"metrics.json")
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
