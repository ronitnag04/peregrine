"""
Generate predictions for every parameter combination using trained checkpoints.

This script:
- Loads model checkpoint from the local `checkpoints/` dir
- Reads a param_Sweep.csv file containing the parameter combinations to predict on
- Outputs predictions to a predictions.npy file
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import torch
import torch_neuronx
import torch_xla.core.xla_model as xm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

from model import PeregrineMLModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Peregrine ML model predictions over parameter sweep.")
    parser.add_argument(
        "--parameter-sweep",
        type=str,
        default="prediction_sweep.pkl",
        help="Path to prediction sweep.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="checkpoints/sweep_results_16ki_checkpoint.pt",
        help="Path to PeregrineMlModel checkpoint.pt.",
    )
    parser.add_argument(
        "--scaler-file",
        type=str,
        default="checkpoints/sweep_results_16ki_scaler.joblib",
        help="Path to StandardScaler (joblib).",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, input_size: int, device: torch.device) -> PeregrineMLModel:
    model = PeregrineMLModel(input_size=input_size, hidden_dims=[256, 128], output_size=1).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def load_scaler(scaler_path: Path) -> StandardScaler:
    return joblib.load(scaler_path)


def load_parameter_sweep(parameter_sweep_path: Path) -> pd.DataFrame:
    if parameter_sweep_path.endswith(".csv"):
        return pd.read_csv(parameter_sweep_path)
    elif parameter_sweep_path.endswith(".pkl"):
        return pd.read_pickle(parameter_sweep_path)
    else:
        raise ValueError(f"Unsupported parameter sweep format: {parameter_sweep_path}")


def one_hot_encode_categorical_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    benchmark_dummies = pd.get_dummies(dataset["benchmark"], prefix="benchmark")
    branch_predictor_dummies = pd.get_dummies(dataset["branch_predictor"], prefix="branch_predictor")
    l1d_stride_prefetch_dummies = pd.get_dummies(dataset["stride_prefetcher_degree"], prefix="stride_prefetcher_degree")
    dataset_without_categorical_columns = dataset.drop(columns=["benchmark", "branch_predictor", "stride_prefetcher_degree"])
    return pd.concat([dataset_without_categorical_columns, 
                      benchmark_dummies, 
                      branch_predictor_dummies, 
                      l1d_stride_prefetch_dummies], axis=1)


def convert_memsizes_to_numerical(dataset: pd.DataFrame) -> pd.DataFrame:
    memsize_colummns = ["l1d_size", "l1i_size", "l2_size"]
    for col in memsize_colummns:
        dataset[col] = dataset[col].str.replace("KiB", "").str.replace("MiB", "").astype(int)
    return dataset


def main() -> None:
    args = parse_args()
    # Enforce XLA/Neuron usage
    device = 'xla'
    print(f"Using device: {device}")

    parameter_sweep = load_parameter_sweep(args.parameter_sweep)
    scaler = load_scaler(args.scaler_file)

    predict_features = one_hot_encode_categorical_columns(parameter_sweep)
    predict_features = convert_memsizes_to_numerical(predict_features)

    # Standardize only continuous columns; leave one-hot benchmark columns as 0/1
    continuous_cols = [c for c in predict_features.columns if not (c.startswith("benchmark_") or 
                                                                   c.startswith("branch_predictor_") or 
                                                                   c.startswith("stride_prefetcher_degree"))]

    predict_features[continuous_cols] = scaler.transform(predict_features[continuous_cols])
    predict_features = predict_features.astype(float)
    predict_features_t = torch.from_numpy(predict_features.to_numpy(copy=True)).float()

    predict_loader = DataLoader(TensorDataset(predict_features_t), batch_size=128, shuffle=False)

    predictions = []

    model = load_model(args.checkpoint_file, input_size=predict_features.shape[1], device=device)
    model.eval()
    
    total_start = time.perf_counter()
    with torch.no_grad():
        for inputs in predict_loader:
            input_batch = torch.stack(inputs).to(device)
            outputs = model(input_batch)
            predictions.append(outputs.detach().to("cpu").numpy().flatten())
    total_duration = time.perf_counter() - total_start

    predictions = np.concatenate(predictions, axis=0)
    np.save("predictions.npy", predictions)

    print(f"Total time taken for {predictions.shape} predictions: {total_duration} seconds")

if __name__ == "__main__":
    main()

