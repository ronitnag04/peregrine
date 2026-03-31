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


def load_model(checkpoint_path: Path, device: torch.device, **model_kwargs) -> PeregrineMLModel:
    model = PeregrineMLModel(**model_kwargs).to(device)
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


def pre_process_features(features: pd.DataFrame) -> pd.DataFrame:
    stride_prefetch_dummies = pd.get_dummies(features["stride_prefetcher_degree"], prefix="stride_prefetcher_degree")
    features = features.drop(columns=["benchmark", "stride_prefetcher_degree"])
    features = pd.concat([features, stride_prefetch_dummies], axis=1)

    memsize_colummns = ["l1d_size", "l1i_size", "l2_size"]
    for col in memsize_colummns:
        features[col] = features[col].apply(lambda x: 
            int(x.replace("KiB", "")) * 1024 if "KiB" in x 
            else int(x.replace("MiB", "")) * 1024**2 if "MiB" in x 
            else int(x))

    return features


def main() -> None:
    args = parse_args()
    # Enforce XLA/Neuron usage
    device = 'xla'
    print(f"Using device: {device}")

    parameter_sweep = load_parameter_sweep(args.parameter_sweep)
    scaler = load_scaler(args.scaler_file)

    # Pre-process features
    predict_features = pre_process_features(parameter_sweep)

    # Use loaded scaler to transform features
    predict_features = predict_features.copy().astype(float)
    predict_features[predict_features.columns] = scaler.transform(predict_features[predict_features.columns])
    
    # Split into program and configuration features
    prog_feature_columns = [col for col in predict_features.columns if col.startswith("prog_") or col.startswith("cache_")]
    predict_prog_features = predict_features[prog_feature_columns]
    predict_config_features = predict_features.drop(columns=prog_feature_columns)
    num_prog_features = len(predict_prog_features.columns)
    num_config_features = len(predict_config_features.columns)

    # Configure TensorDatasets for prediction inputs to model
    predict_prog_features_t = torch.from_numpy(predict_prog_features.to_numpy(copy=True)).float()
    predict_config_features_t = torch.from_numpy(predict_config_features.to_numpy(copy=True)).float()
    predict_ds = TensorDataset(predict_prog_features_t, predict_config_features_t)
    predict_loader = DataLoader(predict_ds, batch_size=128, shuffle=False)


    predictions = []
    model = load_model(args.checkpoint_file, device=device, 
        prog_size=num_prog_features,          
        config_size=num_config_features
    )
    model.eval()
    
    total_start = time.perf_counter()
    with torch.no_grad():
        for prog_inputs, config_inputs in predict_loader:
            prog_inputs = prog_inputs.to(device)
            config_inputs = config_inputs.to(device)
            outputs = model(prog_inputs, config_inputs)
            predictions.append(outputs.detach().to("cpu").numpy().flatten())
    total_duration = time.perf_counter() - total_start

    predictions = np.concatenate(predictions, axis=0)
    np.save("predictions.npy", predictions)

    print(f"Total time taken for {predictions.shape} predictions: {total_duration} seconds")

if __name__ == "__main__":
    main()

