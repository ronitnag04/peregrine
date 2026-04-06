"""
Generate predictions for every parameter combination using trained checkpoints.

This script:
- Loads model checkpoint from the local `checkpoints/` dir
- Reads a training_data.csv file containing the parameter combinations to predict on and the measured CPI
- Adds a "Predicted CPI" column to the dataframe with the predictions from the model
- Saves the dataframe to a new file called training_data_with_predictions.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
import os
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
from train import pre_process_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Peregrine ML model predictions over parameter sweep.")
    parser.add_argument(
        "-d",
        "--training-data",
        type=str,
        default="training_data.csv",
        help="Path to training data.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory containing checkpoint and metrics files",
        default=".",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device, **model_kwargs) -> PeregrineMLModel:
    model = PeregrineMLModel(**model_kwargs).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def load_scaler(scaler_path: Path) -> StandardScaler:
    return joblib.load(scaler_path)


def load_training_data(training_data_path: Path) -> pd.DataFrame:
    return pd.read_csv(training_data_path)


def main() -> None:
    args = parse_args()
    # Enforce XLA/Neuron usage
    device = 'xla'
    print(f"Using device: {device}")

    checkpoint_path = Path(args.output_dir) / "checkpoint.pt"
    scaler_path = Path(args.output_dir) / "scaler.joblib"

    training_data = load_training_data(args.training_data)
    scaler = load_scaler(scaler_path)

    # Pre-process features (same as train.py - drops cpi, benchmark, stride_prefetcher_degree)
    predict_features = pre_process_features(training_data)

    # Use loaded scaler to transform features (same standardization as train.py)
    predict_features = predict_features.copy().astype(float)
    predict_features[predict_features.columns] = scaler.transform(predict_features[predict_features.columns])
    
    # Convert to tensor (single input like train.py, not split into prog/config)
    predict_features_t = torch.from_numpy(predict_features.to_numpy(copy=True)).float()
    predict_ds = TensorDataset(predict_features_t)
    predict_loader = DataLoader(predict_ds, batch_size=128, shuffle=False)

    # Load model with same structure as train.py
    model = load_model(checkpoint_path, device=device, 
        input_size=predict_features.shape[1],          
        hidden_dims=[256, 128],
        output_size=1
    )
    model.eval()
    
    predictions = []
    total_start = time.perf_counter()
    with torch.no_grad():
        for inputs, in predict_loader:  # Single input tensor
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.detach().to("cpu").numpy().flatten())
    total_duration = time.perf_counter() - total_start

    predictions = np.concatenate(predictions, axis=0)
    
    # Add predictions as a new column to the original dataframe
    training_data["Predicted CPI"] = predictions
    
    # Save the dataframe with predictions to a new CSV file
    output_path = os.path.join(args.output_dir, "training_data_with_predictions.csv")
    training_data.to_csv(output_path, index=False)
    
    print(f"Total time taken for {predictions.shape[0]} predictions: {total_duration} seconds")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()

