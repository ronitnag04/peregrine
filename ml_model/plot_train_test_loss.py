"""Plot train/eval loss curves"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt


def load_losses(path: Path) -> Tuple[list[int], list[float], list[float]]:
    """Read epochs, train_loss, eval_loss lists from a JSON metrics file."""
    with path.open() as f:
        data = json.load(f)

    epochs: list[int] = []
    train_losses: list[float] = []
    eval_losses: list[float] = []
    percent_errors: list[float] = []

    for i, entry in enumerate(data["epochs"]):
        epochs.append(i)
        train_losses.append(float(entry["train_loss"]))
        eval_losses.append(float(entry["eval_loss"]))
        percent_errors.append(float(entry["percent_error"]))

    return epochs, train_losses, eval_losses, percent_errors


def plot_losses(
    epochs: Iterable[int],
    train_losses: Iterable[float],
    eval_losses: Iterable[float],
    title: str,
    output_path: Path,
) -> None:
    """Generate and save a loss plot."""
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train loss", marker="o")
    plt.plot(epochs, eval_losses, label="Eval loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_percent_errors(
    epochs: Iterable[int],
    percent_errors: Iterable[float],
    title: str,
    output_path: Path,
) -> None:
    """Generate and save a percent error plot."""
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, percent_errors, label="Percent error", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Percent error")
    plt.ylim(0, 100)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training/eval losses and percent error from a metrics JSON file.",
    )
    parser.add_argument(
        "-t", 
        "--training-metrics",   
        type=Path,
        help="Path to the training_metrics JSON file (e.g. training_metrics/collatz-pin_metrics.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    path: Path = args.training_metrics

    dataset_name = os.path.basename(path).split(".")[0]

    plt.switch_backend("Agg")

    epochs, train, eval, percent_errors = load_losses(path)

    plot_losses(
        epochs,
        train,
        eval,
        title="Train vs Eval Loss",
        output_path=plots_dir / f"{dataset_name}_losses.png",
    )

    plot_percent_errors(
        epochs,
        percent_errors,
        title="Percent Error",
        output_path=plots_dir / f"{dataset_name}_percent_errors.png",
    )

    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
