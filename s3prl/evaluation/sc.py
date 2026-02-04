#!/usr/bin/env python3
"""Speech Commands evaluation script.

Plots training curves and confusion matrix from experiment results.

Usage:
    python sc_evaluation.py <run_directory>

Example:
    python sc_evaluation.py result/downstream/streaming_wavlm_24to24_200ms_sc
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_logs(log_dir: Path) -> dict:
    """Load tensorboard event files and extract scalars."""
    # Find event files
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No tensorboard event files found in {log_dir}")

    # Use the most recent event file if multiple exist
    event_file = sorted(event_files)[-1]
    print(f"Loading: {event_file}")

    # Load event accumulator
    ea = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalars
        },
    )
    ea.Reload()

    # Get available tags
    tags = ea.Tags().get("scalars", [])
    print(f"Available tags: {tags}")

    # Extract data for each tag
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {"steps": steps, "values": values}

    return data


def load_predictions(run_dir: Path):
    """Load ground truth and predictions from text files.

    Args:
        run_dir: Path to run directory containing test_truth.txt and test_predict.txt

    Returns:
        y_true: List of ground truth labels
        y_pred: List of predicted labels
    """
    truth_file = run_dir / "test_truth.txt"
    pred_file = run_dir / "test_predict.txt"

    if not truth_file.exists():
        raise FileNotFoundError(f"Truth file not found: {truth_file}")
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    # Load truth labels
    y_true = []
    with open(truth_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                y_true.append(parts[-1])  # Label is the last column

    # Load predicted labels
    y_pred = []
    with open(pred_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                y_pred.append(parts[-1])  # Prediction is the last column

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Mismatch: {len(y_true)} truth labels vs {len(y_pred)} predictions"
        )

    print(f"Loaded {len(y_true)} samples")
    return y_true, y_pred


def plot_curves(data: dict, output_path: Path) -> None:
    """Plot loss and accuracy curves with dual y-axes."""
    fig, ax1 = plt.subplots()

    # Colors for different curves
    colors = {
        "train_loss": "blue",
        "train_acc": "red",
        "dev_loss": "blue",
        "dev_acc": "red",
    }

    # Line styles
    linestyles = {
        "train_loss": "-",
        "train_acc": "-",
        "dev_loss": "--",
        "dev_acc": "--",
    }

    # Normalize tag names - extract the metric part after the last '/'
    normalized_data = {}
    for tag, values in data.items():
        metric_name = tag.split("/")[-1]

        if "train" in metric_name.lower() and "loss" in metric_name.lower():
            normalized_data["train_loss"] = values
        elif "dev" in metric_name.lower() and "loss" in metric_name.lower():
            normalized_data["dev_loss"] = values
        elif "valid" in metric_name.lower() and "loss" in metric_name.lower():
            normalized_data["dev_loss"] = values
        elif "train" in metric_name.lower() and "acc" in metric_name.lower():
            normalized_data["train_acc"] = values
        elif "dev" in metric_name.lower() and "acc" in metric_name.lower():
            normalized_data["dev_acc"] = values
        elif "valid" in metric_name.lower() and "acc" in metric_name.lower():
            normalized_data["dev_acc"] = values

    print(f"Normalized tags: {list(normalized_data.keys())}")

    # Left y-axis: Loss
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Loss", color="blue", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="blue")

    loss_lines = []
    loss_labels = []

    for tag in ["train_loss", "dev_loss"]:
        if tag in normalized_data:
            steps = normalized_data[tag]["steps"]
            values = normalized_data[tag]["values"]
            (line,) = ax1.plot(
                steps,
                values,
                color=colors[tag],
                linestyle=linestyles[tag],
                linewidth=1.5,
                alpha=0.8,
            )
            loss_lines.append(line)
            label = "Train Loss" if tag == "train_loss" else "Dev Loss"
            loss_labels.append(label)
            print(f"  Plotted {tag}: {len(steps)} points")

    # Right y-axis: Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="red", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="red")

    acc_lines = []
    acc_labels = []

    for tag in ["train_acc", "dev_acc"]:
        if tag in normalized_data:
            steps = normalized_data[tag]["steps"]
            values = normalized_data[tag]["values"]
            (line,) = ax2.plot(
                steps,
                values,
                color=colors[tag],
                linestyle=linestyles[tag],
                linewidth=1.5,
                alpha=0.8,
            )
            acc_lines.append(line)
            label = "Train Acc" if tag == "train_acc" else "Dev Acc"
            acc_labels.append(label)
            print(f"  Plotted {tag}: {len(steps)} points")

    # Combine legends
    lines = loss_lines + acc_lines
    labels = loss_labels + acc_labels
    ax1.legend(lines, labels, loc="center right", fontsize=10)

    # Title
    run_name = output_path.parent.name
    plt.title(f"Training Curves: {run_name}", fontsize=14)

    # Grid
    ax1.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")

    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    """Plot and save confusion matrix.

    Args:
        y_true: List of ground truth labels
        y_pred: List of predicted labels
        output_path: Path to save the figure
    """
    # Get unique labels and sort them
    labels = sorted(list(set(y_true) | set(y_pred)))
    print(f"Number of classes: {len(labels)}")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Calculate accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"Test Accuracy: {accuracy:.4f} ({np.trace(cm)}/{np.sum(cm)})")

    # Create figure
    fig_size = max(10, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Plot confusion matrix (without colorbar)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45, colorbar=False)

    # Title with accuracy
    run_name = output_path.parent.name
    plt.title(f"Confusion Matrix: {run_name}\nAccuracy: {accuracy:.4f}", fontsize=14)

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Speech Commands evaluation: plot training curves and confusion matrix"
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to the run directory containing tensorboard logs and prediction files",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}")
        sys.exit(1)

    print(f"Run directory: {run_dir}")
    print("=" * 60)

    # 1. Plot training curves
    print("\n[1] Plotting training curves...")
    try:
        data = load_tensorboard_logs(run_dir)
        if data:
            output_path = run_dir / "training_curves.png"
            plot_curves(data, output_path)
        else:
            print("Warning: No data found in tensorboard logs")
    except FileNotFoundError as e:
        print(f"Warning: {e}")

    # 2. Plot confusion matrix
    print("\n[2] Plotting confusion matrix...")
    try:
        y_true, y_pred = load_predictions(run_dir)
        output_path = run_dir / "confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, output_path)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
