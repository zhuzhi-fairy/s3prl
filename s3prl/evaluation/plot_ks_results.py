#!/usr/bin/env python3
"""Keyword Spotting (KS) evaluation visualization.

Plots accuracy vs lookahead size for different context window lengths,
with one figure per model and an optional combined comparison figure.

Usage:
    python plot_ks_results.py \
        --model-names streaming_wavlm_12L streaming_wavlm_24L \
        --window-sizes 64 128 256 512 \
        --lookahead-sizes 0 1 2 3 5 10

    # Custom result directory and output format
    python plot_ks_results.py \
        --model-names streaming_wavlm_24L \
        --window-sizes 128 256 512 \
        --lookahead-sizes 0 1 3 5 \
        --result-dir result/downstream/ks \
        --format pdf
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Color palette (colorblind-friendly, from Okabe-Ito)
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def parse_accuracy(log_file: str) -> float:
    """Extract test accuracy from a log file.

    Expects the accuracy value to be the last token on the second-to-last line.
    """
    with open(log_file, "r") as f:
        lines = f.readlines()
    return float(lines[-1].split()[-1]) * 100  # Convert to percentage


def collect_results(
    model_name: str,
    window_sizes: list,
    lookahead_sizes: list,
    result_dir: str,
) -> np.ndarray:
    """Collect accuracy results into a 2D array [window_sizes x lookahead_sizes].

    Missing results are stored as NaN.
    """
    results = np.full((len(window_sizes), len(lookahead_sizes)), np.nan)
    for i, ws in enumerate(window_sizes):
        for j, la in enumerate(lookahead_sizes):
            log_file = os.path.join(
                result_dir, f"{model_name}_ws{ws}_la{la}", "log.log"
            )
            if not os.path.isfile(log_file):
                print(f"  Warning: missing {log_file}", file=sys.stderr)
                continue
            try:
                results[i, j] = parse_accuracy(log_file)
            except (IndexError, ValueError) as e:
                print(f"  Warning: failed to parse {log_file}: {e}", file=sys.stderr)
    return results


def plot_model_results(
    model_name: str,
    results: np.ndarray,
    window_sizes: list,
    lookahead_sizes: list,
    output_path: str,
) -> None:
    """Plot accuracy vs lookahead for a single model, one line per window size."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, ws in enumerate(window_sizes):
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        valid = ~np.isnan(results[i])
        la_arr = np.array(lookahead_sizes)

        ax.plot(
            la_arr[valid],
            results[i][valid],
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2,
            label=f"Context {ws}",
        )
        # Annotate each point with its accuracy value
        for x, y in zip(la_arr[valid], results[i][valid]):
            ax.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                fontsize=7,
                ha="center",
                color=color,
            )

    ax.set_xlabel("Lookahead Size (frames)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"KS Accuracy — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xticks(lookahead_sizes)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=10, title="Context Length", title_fontsize=10)

    # Ensure y-axis has some padding
    ymin, ymax = ax.get_ylim()
    margin = (ymax - ymin) * 0.08
    ax.set_ylim(ymin - margin, ymax + margin)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_combined(
    all_model_results: dict,
    window_sizes: list,
    lookahead_sizes: list,
    output_path: str,
) -> None:
    """Plot a combined figure comparing all models at the largest window size."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ws_idx = -1  # Use the largest window size

    for i, (model_name, results) in enumerate(all_model_results.items()):
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        valid = ~np.isnan(results[ws_idx])
        la_arr = np.array(lookahead_sizes)

        ax.plot(
            la_arr[valid],
            results[ws_idx][valid],
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2,
            label=model_name,
        )

    ax.set_xlabel("Lookahead Size (frames)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"KS Model Comparison (Context {window_sizes[-1]})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(lookahead_sizes)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot KS accuracy vs lookahead size for streaming models."
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        required=True,
        help="List of model names corresponding to the results.",
    )
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        required=True,
        help="List of window sizes to evaluate.",
    )
    parser.add_argument(
        "--lookahead-sizes",
        type=int,
        nargs="+",
        required=True,
        help="List of lookahead sizes to evaluate.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="result/downstream/ks",
        help="Root directory containing experiment results (default: result/downstream/ks).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output figure format (default: png).",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    all_model_results = {}

    for model_name in args.model_names:
        print(f"Collecting results for {model_name}...")
        results = collect_results(
            model_name, args.window_sizes, args.lookahead_sizes, args.result_dir
        )
        all_model_results[model_name] = results

        out = os.path.join(args.result_dir, f"{model_name}.{args.format}")
        plot_model_results(
            model_name, results, args.window_sizes, args.lookahead_sizes, out
        )

    # Combined comparison when there are multiple models
    if len(args.model_names) > 1:
        out = os.path.join(args.result_dir, f"comparison.{args.format}")
        plot_combined(all_model_results, args.window_sizes, args.lookahead_sizes, out)


if __name__ == "__main__":
    main()
