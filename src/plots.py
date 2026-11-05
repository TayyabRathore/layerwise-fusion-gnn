#------------------------------------------------------------------------------#
# plots.py -- figures for the report, saved at high resolution.                 #
# Figures are one IEEE column wide (3.5 in) so text stays readable in the PDF.  #
# Author: Tayyab Ur Rehman Rathore                                              #
#------------------------------------------------------------------------------#
"""Plotting utilities."""
import os

import matplotlib
matplotlib.use("Agg")                                     # works without a display
import matplotlib.pyplot as plt
import numpy as np

COL_W = 3.5                                               # inches, one IEEE column
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})


def _save(fig, path: str, dpi: int = 600) -> None:
    """Save a figure as PNG (600 dpi, as the course guideline recommends) and PDF."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", path + ".png / .pdf")


def plot_training_curves(histories: dict, title: str, path: str) -> None:
    """Training loss (top) and validation accuracy (bottom) per epoch.

    Args:
        histories: {model name: list of per-seed histories}, each history a list
            of dicts with ``loss`` and ``val_acc``. Seeds are averaged.
        title: figure title (e.g. dataset and feature).
        path: output path without extension.
    """
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 4.4), sharex=True)
    for name, runs in histories.items():
        n = min(len(h) for h in runs)
        loss = np.mean([[e["loss"] for e in h[:n]] for h in runs], axis=0)
        val = np.mean([[e["val_acc"] for e in h[:n]] for h in runs], axis=0)
        axes[0].plot(range(1, n + 1), loss, marker="o", ms=2, label=name)
        axes[1].plot(range(1, n + 1), 100 * val, marker="o", ms=2, label=name)
    axes[0].set(title=f"{title}: training loss", ylabel="Cross-entropy loss")
    axes[1].set(title=f"{title}: validation accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_model_comparison(summary, metric: str, path: str) -> None:
    """Bar chart (mean +/- std over seeds) of one metric for every model.

    Args:
        summary: pandas DataFrame with columns ``model``, ``<metric>_mean``, ``<metric>_std``.
        metric: e.g. ``"accuracy"`` or ``"macro_f1"``.
        path: output path without extension.
    """
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    x = np.arange(len(summary))
    ax.bar(x, 100 * summary[f"{metric}_mean"], yerr=100 * summary[f"{metric}_std"], capsize=3, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=30, ha="right")
    ax.set_ylim(0, 100)                                   # bars start at zero
    ax.set(ylabel=f"Test {metric.replace('_', '-')} (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, path)
