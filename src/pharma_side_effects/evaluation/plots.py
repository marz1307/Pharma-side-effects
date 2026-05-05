"""Plotting helpers for class balance and confusion matrices."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def plot_class_distribution(
    labels: pd.Series,
    title: str = "Side Effect Class Distribution",
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot a horizontal bar chart of class frequencies.

    Args:
        labels: Series of class labels.
        title: Plot title.
        output_path: Optional path to save the figure as PNG.
        figsize: Figure size in inches.

    Returns:
        The created matplotlib :class:`Figure`.
    """
    counts = labels.value_counts()
    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("hsv", len(counts))
    sns.barplot(
        x=counts.values,
        y=counts.index,
        hue=counts.index,
        palette=palette,
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Number of Records")
    ax.set_ylabel("Side Effect Category")
    sns.despine(left=True, bottom=True)
    for i, count in enumerate(counts.values):
        ax.text(count + max(counts.values) * 0.01, i, str(count), va="center", fontsize=10)
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 10),
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Args:
        matrix: Square confusion matrix.
        labels: Class label order matching ``matrix``.
        title: Plot title.
        output_path: Optional path to save the figure as PNG.
        figsize: Figure size in inches.

    Returns:
        The created matplotlib :class:`Figure`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
