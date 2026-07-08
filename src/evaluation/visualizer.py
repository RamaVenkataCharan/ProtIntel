"""Visualization utilities for evaluation results.

Generates confusion matrices, ROC curves, PR curves, per-class
accuracy bar charts, and training history plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns

from src.utils.io_utils import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Color palette matching the ProtIntel design
COLORS = {
    "helix": "#FF6B6B",
    "sheet": "#4ECDC4",
    "coil": "#95A5A6",
    "primary": "#00D4FF",
    "secondary": "#7C3AED",
    "background": "#0A0E1A",
}

Q3_CLASSES = ["Helix (H)", "Sheet (E)", "Coil (C)"]
Q8_CLASSES = ["H (α-helix)", "E (β-strand)", "G (3₁₀-helix)", "I (π-helix)",
              "B (β-bridge)", "T (turn)", "S (bend)", "C (coil)"]


class Visualizer:
    """Generates plots and visualizations for evaluation results.

    All plots can be saved as PNG, SVG, or interactive HTML (Plotly).

    Args:
        output_dir: Directory to save generated plots.
    """

    def __init__(self, output_dir: str | Path = "logs/plots") -> None:
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

    def plot_confusion_matrix(
        self,
        matrix: list[list[float]],
        class_names: list[str],
        title: str = "Confusion Matrix",
        filename: str = "confusion_matrix",
    ) -> Path:
        """Plot a normalized confusion matrix heatmap.

        Args:
            matrix: 2D list of normalized confusion matrix values.
            class_names: List of class labels.
            title: Plot title.
            filename: Output filename (without extension).

        Returns:
            Path to the saved plot.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            np.array(matrix),
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={"label": "Proportion"},
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        output_path = self.output_dir / f"{filename}.png"
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved confusion matrix: {output_path}")
        return output_path

    def plot_per_class_accuracy(
        self,
        results: dict[str, Any],
        task: str = "q3",
        filename: str = "per_class_accuracy",
    ) -> Path:
        """Plot per-class accuracy as a bar chart.

        Args:
            results: Evaluation results dictionary.
            task: 'q3' or 'q8'.
            filename: Output filename.

        Returns:
            Path to the saved plot.
        """
        if task == "q3":
            classes = ["H", "E", "C"]
            colors = [COLORS["helix"], COLORS["sheet"], COLORS["coil"]]
        else:
            classes = ["H", "E", "G", "I", "B", "T", "S", "C"]
            colors = plt.cm.Set3(np.linspace(0, 1, 8))

        accuracies = [
            results.get(f"{task}_accuracy_{cls}", 0.0) for cls in classes
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(classes, accuracies, color=colors, edgecolor="white", linewidth=1.5)

        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center", va="bottom", fontweight="bold",
            )

        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"{task.upper()} Per-Class Accuracy", fontsize=14, fontweight="bold")
        ax.axhline(y=results.get(f"{task}_accuracy", 0), color="gray",
                   linestyle="--", alpha=0.7, label="Overall")
        ax.legend()
        plt.tight_layout()

        output_path = self.output_dir / f"{filename}_{task}.png"
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)

        logger.info(f"Saved per-class accuracy: {output_path}")
        return output_path

    def plot_training_history(
        self,
        history: dict[str, list[float]],
        filename: str = "training_history",
    ) -> Path:
        """Plot training and validation loss/accuracy curves.

        Args:
            history: Training history dictionary with 'train_loss',
                'val_loss', 'train_q3_accuracy', 'val_q3_accuracy'.
            filename: Output filename.

        Returns:
            Path to the saved plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history.get("train_loss", [])) + 1)

        # Loss plot
        if "train_loss" in history:
            axes[0].plot(epochs, history["train_loss"], label="Train Loss",
                        color=COLORS["primary"], linewidth=2)
        if "val_loss" in history:
            axes[0].plot(epochs, history["val_loss"], label="Val Loss",
                        color=COLORS["secondary"], linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss", fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy plot
        if "train_q3_accuracy" in history:
            axes[1].plot(epochs, history["train_q3_accuracy"], label="Train Q3",
                        color=COLORS["primary"], linewidth=2)
        if "val_q3_accuracy" in history:
            axes[1].plot(epochs, history["val_q3_accuracy"], label="Val Q3",
                        color=COLORS["secondary"], linewidth=2)
        if "val_q8_accuracy" in history:
            axes[1].plot(epochs, history["val_q8_accuracy"], label="Val Q8",
                        color=COLORS["helix"], linewidth=2, linestyle="--")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training & Validation Accuracy", fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / f"{filename}.png"
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)

        logger.info(f"Saved training history: {output_path}")
        return output_path

    def plot_residue_confidence_interactive(
        self,
        sequence: str,
        confidence: list[float],
        q3_preds: list[str],
        filename: str = "confidence_plot",
    ) -> Path:
        """Create an interactive Plotly bar chart of per-residue confidence.

        Args:
            sequence: Amino acid sequence string.
            confidence: List of confidence values per residue.
            q3_preds: List of Q3 predictions per residue.
            filename: Output filename.

        Returns:
            Path to the saved HTML file.
        """
        color_map = {"H": COLORS["helix"], "E": COLORS["sheet"], "C": COLORS["coil"]}
        colors = [color_map.get(p, "#666666") for p in q3_preds]

        fig = go.Figure(data=[
            go.Bar(
                x=list(range(len(sequence))),
                y=confidence,
                marker_color=colors,
                text=[f"{aa}<br>{pred}<br>{conf:.3f}"
                      for aa, pred, conf in zip(sequence, q3_preds, confidence)],
                hoverinfo="text",
            )
        ])

        fig.update_layout(
            title="Per-Residue Prediction Confidence",
            xaxis_title="Residue Position",
            yaxis_title="Confidence",
            yaxis_range=[0, 1],
            template="plotly_dark",
            paper_bgcolor=COLORS["background"],
            plot_bgcolor="#1a1f2e",
        )

        output_path = self.output_dir / f"{filename}.html"
        fig.write_html(str(output_path))

        logger.info(f"Saved interactive confidence plot: {output_path}")
        return output_path
