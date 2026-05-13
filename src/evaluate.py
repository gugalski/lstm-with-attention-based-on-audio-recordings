from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from .model import LSTMWithAttention

logger = logging.getLogger(__name__)


def evaluate(
    model: LSTMWithAttention,
    loader: DataLoader,
    device: torch.device,
    output_dir: Optional[Path] = None,
) -> dict[str, float]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[int] = []
    all_weights: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, weights = model(x)
            all_logits.append(logits.cpu())
            all_labels.extend(y.tolist())
            all_weights.extend(weights.cpu().unbind(0))

    logits_t = torch.cat(all_logits)
    probs = torch.softmax(logits_t, dim=1)[:, 1].numpy()
    preds = logits_t.argmax(dim=1).numpy()
    labels = np.array(all_labels)

    has_both_classes = len(np.unique(labels)) > 1
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "auc_roc": roc_auc_score(labels, probs) if has_both_classes else float("nan"),
        "auc_pr": average_precision_score(labels, probs) if has_both_classes else float("nan"),
        "mcc": matthews_corrcoef(labels, preds),
        "kappa": cohen_kappa_score(labels, preds),
    }

    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    report = classification_report(labels, preds, target_names=["healthy", "depression"])
    logger.info("\n%s", report)

    if output_dir is not None:
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        _plot_confusion_matrix(labels, preds, fig_dir)
        if has_both_classes:
            _plot_roc(labels, probs, fig_dir, metrics["auc_roc"])
            _plot_pr(labels, probs, fig_dir)
        _plot_attention_heatmap(all_weights, labels, preds, fig_dir)
        (fig_dir / "classification_report.txt").write_text(report)

    return metrics


def _plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, fig_dir: Path) -> None:
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
        xticklabels=["healthy", "depression"],
        yticklabels=["healthy", "depression"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalised)")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _plot_roc(labels: np.ndarray, probs: np.ndarray, fig_dir: Path, auc: float) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def _plot_pr(labels: np.ndarray, probs: np.ndarray, fig_dir: Path) -> None:
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_curve.png", dpi=150)
    plt.close(fig)


def _plot_attention_heatmap(
    weights: list[torch.Tensor],
    labels: np.ndarray,
    preds: np.ndarray,
    fig_dir: Path,
) -> None:
    correct_mask = labels == preds
    wrong_mask = ~correct_mask

    def mean_weights(mask: np.ndarray) -> Optional[np.ndarray]:
        selected = [weights[i].numpy() for i in np.where(mask)[0] if i < len(weights)]
        if not selected:
            return None
        min_len = min(w.shape[0] for w in selected)
        return np.stack([w[:min_len] for w in selected]).mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, mask, title in zip(axes, [correct_mask, wrong_mask], ["Correct", "Misclassified"]):
        avg = mean_weights(mask)
        if avg is None:
            ax.set_title(f"{title} (no samples)")
            continue
        sns.heatmap(avg[np.newaxis, :], ax=ax, cmap="viridis", cbar=True)
        ax.set_title(f"Attention — {title}")
        ax.set_xlabel("Time step")
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(fig_dir / "attention_heatmap.png", dpi=150)
    plt.close(fig)
