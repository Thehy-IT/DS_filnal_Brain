"""
evaluate.py — Full evaluation suite for BrainTumorAI.

Generates (inspired by DAKHDL's evaluation section):
  - Confusion matrix heatmap
  - Per-class classification report
  - Training/Validation loss + accuracy curves
  - ROC-AUC curves (one-vs-rest, per class)
  - Evaluation summary JSON

All figures saved to reports/figures/. Summary saved to reports/.
"""
import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe in server/notebook contexts
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from src.config import data_cfg, train_cfg
from src.utils import load_history


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _figures_dir() -> str:
    path = os.path.join(train_cfg.reports_dir, "figures")
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 1. Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    save: bool = True,
) -> np.ndarray:
    """
    Plot and optionally save a normalised confusion matrix heatmap.

    Args:
        y_true:      Ground-truth integer labels.
        y_pred:      Predicted integer labels.
        class_names: Display labels for axes.
        save:        If True, write PNG to reports/figures/.

    Returns:
        cm: Raw (unnormalised) confusion matrix as numpy array.
    """
    class_names = class_names or data_cfg.class_names
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Confusion Matrix — BrainTumorAI", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Normalised (row %)"],
        ["d", ".2%"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()

    if save:
        path = os.path.join(_figures_dir(), "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Evaluate] Confusion matrix → {path}")

    plt.close(fig)
    return cm


# ---------------------------------------------------------------------------
# 2. Training History Curves
# ---------------------------------------------------------------------------

def plot_training_history(
    history: Optional[Dict] = None,
    history_path: Optional[str] = None,
    save: bool = True,
) -> None:
    """
    Plot training & validation loss and accuracy curves.

    Mirrors DAKHDL's training history visualisation.

    Args:
        history:      Dict with keys train_loss/val_loss/train_acc/val_acc.
        history_path: If history is None, load from this JSON path.
        save:         If True, write PNG to reports/figures/.
    """
    if history is None:
        if history_path is None:
            history_path = os.path.join(train_cfg.reports_dir, "training_history.json")
        history = load_history(history_path)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — BrainTumorAI", fontsize=14, fontweight="bold")

    # --- Loss ---
    ax_loss.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    ax_loss.plot(epochs, history["val_loss"], "r-o", markersize=4, label="Val Loss")
    ax_loss.set_title("Loss per Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # --- Accuracy ---
    ax_acc.plot(epochs, history["train_acc"], "b-o", markersize=4, label="Train Acc")
    ax_acc.plot(epochs, history["val_acc"], "r-o", markersize=4, label="Val Acc")
    ax_acc.set_title("Accuracy per Epoch")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    # Mark best val_acc epoch
    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_acc = max(history["val_acc"])
    ax_acc.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.6)
    ax_acc.annotate(
        f"Best\n{best_acc:.4f}",
        xy=(best_epoch, best_acc),
        xytext=(best_epoch + 0.5, best_acc - 0.05),
        fontsize=8,
        color="green",
    )

    plt.tight_layout()

    if save:
        path = os.path.join(_figures_dir(), "training_history.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Evaluate] Training history → {path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. ROC-AUC Curves (one-vs-rest per class)
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true: List[int],
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    save: bool = True,
) -> Dict[str, float]:
    """
    Plot ROC curves and compute AUC for each class (one-vs-rest).

    Args:
        y_true:      Ground-truth integer labels.
        y_prob:      Softmax probabilities, shape (N, num_classes).
        class_names: Display names for each class.
        save:        If True, write PNG to reports/figures/.

    Returns:
        Dict mapping class name → AUC score.
    """
    class_names = class_names or data_cfg.class_names
    num_classes = len(class_names)

    # Binarise ground truth for one-vs-rest
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("ROC Curves — BrainTumorAI (One-vs-Rest)", fontsize=13, fontweight="bold")

    auc_scores: Dict[str, float] = {}
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[cls] = roc_auc
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        path = os.path.join(_figures_dir(), "roc_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Evaluate] ROC curves → {path}")

    plt.close(fig)
    return auc_scores


# ---------------------------------------------------------------------------
# 4. Macro evaluation summary
# ---------------------------------------------------------------------------

def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    save_report: bool = True,
) -> Dict:
    """
    Run full evaluation on a DataLoader and return a metrics summary dict.

    Performs:
      - Confusion matrix plot
      - Per-class classification report (precision / recall / F1)
      - ROC-AUC per class
      - JSON summary saved to reports/

    Args:
        model:       Trained PyTorch model (already on device, in eval mode).
        data_loader: DataLoader for the evaluation set.
        device:      torch.device.
        class_names: Override class list.
        save_report: If True, persist JSON and figures.

    Returns:
        Dict with keys: accuracy, macro_f1, auc_scores, report_str.
    """
    class_names = class_names or data_cfg.class_names
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)

    # --- Classification report ---
    report_str = classification_report(
        all_labels, all_preds, target_names=class_names
    )
    print("\n" + "="*60)
    print("  EVALUATION REPORT")
    print("="*60)
    print(report_str)

    # --- Confusion matrix ---
    plot_confusion_matrix(all_labels, all_preds, class_names=class_names, save=save_report)

    # --- ROC-AUC ---
    auc_scores = plot_roc_curves(
        all_labels, all_probs, class_names=class_names, save=save_report
    )
    macro_auc = float(np.mean(list(auc_scores.values())))
    print(f"Macro ROC-AUC: {macro_auc:.4f}")
    for cls, s in auc_scores.items():
        print(f"  {cls:>12}: AUC = {s:.4f}")

    # --- Accuracy ---
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    summary = {
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "macro_auc": round(macro_auc, 6),
        "per_class_auc": {k: round(v, 6) for k, v in auc_scores.items()},
        "classification_report": report_str,
    }

    if save_report:
        report_path = os.path.join(train_cfg.reports_dir, "evaluation_summary.json")
        os.makedirs(train_cfg.reports_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[Evaluate] Summary → {report_path}")

    return summary
