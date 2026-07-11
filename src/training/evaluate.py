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
matplotlib.use("Agg")  # Backend không cần màn hình — an toàn trong server/notebook
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
# Helper nội bộ — trả về đường dẫn thư mục lưu hình ảnh, tự tạo nếu chưa có
# ---------------------------------------------------------------------------

def _figures_dir() -> str:
    path = os.path.join(train_cfg.reports_dir, "figures")
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 1. Confusion Matrix — ma trận nhầm lẫn, hiển thị số lần đúng/sai mỗi lớp
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    save: bool = True,
) -> np.ndarray:
    """
    Vẽ và lưu confusion matrix dạng heatmap (cả số đếm lẫn tỷ lệ phần trăm).

    Args:
        y_true:      Nhãn thực tế (ground truth).
        y_pred:      Nhãn dự đoán của model.
        class_names: Tên các lớp để hiển thị trên trục.
        save:        Nếu True, lưu file PNG vào reports/figures/.

    Returns:
        cm: Ma trận nhầm lẫn thô (chưa normalize) dạng numpy array.
    """
    class_names = class_names or data_cfg.class_names
    # Tính confusion matrix thô (số lượng)
    cm = confusion_matrix(y_true, y_pred)
    # Normalize theo hàng: mỗi ô = tỷ lệ % so với tổng lớp thực tế
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    # Vẽ 2 subplot: bên trái số đếm, bên phải tỷ lệ %
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
            annot=True,      # Hiển thị số trong từng ô
            fmt=fmt,
            cmap="Blues",    # Màu xanh đậm = giá trị cao
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
        print(f"[Evaluate] Confusion matrix -> {path}")

    plt.close(fig)
    return cm


# ---------------------------------------------------------------------------
# 2. Training History Curves — biểu đồ loss và accuracy qua từng epoch
# ---------------------------------------------------------------------------

def plot_training_history(
    history: Optional[Dict] = None,
    history_path: Optional[str] = None,
    save: bool = True,
) -> None:
    """
    Vẽ đường cong loss và accuracy của tập train và validation theo epoch.

    Args:
        history:      Dict chứa train_loss/val_loss/train_acc/val_acc.
        history_path: Nếu history=None, tải từ file JSON này.
        save:         Nếu True, lưu PNG vào reports/figures/.
    """
    # Cho phép truyền dict trực tiếp hoặc đọc từ file JSON
    if history is None:
        if history_path is None:
            history_path = os.path.join(train_cfg.reports_dir, "training_history.json")
        history = load_history(history_path)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — BrainTumorAI", fontsize=14, fontweight="bold")

    # Biểu đồ Loss: xanh = train, đỏ = validation
    ax_loss.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    ax_loss.plot(epochs, history["val_loss"], "r-o", markersize=4, label="Val Loss")
    ax_loss.set_title("Loss per Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Biểu đồ Accuracy: xanh = train, đỏ = validation
    ax_acc.plot(epochs, history["train_acc"], "b-o", markersize=4, label="Train Acc")
    ax_acc.plot(epochs, history["val_acc"], "r-o", markersize=4, label="Val Acc")
    ax_acc.set_title("Accuracy per Epoch")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    # Đánh dấu epoch có val_acc tốt nhất bằng đường thẳng xanh lá
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
        print(f"[Evaluate] Training history -> {path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. ROC-AUC Curves — đường cong ROC cho từng lớp theo chiến lược one-vs-rest
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true: List[int],
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    save: bool = True,
) -> Dict[str, float]:
    """
    Vẽ đường cong ROC và tính AUC cho từng lớp (one-vs-rest).

    Args:
        y_true:      Nhãn thực tế.
        y_prob:      Xác suất softmax, shape (N, num_classes).
        class_names: Tên các lớp.
        save:        Nếu True, lưu PNG.

    Returns:
        Dict ánh xạ tên lớp -> điểm AUC.
    """
    class_names = class_names or data_cfg.class_names
    num_classes = len(class_names)

    # Chuyển nhãn sang dạng one-hot để tính ROC one-vs-rest
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("ROC Curves - BrainTumorAI (One-vs-Rest)", fontsize=13, fontweight="bold")

    auc_scores: Dict[str, float] = {}
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        # Tính FPR/TPR cho từng lớp
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[cls] = roc_auc
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls}  (AUC = {roc_auc:.3f})")

    # Đường chéo = random classifier (AUC = 0.5), dùng làm baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        path = os.path.join(_figures_dir(), "roc_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Evaluate] ROC curves -> {path}")

    plt.close(fig)
    return auc_scores


# ---------------------------------------------------------------------------
# 4. evaluate_model — hàm đánh giá tổng hợp, gọi tất cả các bước trên
# ---------------------------------------------------------------------------

def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    save_report: bool = True,
) -> Dict:
    """
    Chạy đánh giá đầy đủ trên DataLoader và trả về dict tổng hợp các chỉ số.

    Bao gồm:
      - Confusion matrix
      - Classification report (precision / recall / F1)
      - ROC-AUC từng lớp
      - Lưu JSON tóm tắt vào reports/

    Args:
        model:       Model đã huấn luyện (đang ở eval mode, đúng device).
        data_loader: DataLoader của tập đánh giá.
        device:      torch.device.
        class_names: Override danh sách lớp.
        save_report: Nếu True, lưu JSON và các hình ảnh.

    Returns:
        Dict với các key: accuracy, macro_f1, auc_scores, report_str.
    """
    class_names = class_names or data_cfg.class_names
    model.eval()  # Tắt dropout và batch norm training mode

    all_preds, all_labels, all_probs = [], [], []

    # Chạy inference trên toàn bộ tập đánh giá, không tính gradient
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            # Chuyển logits sang xác suất bằng softmax
            probs = torch.softmax(outputs, dim=1)
            # Lấy class có xác suất cao nhất
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Chuyển danh sách xác suất sang numpy array 2D (N, num_classes)
    all_probs = np.array(all_probs)

    # In classification report: precision, recall, F1 cho từng lớp
    report_str = classification_report(
        all_labels, all_preds, target_names=class_names
    )
    print("\n" + "="*60)
    print("  EVALUATION REPORT")
    print("="*60)
    print(report_str)

    # Vẽ confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names=class_names, save=save_report)

    # Vẽ và tính ROC-AUC từng lớp
    auc_scores = plot_roc_curves(
        all_labels, all_probs, class_names=class_names, save=save_report
    )
    # Macro AUC = trung bình AUC của 4 lớp
    macro_auc = float(np.mean(list(auc_scores.values())))
    print(f"Macro ROC-AUC: {macro_auc:.4f}")
    for cls, s in auc_scores.items():
        print(f"  {cls:>12}: AUC = {s:.4f}")

    # Tính accuracy tổng và macro F1
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    # Đóng gói tất cả chỉ số vào dict
    summary = {
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "macro_auc": round(macro_auc, 6),
        "per_class_auc": {k: round(v, 6) for k, v in auc_scores.items()},
        "classification_report": report_str,
    }

    # Lưu tóm tắt ra JSON để tham khảo sau
    if save_report:
        report_path = os.path.join(train_cfg.reports_dir, "evaluation_summary.json")
        os.makedirs(train_cfg.reports_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[Evaluate] Summary -> {report_path}")

    return summary
