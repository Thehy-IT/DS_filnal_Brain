"""
train.py — Full training pipeline for BrainTumorAI.

Improvements over v1 (all learned from DAKHDL's 02_HuanLuyen_Model.ipynb):
  - set_seed() for full reproducibility
  - Stratified train/val split (fixes the transform-contamination bug)
  - class_weight passed to CrossEntropyLoss
  - 2-phase fine-tuning: freeze backbone -> then unfreeze all layers
  - EarlyStopping (patience from config)
  - ReduceLROnPlateau scheduler (per-epoch)
  - Training history saved as JSON for later plotting
  - Verbose per-epoch logging with loss + accuracy
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from src.config import data_cfg, train_cfg, SEED
from src.utils import set_seed, get_device, save_history
from src.preprocessing.dataset import split_dataset
from src.preprocessing.transforms import get_train_transforms, get_valid_transforms
from src.training.models import get_model


# ---------------------------------------------------------------------------
# EarlyStopping — dừng huấn luyện sớm khi model không còn cải thiện
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Dừng training khi val_acc không cải thiện sau `patience` epochs liên tiếp.
    Tương tự EarlyStopping(monitor='val_accuracy', restore_best_weights=True) của Keras.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience         # Số epochs chờ trước khi dừng
        self.min_delta = min_delta       # Ngưỡng cải thiện tối thiểu (bỏ qua thay đổi quá nhỏ)
        self.best_val_acc: float = 0.0   # Accuracy tốt nhất ghi nhận được
        self.counter: int = 0            # Đếm số epochs không cải thiện
        self.best_state: dict = {}       # Snapshot trọng số của epoch tốt nhất

    def step(self, val_acc: float, model: nn.Module) -> bool:
        """
        Gọi mỗi epoch. Trả về True nếu cần dừng training.
        Tự động lưu snapshot trọng số mỗi khi tìm thấy kỷ lục mới.
        """
        if val_acc > self.best_val_acc + self.min_delta:
            # Tìm được kết quả tốt hơn — reset bộ đếm và lưu trọng số
            self.best_val_acc = val_acc
            self.counter = 0
            # Clone toàn bộ state_dict (chi phí thấp, EfficientNetB0 chỉ ~5 MB)
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            # Không cải thiện — tăng bộ đếm
            self.counter += 1

        if self.counter >= self.patience:
            print(
                f"\n[EarlyStopping] No improvement for {self.patience} epochs. "
                f"Best val_acc = {self.best_val_acc:.4f}"
            )
            return True  # Ra hiệu dừng training
        return False

    def restore_best_weights(self, model: nn.Module) -> None:
        """Phục hồi trọng số của epoch tốt nhất vào model hiện tại."""
        if self.best_state:
            model.load_state_dict(self.best_state)
            print(f"[EarlyStopping] Restored best weights (val_acc={self.best_val_acc:.4f})")


# ---------------------------------------------------------------------------
# _run_epoch — chạy một epoch (train hoặc eval), tái sử dụng cho cả 2 phase
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    is_train: bool,
    scaler: torch.cuda.amp.GradScaler = None,
) -> tuple:
    """Chạy một epoch (train hoặc eval). Trả về (avg_loss, accuracy, preds, labels)."""
    # Bật/tắt dropout và batch norm tùy theo chế độ
    model.train(is_train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    # Bật gradient khi train, tắt khi eval để tiết kiệm bộ nhớ
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    desc = "Train" if is_train else "Val"

    with ctx:
        for images, labels in tqdm(loader, desc=desc, leave=False):
            # Chuyển dữ liệu lên đúng thiết bị (CPU/GPU)
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                # Xóa gradient từ batch trước (tránh tích lũy)
                optimizer.zero_grad()

            # Automatic Mixed Precision (AMP): dùng float16 trên GPU để nhanh hơn
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and scaler is not None)):
                outputs = model(images)           # Forward pass: tính logits
                loss = criterion(outputs, labels) # Tính cross-entropy loss

            if is_train:
                if scaler is not None and device.type == 'cuda':
                    # AMP backward: scale loss trước khi backward để tránh underflow float16
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard backward pass (CPU hoặc không dùng AMP)
                    loss.backward()
                    optimizer.step()

            # Tích lũy loss (nhân với batch size để lấy tổng tuyệt đối)
            total_loss += loss.item() * images.size(0)
            # Lấy class có xác suất cao nhất làm dự đoán
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Trung bình loss trên toàn bộ dataset (không phải trên batch)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# _train_phase — chạy một phase training với early stopping
# ---------------------------------------------------------------------------

def _train_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    early_stopping: EarlyStopping,
    device: torch.device,
    num_epochs: int,
    history: dict,
    phase_name: str,
    save_dir: str,
    model_name: str,
    scaler: torch.cuda.amp.GradScaler = None,
) -> bool:
    """
    Chạy training cho `num_epochs` epoch với early stopping và checkpoint.
    Trả về True nếu early stopping được kích hoạt.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE: {phase_name}  ({num_epochs} epochs max)")
    print(f"{'='*60}")

    best_val_acc = 0.0
    os.makedirs(save_dir, exist_ok=True)
    stopped_early = False

    for epoch in range(num_epochs):
        # Chạy training epoch
        train_loss, train_acc, _, _ = _run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True, scaler=scaler
        )
        # Chạy validation epoch (không có gradient, không cập nhật trọng số)
        val_loss, val_acc, val_preds, val_labels = _run_epoch(
            model, val_loader, criterion, optimizer, device, is_train=False, scaler=scaler
        )

        # Scheduler giảm LR khi val_acc ngừng tăng
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch+1:>2}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Ghi lại metrics của epoch này vào lịch sử
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Lưu checkpoint khi đạt accuracy tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  [Saved] Checkpoint saved  (val_acc={val_acc:.4f}) -> {save_path}")

        # Kiểm tra điều kiện early stopping
        if early_stopping.step(val_acc, model):
            stopped_early = True
            break

        # Nghỉ 5 giây giữa các epoch để GPU không quá nhiệt (quan trọng khi train lâu)
        if epoch < num_epochs - 1:
            print(f"  [Safety] Pausing for 5 seconds to cool down hardware...")
            time.sleep(5)

    return stopped_early, val_preds, val_labels


# ---------------------------------------------------------------------------
# train_model — hàm huấn luyện chính, điều phối toàn bộ quy trình 2 phase
# ---------------------------------------------------------------------------

def train_model(
    data_dir: str = None,
    model_name: str = None,
    batch_size: int = None,
    epochs: int = None,
    lr: float = None,
    save_dir: str = None,
) -> None:
    """
    Điểm vào chính để huấn luyện BrainTumorAI.

    Chiến lược 2-phase fine-tuning (từ DAKHDL):
      Phase 1 — Đóng băng backbone, chỉ train lớp classifier head (hội tụ nhanh).
      Phase 2 — Mở băng toàn bộ mạng để fine-tuning sâu (accuracy cao).

    Args:
        data_dir:   Đường dẫn dữ liệu training. Mặc định lấy từ config.
        model_name: 'efficientnet' | 'densenet'. Mặc định từ config.
        batch_size: Kích thước batch. Mặc định từ config.
        epochs:     Tổng số epochs (2 phase cộng lại). Mặc định từ config.
        lr:         Learning rate cho phase 2. Mặc định từ config.
        save_dir:   Thư mục lưu checkpoint. Mặc định từ config.
    """
    # Fallback về config nếu tham số không được truyền vào
    data_dir = data_dir or data_cfg.data_dir
    model_name = model_name or train_cfg.model_name
    batch_size = batch_size or train_cfg.batch_size
    save_dir = save_dir or train_cfg.save_dir

    # Bước 1: Cố định seed để kết quả lặp lại
    set_seed(SEED)
    device = get_device()

    # Bước 2: Tải và chia dataset (stratified split — giữ tỷ lệ lớp đồng đều)
    print("\n[Train] Preparing datasets ...")
    train_ds, val_ds = split_dataset(
        train_dir=data_dir,
        train_transform=get_train_transforms(),
        val_transform=get_valid_transforms(),
    )

    # Lấy class weights từ dataset gốc (truy cập qua .dataset vì là Subset)
    class_weights = train_ds.dataset.class_weights.to(device)

    # DataLoader training: shuffle=True để tránh model học thứ tự ảnh
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=data_cfg.num_workers, pin_memory=True
    )
    # DataLoader validation: shuffle=False vì thứ tự không ảnh hưởng đánh giá
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=data_cfg.num_workers, pin_memory=True
    )

    # Bước 3: Khởi tạo model và chuyển lên thiết bị tính toán
    print(f"\n[Train] Initialising model: {model_name} ...")
    model = get_model(model_name=model_name, num_classes=len(data_cfg.class_names), pretrained=True)
    model = model.to(device)

    # Hàm loss với trọng số lớp — giúp model không bỏ qua các lớp thiểu số
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # GradScaler: kết hợp với autocast để dùng AMP, tăng tốc training trên GPU
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Dictionary theo dõi metrics qua từng epoch
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Early stopping dùng chung cho cả 2 phase
    early_stopping = EarlyStopping(patience=train_cfg.early_stopping_patience)

    # ==================================================================
    # PHASE 1 — Đóng băng backbone, chỉ train lớp classifier head
    # Mục tiêu: khởi tạo tốt lớp cuối trước khi fine-tune toàn bộ mạng
    # ==================================================================
    if model_name.lower() in ["efficientnet", "densenet"]:
        print(f"\n[Phase 1] Freezing {model_name} backbone ...")
        # Tắt gradient tất cả tham số (không update backbone)
        for param in model.model.parameters():
            param.requires_grad = False
        # Bật lại gradient cho lớp phân loại cuối (classifier head)
        for param in model.model.classifier.parameters():
            param.requires_grad = True

        # AdamW optimizer — chỉ tối ưu các tham số đang bật gradient
        optimizer_p1 = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg.phase1_lr,
            weight_decay=train_cfg.weight_decay,
        )
        # ReduceLROnPlateau: giảm LR khi val_acc ngừng tăng (mode=max)
        scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_p1,
            mode="max",
            factor=train_cfg.lr_scheduler_factor,
            patience=train_cfg.lr_scheduler_patience,
        )

        stopped, _, _ = _train_phase(
            model, train_loader, val_loader, criterion,
            optimizer_p1, scheduler_p1, early_stopping,
            device, train_cfg.phase1_epochs, history,
            phase_name="Phase 1 — Classifier Head",
            save_dir=save_dir, model_name=model_name,
            scaler=scaler,
        )
    else:
        print(f"[Phase 1] Skipped for model {model_name} (if no clear backbone to freeze) — please verify implementation if needed")

    # ==================================================================
    # PHASE 2 — Mở băng toàn bộ mạng, fine-tune với LR nhỏ hơn
    # Mục tiêu: tinh chỉnh toàn bộ feature extractor để đạt accuracy cao nhất
    # ==================================================================
    print("\n[Phase 2] Unfreezing ALL layers for fine-tuning ...")
    # Bật gradient cho tất cả tham số (kể cả backbone)
    for param in model.parameters():
        param.requires_grad = True

    # Reset counter của early stopping để bắt đầu đếm mới cho phase 2
    early_stopping.counter = 0

    # LR nhỏ hơn nhiều (1e-4 thay vì 1e-3) để không phá hỏng trọng số pre-trained
    optimizer_p2 = optim.AdamW(
        model.parameters(),
        lr=lr or train_cfg.phase2_lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler_p2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p2,
        mode="max",
        factor=train_cfg.lr_scheduler_factor,
        patience=train_cfg.lr_scheduler_patience,
    )

    _, final_preds, final_labels = _train_phase(
        model, train_loader, val_loader, criterion,
        optimizer_p2, scheduler_p2, early_stopping,
        device, train_cfg.phase2_epochs, history,
        phase_name="Phase 2 — Full Fine-Tuning",
        save_dir=save_dir, model_name=model_name,
        scaler=scaler,
    )

    # Phục hồi trọng số tốt nhất từ cả 2 phase về model hiện tại
    early_stopping.restore_best_weights(model)

    # ==================================================================
    # Báo cáo cuối — in classification report và lưu lịch sử training
    # ==================================================================
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest val_acc: {early_stopping.best_val_acc:.4f}\n")
    print("Classification Report (Validation Set):")
    print(
        classification_report(
            final_labels, final_preds,
            target_names=data_cfg.class_names
        )
    )

    # Lưu toàn bộ lịch sử training ra JSON để vẽ biểu đồ sau
    history_path = os.path.join(
        train_cfg.reports_dir, "training_history.json"
    )
    save_history(history, history_path)
    print(f"\nTraining history -> {history_path}")


# ---------------------------------------------------------------------------
# CLI entry point — chạy trực tiếp từ command line với argparse
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train BrainTumorAI (EfficientNetB0 / DenseNet121)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to Training data directory (overrides config / .env)"
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        choices=["efficientnet", "densenet"],
        help="Model architecture to train"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Mini-batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Total number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate for phase-2 fine-tuning"
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save model checkpoints"
    )
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
    )
