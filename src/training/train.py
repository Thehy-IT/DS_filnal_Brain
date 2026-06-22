"""
train.py — Full training pipeline for BrainTumorAI.

Improvements over v1 (all learned from DAKHDL's 02_HuanLuyen_Model.ipynb):
  - set_seed() for full reproducibility
  - Stratified train/val split (fixes the transform-contamination bug)
  - class_weight passed to CrossEntropyLoss
  - 2-phase fine-tuning: freeze backbone → then unfreeze all layers
  - EarlyStopping (patience from config)
  - ReduceLROnPlateau scheduler (per-epoch)
  - Training history saved as JSON for later plotting
  - Verbose per-epoch logging with loss + accuracy
"""
import argparse
import os

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
# EarlyStopping helper
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when val_acc has not improved for `patience` epochs.
    Mirrors DAKHDL's Keras EarlyStopping(monitor='val_accuracy', restore_best_weights=True).
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_acc: float = 0.0
        self.counter: int = 0
        self.best_state: dict = {}

    def step(self, val_acc: float, model: nn.Module) -> bool:
        """
        Returns True if training should stop.
        Saves model state whenever a new best is found.
        """
        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.counter = 0
            # Deep copy the state dict (cheap for ~5 MB EfficientNetB0)
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(
                f"\n[EarlyStopping] No improvement for {self.patience} epochs. "
                f"Best val_acc = {self.best_val_acc:.4f}"
            )
            return True
        return False

    def restore_best_weights(self, model: nn.Module) -> None:
        """Load the weights from the best epoch back into the model."""
        if self.best_state:
            model.load_state_dict(self.best_state)
            print(f"[EarlyStopping] Restored best weights (val_acc={self.best_val_acc:.4f})")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    is_train: bool,
) -> tuple:
    """Run one epoch (train or eval). Returns (avg_loss, accuracy)."""
    model.train(is_train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    desc = "Train" if is_train else "Val"

    with ctx:
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


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
) -> bool:
    """
    Run training for `num_epochs` with early stopping.
    Returns True if early stopping was triggered.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE: {phase_name}  ({num_epochs} epochs max)")
    print(f"{'='*60}")

    best_val_acc = 0.0
    os.makedirs(save_dir, exist_ok=True)
    stopped_early = False

    for epoch in range(num_epochs):
        train_loss, train_acc, _, _ = _run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        val_loss, val_acc, val_preds, val_labels = _run_epoch(
            model, val_loader, criterion, optimizer, device, is_train=False
        )

        # Scheduler step (on val_acc)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch+1:>2}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Persist metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Checkpoint saved  (val_acc={val_acc:.4f}) → {save_path}")

        # Early stopping check
        if early_stopping.step(val_acc, model):
            stopped_early = True
            break

    return stopped_early, val_preds, val_labels


def train_model(
    data_dir: str = None,
    model_name: str = None,
    batch_size: int = None,
    epochs: int = None,
    lr: float = None,
    save_dir: str = None,
) -> None:
    """
    Main entry point for training BrainTumorAI.

    Implements a 2-phase fine-tuning strategy (from DAKHDL):
      Phase 1 — Freeze backbone, train only the classifier head (fast convergence).
      Phase 2 — Unfreeze all layers for full fine-tuning (high accuracy).

    Args:
        data_dir:   Path to training data directory. Defaults to config.
        model_name: 'efficientnet' | 'densenet'. Defaults to config.
        batch_size: Mini-batch size. Defaults to config.
        epochs:     TOTAL epochs (phases combined). Defaults to config.
        lr:         Phase-2 fine-tuning learning rate. Defaults to config.
        save_dir:   Directory to save model checkpoints. Defaults to config.
    """
    # ----- Resolve arguments from config if not provided -----
    data_dir = data_dir or data_cfg.data_dir
    model_name = model_name or train_cfg.model_name
    batch_size = batch_size or train_cfg.batch_size
    save_dir = save_dir or train_cfg.save_dir

    # ----- Reproducibility -----
    set_seed(SEED)
    device = get_device()

    # ----- Data -----
    print("\n[Train] Preparing datasets …")
    train_ds, val_ds = split_dataset(
        train_dir=data_dir,
        train_transform=get_train_transforms(),
        val_transform=get_valid_transforms(),
    )

    # Class weights from the full training dataset labels
    # (access the underlying BrainTumorDataset via .dataset)
    class_weights = train_ds.dataset.class_weights.to(device)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=data_cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=data_cfg.num_workers, pin_memory=True
    )

    # ----- Model -----
    print(f"\n[Train] Initialising model: {model_name} …")
    model = get_model(model_name=model_name, num_classes=len(data_cfg.class_names), pretrained=True)
    model = model.to(device)

    # ----- Loss with class weights (address imbalance) -----
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ----- Training history -----
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ----- Early stopping (shared across both phases) -----
    early_stopping = EarlyStopping(patience=train_cfg.early_stopping_patience)

    # ==================================================================
    # PHASE 1 — Freeze backbone, train classifier head only
    # (From DAKHDL: first train only the new top layers)
    # ==================================================================
    if model_name.lower() in ["efficientnet", "densenet"]:
        print(f"\n[Phase 1] Freezing {model_name} backbone …")
        for param in model.model.parameters():
            param.requires_grad = False
        # Unfreeze the final classification head
        for param in model.model.classifier.parameters():
            param.requires_grad = True

        optimizer_p1 = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg.phase1_lr,
            weight_decay=train_cfg.weight_decay,
        )
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
        )
    else:
        print(f"[Phase 1] Skipped for model {model_name} (if no clear backbone to freeze) — please verify implementation if needed")

    # ==================================================================
    # PHASE 2 — Unfreeze all layers for fine-tuning
    # (From DAKHDL: then fine-tune the entire network with a lower LR)
    # ==================================================================
    print("\n[Phase 2] Unfreezing ALL layers for fine-tuning …")
    for param in model.parameters():
        param.requires_grad = True

    # Reset early stopping counter for phase 2
    early_stopping.counter = 0

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
    )

    # Restore best weights found across both phases
    early_stopping.restore_best_weights(model)

    # ==================================================================
    # Final Evaluation
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

    # Save training history
    history_path = os.path.join(
        train_cfg.reports_dir, "training_history.json"
    )
    save_history(history, history_path)
    print(f"\nTraining history → {history_path}")


# ---------------------------------------------------------------------------
# CLI entry point
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
