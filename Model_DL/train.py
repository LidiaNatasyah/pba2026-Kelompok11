"""
train.py — Training Loop, Evaluasi, dan Visualisasi
=====================================================
Menyediakan fungsi modular untuk:
- Melatih model BiLSTM per epoch
- Early stopping
- Evaluasi & laporan klasifikasi
- Visualisasi: training curves, confusion matrix
- Inferensi teks tunggal
"""

import os
import time
import copy

import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from config import (
    DEVICE, LABEL_LIST, NUM_CLASSES, PLOT_DIR
)

matplotlib.use("Agg")   # backend tanpa GUI agar aman di notebook & terminal


# ──────────────────────────────────────────────
# ⚙️  HELPER: seed & criterion
# ──────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set seed untuk reproducibility (PyTorch, NumPy, random)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_criterion(label_counts: dict | None = None) -> nn.CrossEntropyLoss:
    """Membuat CrossEntropyLoss dengan class weight opsional."""
    if label_counts is None:
        return nn.CrossEntropyLoss()

    total = sum(label_counts.values())
    weights = torch.tensor(
        [total / (NUM_CLASSES * label_counts.get(lbl, 1)) for lbl in LABEL_LIST],
        dtype=torch.float,
    ).to(DEVICE)
    return nn.CrossEntropyLoss(weight=weights)


# ──────────────────────────────────────────────
# 🔄 TRAINING & EVALUASI
# ──────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> tuple[float, float]:
    """Melatih BiLSTM selama satu epoch."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch in dataloader:
        x, labels, lengths = batch
        x, labels, lengths = x.to(device), labels.to(device), lengths.to(device)

        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
) -> tuple[float, float, list, list]:
    """Evaluasi BiLSTM pada dataloader."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x, labels, lengths = batch
            x, labels, lengths = x.to(device), labels.to(device), lengths.to(device)

            logits = model(x, lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds   = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────────────
# 🏋️  TRAINING LOOP LENGKAP
# ──────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    save_path: str,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device = DEVICE,
    label_counts: dict | None = None,
) -> dict:
    """Melatih model dengan early stopping dan menyimpan model terbaik."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = get_criterion(label_counts)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    best_val_loss    = float("inf")
    best_model_state = None
    patience_counter = 0
    start_time       = time.time()

    print(f"\n{'='*60}")
    print(f"  Training: BiLSTM  |  Device: {device}")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  Patience: {patience}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0           = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:>3}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_model_state, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⏹ Early stopping di epoch {epoch}")
                break

    total_time = time.time() - start_time
    history["total_time"] = total_time
    print(f"\n  ✅ Selesai dalam {total_time/60:.1f} menit  |  Best Val Loss: {best_val_loss:.4f}")
    print(f"  💾 Model disimpan: {save_path}")

    model.load_state_dict(best_model_state)
    return history


# ──────────────────────────────────────────────
# 📈 VISUALISASI
# ──────────────────────────────────────────────

def plot_training_curves(history: dict, save: bool = True) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Curves — BiLSTM", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="s")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   marker="s")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOT_DIR, "training_curves_bilstm.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Plot disimpan: {path}")
    plt.show()


def plot_confusion_matrix(y_true: list, y_pred: list, save: bool = True) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[l.capitalize() for l in LABEL_LIST], 
        yticklabels=[l.capitalize() for l in LABEL_LIST],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title("Confusion Matrix — BiLSTM", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, "confusion_matrix_bilstm.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Plot disimpan: {path}")
    plt.show()


def print_classification_report(y_true: list, y_pred: list) -> dict:
    print(f"\n{'='*60}")
    print("  Classification Report — BiLSTM")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=[l.capitalize() for l in LABEL_LIST], zero_division=0))

    acc         = accuracy_score(y_true, y_pred)
    f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1 Macro    : {f1_macro:.4f}")
    print(f"  F1 Weighted : {f1_weighted:.4f}")

    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


# ──────────────────────────────────────────────
# 🔍 INFERENSI TEKS TUNGGAL
# ──────────────────────────────────────────────

def predict_single(
    model: nn.Module,
    text: str,
    vocab,
    device: torch.device = DEVICE,
) -> dict:
    """Prediksi sentimen dari satu teks menggunakan BiLSTM."""
    from config import MAX_LEN
    from preprocess import clean_text

    model.eval()
    cleaned = clean_text(text)
    indices = vocab.text_to_indices(cleaned, MAX_LEN)
    length  = max(min(len(cleaned.split()), MAX_LEN), 1)

    x       = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x, lengths)

    probs      = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx   = int(probs.argmax())
    pred_label = LABEL_LIST[pred_idx].capitalize()

    return {
        "label":         pred_label,
        "confidence":    float(probs[pred_idx]),
        "probabilities": {LABEL_LIST[i].capitalize(): float(probs[i]) for i in range(NUM_CLASSES)},
    }