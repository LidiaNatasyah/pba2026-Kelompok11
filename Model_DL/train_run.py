"""
train_run.py — Skrip Terminal: Pipeline Pelatihan Lengkap (BiLSTM)
=========================================================
Menjalankan seluruh pipeline dari awal hingga akhir:
download → preprocess → build vocab → train BiLSTM → save

Jalankan dengan:
    python train_run.py
"""

import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DEVICE, SAMPLE_SIZE,
    VOCAB_SIZE, MAX_LEN, LSTM_BATCH_SIZE, LSTM_LR, LSTM_EPOCHS, LSTM_PATIENCE,
    BILSTM_MODEL_PATH, VOCAB_PATH, PLOT_DIR,
)
from download_data import download_dataset
from preprocess import load_and_clean, show_cleaning_examples
from dataset import Vocabulary, get_lstm_dataloaders
from models import BiLSTMClassifier, count_parameters
from train import (
    set_seed, train_model, evaluate, get_criterion,
    plot_training_curves, plot_confusion_matrix,
    print_classification_report
)

# ──────────────────────────────────────────────
def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ──────────────────────────────────────────────

def main():
    set_seed(42)
    print(f"🖥️  Device: {DEVICE}")
    print(f"📊 Sample size: {SAMPLE_SIZE:,}")
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── 1. DOWNLOAD ──────────────────────────────
    section("1. Download Dataset dari Hugging Face")
    csv_path = download_dataset()

    # ── 2. PREPROCESS ────────────────────────────
    section("2. Preprocessing Bahasa Indonesia")
    df = load_and_clean(csv_path, sample_size=SAMPLE_SIZE)
    show_cleaning_examples(df, n=3)

    # Label counts untuk weighted loss
    label_counts = df["label"].value_counts().to_dict()

    # Plot distribusi kelas
    plt.figure(figsize=(7, 4))
    df["label"].value_counts().sort_values().plot(kind="barh", color=sns.color_palette("Set2"))
    plt.title("Distribusi Kelas Sentimen E-commerce")
    plt.xlabel("Jumlah Data")
    plt.tight_layout()
    dist_path = os.path.join(PLOT_DIR, "distribusi_kelas.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Plot distribusi disimpan: {dist_path}")

    # ── 3. BUILD VOCABULARY ──────────────────────
    section("3. Membangun Vocabulary")
    vocab = Vocabulary()
    vocab.build_vocab(df["cleaned_text"].tolist(), max_size=VOCAB_SIZE)
    vocab.save(VOCAB_PATH)

    # ── 4. DATALOADER ─────────────────────
    section("4. Membuat DataLoaders")
    train_loader, val_loader, test_loader = get_lstm_dataloaders(
        df, vocab, max_len=MAX_LEN, batch_size=LSTM_BATCH_SIZE
    )

    # ── 5. TRAIN BILSTM ───────────────────────────
    section("5. Training BiLSTM")
    bilstm = BiLSTMClassifier(vocab_size=len(vocab))
    print(f"   Jumlah Parameter: {count_parameters(bilstm):,} (Memenuhi syarat ToR <10 Juta)")

    hist_bilstm = train_model(
        model=bilstm,
        train_loader=train_loader, val_loader=val_loader,
        save_path=BILSTM_MODEL_PATH,
        epochs=LSTM_EPOCHS, lr=LSTM_LR, patience=LSTM_PATIENCE,
        device=DEVICE, label_counts=label_counts,
    )
    plot_training_curves(hist_bilstm)

    # Evaluasi BiLSTM pada test set
    criterion = get_criterion(label_counts)
    _, _, preds_bilstm, labels_bilstm = evaluate(
        bilstm, test_loader, criterion, DEVICE
    )
    
    section("6. Evaluasi Akhir Test Set")
    plot_confusion_matrix(labels_bilstm, preds_bilstm)
    metrics_bilstm = print_classification_report(labels_bilstm, preds_bilstm)
    metrics_bilstm["training_time_min"] = hist_bilstm["total_time"] / 60

    print("\n✅ Pipeline BiLSTM selesai! Model dan Vocab tersimpan di folder models/")

if __name__ == "__main__":
    main()