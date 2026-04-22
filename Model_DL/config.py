"""
config.py — Konfigurasi & Konstanta untuk NLP
================================================================
Berisi path, hyperparameter model, dan label mapping untuk
analisis sentimen e-commerce bahasa Indonesia (BiLSTM).
"""

import os
import torch

# ──────────────────────────────────────────────
# 📁 PATH
# ──────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR  = os.path.join(BASE_DIR, "plots")

# 1. DIUBAH: Menyesuaikan nama file output dari download_data.py
RAW_CSV = os.path.join(DATA_DIR, "dataset.csv")

# Buat folder kalau belum ada
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# ──────────────────────────────────────────────
# 📋 DATASET
# ──────────────────────────────────────────────
HF_DATASET = "AIbnuHibban/e-commerce-sentiment-bahasa-indonesia"

# SESUAIKAN DENGAN NAMA KOLOM DI GAMBARMU
TEXT_COL  = "comment"
LABEL_COL = "sentiment"

# Pastikan isi datanya benar-benar bahasa inggris (negative, neutral, positive).
# Kalau di dalam CSV-nya pakai bahasa Indonesia (negatif, netral, positif),
# ubah teks di dalam tanda kutip di bawah ini menyesuaikan isi datanya ya.
LABEL_LIST = [
    "negative",
    "neutral",
    "positive",
]
NUM_CLASSES = len(LABEL_LIST)

# ──────────────────────────────────────────────
# ⚙️  UMUM
# ──────────────────────────────────────────────
RANDOM_SEED = 42
SAMPLE_SIZE = 15_000      # None = pakai semua data
TEST_SIZE   = 0.10        # 80 / 10 / 10  train / val / test
VAL_SIZE    = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# 🧠 HYPERPARAMETER — BiLSTM & BiLSTM+Attention
# ──────────────────────────────────────────────
# BiLSTM tetap dipertahankan sama persis seperti template asli dosen
VOCAB_SIZE  = 20_000
EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3
MAX_LEN     = 128

LSTM_EPOCHS     = 10
LSTM_BATCH_SIZE = 64
LSTM_LR         = 1e-3
LSTM_PATIENCE   = 3       # early stopping patience

# ──────────────────────────────────────────────
# 💾 NAMA FILE MODEL
# ──────────────────────────────────────────────
BILSTM_MODEL_PATH     = os.path.join(MODEL_DIR, "bilstm.pt")
BILSTM_ATT_MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_attention.pt")
VOCAB_PATH            = os.path.join(MODEL_DIR, "vocab.json")
LABEL_ENCODER_PATH    = os.path.join(MODEL_DIR, "label_encoder.json")