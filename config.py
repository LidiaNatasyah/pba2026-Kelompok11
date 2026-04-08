"""
config.py — Konfigurasi & Konstanta untuk NLP
=============================================================
Berisi path, mapping leetspeak, kamus slang e-commerce Indonesia,
dan daftar stopwords dasar.
"""

import os

# ──────────────────────────────────────────────
# 📁 PATH
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# NAMA FILE DIUBAH SESUAI DATASETMU
RAW_CSV = os.path.join(DATA_DIR, "ecommerce_sentiment.csv")

# Buat folder kalau belum ada
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 🔤 LEETSPEAK MAPPING
# ──────────────────────────────────────────────
# Pembeli kadang typo atau mengganti huruf dengan angka
LEETSPEAK_MAP = {
    "0": "o",
    "1": "i",
    "2": "z",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
}

# ──────────────────────────────────────────────
# 💬 KAMUS SLANG E-COMMERCE INDONESIA
# ──────────────────────────────────────────────
# Singkatan & slang yang umum di ulasan belanja online.
SLANG_DICT = {
    # --- Belanja & Produk ---
    "brg": "barang",
    "brng": "barang",
    "bgs": "bagus",
    "bgsgt": "bagus banget",
    "ori": "original",
    "min": "admin",
    "packing": "kemasan",
    "seller": "penjual",
    "hrg": "harga",
    "murmer": "murah meriah",
    "rekomen": "rekomendasi",
    "nyampe": "sampai",
    "mendarat": "sampai",
    "kurir": "kurir",
    "cpt": "cepat",
    "cepet": "cepat",
    "lmyn": "lumayan",
    "kcewa": "kecewa",
    "pict": "gambar",
    "realpict": "sesuai gambar",
    
    # --- Slang umum ---
    "gw": "gue",
    "gua": "gue",
    "lu": "lo",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "g": "tidak",
    "tdk": "tidak",
    "gk": "tidak",
    "kyk": "kayak",
    "bgt": "banget",
    "bngt": "banget",
    "udh": "sudah",
    "udah": "sudah",
    "blm": "belum",
    "yg": "yang",
    "dgn": "dengan",
    "sm": "sama",
    "tp": "tapi",
    "tpi": "tapi",
    "krn": "karena",
    "kalo": "kalau",
    "klo": "kalau",
    "jgn": "jangan",
    "gpp": "tidak apa-apa",
    "cmn": "cuman",
    "lg": "lagi",
    "aja": "saja",
    "bs": "bisa",
    "dr": "dari",
    "utk": "untuk",
    "pdhl": "padahal",
    "msh": "masih"
}

# ──────────────────────────────────────────────
# 🛑 KOLOM DATASET
# ──────────────────────────────────────────────
# Berdasarkan struktur asli dataset e-commerce milikmu
TEXT_COL = "comment"
LABEL_COL = "sentiment"
# ──────────────────────────────────────────────
# 🎯 PYCARET SETTINGS
# ──────────────────────────────────────────────
SESSION_ID = 42        # Random seed untuk reprodusibilitas
TRAIN_SIZE = 0.8       # 80% train, 20% test
N_TOP_MODELS = 3       # Ubah jadi 3 karena kamu cuma butuh (SVM, LR, LightGBM)