"""
train.py — AutoML Training Pipeline via PyCaret
================================================
Setup PyCaret, compare models, tune, evaluate, dan finalize + export.
"""

import os
import warnings
import pandas as pd

from config import (
    LABEL_COL,
    MODEL_DIR,
    SESSION_ID,
    TRAIN_SIZE,
    N_TOP_MODELS,
)

# Suppress warnings agar output lebih bersih
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# ⚙️ SETUP PYCARET
# ══════════════════════════════════════════════

def setup_pycaret(df: pd.DataFrame):
    from pycaret.classification import setup

    print("⚙️  Menginisialisasi PyCaret...")
    print(f"   Kolom teks  : cleaned_text")
    print(f"   Kolom label : {LABEL_COL}")
    
    # Siapkan DataFrame hanya dengan kolom yang diperlukan
    df_model = df[["cleaned_text", LABEL_COL]].copy()

    # use_gpu di-set ke False agar tidak error
    s = setup(
        data=df_model,
        target=LABEL_COL,
        text_features=["cleaned_text"],
        session_id=SESSION_ID,
        train_size=TRAIN_SIZE,
        verbose=True,
        html=False, 
        use_gpu=False, 
        fold=5,
    )

    print("✅ PyCaret setup selesai!")
    return s

# ══════════════════════════════════════════════
# 🏟️ MODEL ARENA — COMPARE MODELS
# ══════════════════════════════════════════════

def compare_all_models(sort: str = "F1", n_select: int = None):
    # pull digunakan untuk mengambil tabel hasil perbandingan
    from pycaret.classification import compare_models, pull 

    if n_select is None:
        n_select = N_TOP_MODELS

    print(f"🏟️  Memulai Model Arena (sort by {sort})...")
    
    # Memilih 3 model sesuai permintaan tugas besar
    best = compare_models(
        include=['svm', 'lr', 'lightgbm'],
        sort=sort, 
        n_select=n_select,
    )

    # Simpan tabel hasil ke CSV untuk laporan
    results_df = pull()
    results_df.to_csv("hasil_perbandingan_model.csv", index=False)
    print("\n📊 Tabel perbandingan telah disimpan ke: hasil_perbandingan_model.csv")

    print(f"✅ Selesai! Top {n_select} model telah dipilih.")
    return best

# ══════════════════════════════════════════════
# 🎯 TUNING & SAVING
# ══════════════════════════════════════════════

def tune_best(model, optimize: str = "F1"):
    from pycaret.classification import tune_model
    print(f"🎯 Tuning hyperparameter (optimize: {optimize})...")
    tuned = tune_model(model, optimize=optimize)
    print("✅ Tuning selesai!")
    return tuned

def finalize_and_save(model, filename: str = "nlp_pipeline_final"):
    from pycaret.classification import finalize_model, save_model

    print("💾 Memfinalisasi model (retrain pada seluruh data)...")
    final = finalize_model(model)

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, filename)
    save_model(final, save_path)
    print(f"✅ Model disimpan: {save_path}.pkl")

    return final