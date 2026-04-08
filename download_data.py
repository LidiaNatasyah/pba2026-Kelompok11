"""
download_data.py — Auto-Download Dataset dari Hugging Face
==========================================================
Mendownload dataset "E-commerce Sentiment Bahasa Indonesia" 
secara otomatis menggunakan library datasets.
"""

import os
import pandas as pd
from datasets import load_dataset

from config import DATA_DIR, RAW_CSV

# ──────────────────────────────────────────────
# Konstanta Hugging Face
# ──────────────────────────────────────────────
HF_DATASET = "AIbnuHibban/e-commerce-sentiment-bahasa-indonesia"


def download_dataset() -> str:
    """
    Download dataset dari Hugging Face ke folder data/.

    Alur:
    1. Cek apakah file CSV sudah ada → skip jika sudah.
    2. Download via library 'datasets'.
    3. Konversi ke Pandas DataFrame dan simpan sebagai CSV.

    Returns:
        str: Path absolut ke file CSV yang siap dipakai.
    """

    # ── Sudah ada? Skip ──
    if os.path.exists(RAW_CSV):
        print(f"✅ Dataset sudah ada: {RAW_CSV}")
        return RAW_CSV

    # ── Download via Hugging Face datasets ──
    print(f"📥 Mendownload dataset dari {HF_DATASET}...")
    try:
        # Mengunduh dataset
        dataset = load_dataset(HF_DATASET)
        
        # Mengambil bagian 'train' dan mengubahnya menjadi DataFrame
        df = pd.DataFrame(dataset['train'])
        
        # Memastikan folder direktori tujuan sudah dibuat
        os.makedirs(os.path.dirname(RAW_CSV), exist_ok=True)
        
        # Menyimpan DataFrame ke format CSV
        df.to_csv(RAW_CSV, index=False)
        
        print(f"✅ Dataset berhasil diunduh dan disalin ke: {RAW_CSV}")
        return RAW_CSV

    except Exception as e:
        raise RuntimeError(
            f"❌ Gagal mendownload dataset dari Hugging Face.\n"
            f"   Error: {e}\n"
            f"   Pastikan library 'datasets' dan 'pandas' sudah terinstall."
        ) from e


# ──────────────────────────────────────────────
# Jika dijalankan langsung: python download_data.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    path = download_dataset()
    print(f"\n🎉 Dataset siap digunakan untuk PyCaret: {path}")