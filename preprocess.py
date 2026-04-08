"""
preprocess.py — Custom Text Cleaning untuk Ulasan E-commerce
================================================================
Pipeline: lowercase → hapus URL/mention → normalisasi leetspeak →
ekspansi slang e-commerce → hapus karakter non-alfabet → strip whitespace.
"""

import re
import pandas as pd

# Pastikan config.py sudah di-update dengan versi e-commerce
from config import LEETSPEAK_MAP, SLANG_DICT, TEXT_COL, LABEL_COL, RAW_CSV

# ══════════════════════════════════════════════
# 🔧 FUNGSI PEMBANTU (Sama persis logikanya)
# ══════════════════════════════════════════════

def normalize_leetspeak(text: str) -> str:
    result = []
    for i, char in enumerate(text):
        if char in LEETSPEAK_MAP:
            prev_is_alpha = (i > 0 and text[i - 1].isalpha())
            next_is_alpha = (i < len(text) - 1 and text[i + 1].isalpha())

            if prev_is_alpha or next_is_alpha:
                result.append(LEETSPEAK_MAP[char])
            else:
                result.append(char)
        else:
            result.append(char)
    return "".join(result)

def expand_slang(text: str) -> str:
    words = text.split()
    expanded = [SLANG_DICT.get(w, w) for w in words]
    return " ".join(expanded)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = normalize_leetspeak(text)
    text = expand_slang(text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ══════════════════════════════════════════════
# 📂 FUNGSI UTAMA (Diperbarui untuk e-commerce)
# ══════════════════════════════════════════════

def load_and_clean(csv_path: str = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = RAW_CSV

    print(f"📂 Membaca dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"   Jumlah baris awal: {len(df):,}")
    
    # Kunci Utama: Hanya ambil kolom yang kita butuhkan untuk PyCaret
    # Jika di datamu namanya beda, kita setel sesuai config
    df = df[[TEXT_COL, LABEL_COL]].copy()

    # Hapus baris kosong
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)

    # Bersihkan teks ulasan
    print("🧹 Membersihkan teks ulasan...")
    df["cleaned_text"] = df[TEXT_COL].apply(clean_text)

    # Hapus ulasan yang setelah dibersihkan ternyata jadi kosong (misal isinya cuma emoji)
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)

    print(f"✅ Selesai! Jumlah baris bersih: {len(df):,}")
    return df

def show_cleaning_examples(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    sample = df.sample(n=min(n, len(df)), random_state=42)
    return sample[[TEXT_COL, "cleaned_text", LABEL_COL]].rename(
        columns={TEXT_COL: "original", "cleaned_text": "cleaned", LABEL_COL: "label"}
    )

# ──────────────────────────────────────────────
# Jika dijalankan langsung: python preprocess.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    df = load_and_clean()
    print("\n📋 Contoh hasil pembersihan:")
    print(show_cleaning_examples(df).to_string(index=False))