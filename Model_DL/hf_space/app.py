"""
app.py — Hugging Face Space: E-commerce Sentiment Analyzer
=============================================================
Gradio app untuk model BiLSTM klasifikasi sentimen ulasan e-commerce.

Upload model ke HF Hub terlebih dahulu:
    bilstm.pt, vocab.json

Jalankan lokal:
    python app.py
"""

import os
import re
import json
import torch
import torch.nn as nn
import gradio as gr

# ──────────────────────────────────────────────
# Konfigurasi
# ──────────────────────────────────────────────
LABEL_LIST = [
    "negative",
    "neutral",
    "positive",
]
NUM_CLASSES  = len(LABEL_LIST)
VOCAB_SIZE   = 20_000
EMBED_DIM    = 128
HIDDEN_DIM   = 256
NUM_LAYERS   = 2
DROPOUT      = 0.3
MAX_LEN      = 128

DEVICE = torch.device("cpu")   # HF free tier: CPU only

# ──────────────────────────────────────────────
# Text cleaning (Duplikasi dari preprocess.py agar app mandiri)
# ──────────────────────────────────────────────
SLANG_MAP = {
    "yg": "yang", "bgt": "banget", "bgtt": "banget", "ga": "tidak",
    "gak": "tidak", "gk": "tidak", "nggak": "tidak", "dgn": "dengan",
    "krn": "karena", "karna": "karena", "tp": "tapi", "udh": "sudah",
    "sdh": "sudah", "bagusss": "bagus", "jg": "juga", "dr": "dari",
    "utk": "untuk", "pdhl": "padahal", "bs": "bisa", "aj": "saja",
    "aja": "saja", "bkn": "bukan", "sampe": "sampai", "toko": "toko",
    "brg": "barang", "ori": "original"
}

def clean_text(text: str) -> str:
    """Membersihkan teks ulasan bahasa Indonesia."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    
    # Normalisasi slang
    words = text.split()
    words = [SLANG_MAP.get(word, word) for word in words]
    text = " ".join(words)
    
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ──────────────────────────────────────────────
# Vocabulary 
# ──────────────────────────────────────────────
class Vocabulary:
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)

    def text_to_indices(self, text: str, max_len: int = MAX_LEN) -> list:
        tokens  = text.split()[:max_len]
        indices = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        indices += [self.PAD_IDX] * (max_len - len(indices))
        return indices

    def __len__(self):
        return len(self.word2idx)

# ──────────────────────────────────────────────
# Model Definition
# ──────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 num_classes=NUM_CLASSES, dropout=DROPOUT, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        packed   = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(last_hidden))

# ──────────────────────────────────────────────
# Load Model & Vocab
# ──────────────────────────────────────────────
def _load_model():
    """Muat model BiLSTM dan Vocab."""
    # Menyesuaikan path untuk Hugging Face (biasanya di root folder yang sama)
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    
    # Fallback jika model langsung ditaruh di folder yang sama dengan app.py
    if not os.path.exists(models_dir):
        models_dir = os.path.dirname(__file__)

    vocab_path  = os.path.join(models_dir, "vocab.json")
    bilstm_path = os.path.join(models_dir, "bilstm.pt")

    vocab = Vocabulary()
    if os.path.exists(vocab_path):
        vocab.load(vocab_path)
    else:
        print(f"⚠️ vocab.json tidak ditemukan di {vocab_path}")

    model = None
    if os.path.exists(bilstm_path):
        model = BiLSTMClassifier(vocab_size=len(vocab))
        model.load_state_dict(torch.load(bilstm_path, map_location=DEVICE))
        model.eval()
        print("✅ BiLSTM dimuat")
    else:
        print(f"⚠️ bilstm.pt tidak ditemukan di {bilstm_path}")

    return model, vocab

MODEL, VOCAB = _load_model()

# ──────────────────────────────────────────────
# Fungsi Prediksi Utama
# ──────────────────────────────────────────────
def predict(text: str):
    """Prediksi sentimen dari teks input."""
    if not text or not text.strip():
        return "—", {}

    if MODEL is None:
        return "Model BiLSTM belum dimuat.", {}

    cleaned = clean_text(text)
    indices = VOCAB.text_to_indices(cleaned, MAX_LEN)
    
    # Antisipasi teks kosong setelah dibersihkan
    if len(cleaned.split()) == 0:
        return "Netral (Teks Kosong/Invalid)", {l: 0.0 for l in LABEL_LIST}
        
    length  = max(min(len(cleaned.split()), MAX_LEN), 1)
    
    x       = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    lengths = torch.tensor([length], dtype=torch.long)

    with torch.no_grad():
        logits = MODEL(x, lengths)
        probs  = torch.softmax(logits, dim=1).squeeze(0).numpy()

    pred_idx   = int(probs.argmax())
    pred_label = LABEL_LIST[pred_idx].capitalize()

    conf_dict = {LABEL_LIST[i].capitalize(): round(float(probs[i]), 4)
                 for i in range(NUM_CLASSES)}

    return pred_label, conf_dict

# ──────────────────────────────────────────────
# Gradio Interface
# ──────────────────────────────────────────────
CONTOH_TEKS = [
    "Barangnya bagus bgt, nyampenya juga cepet. Makasih min!",
    "Kecewa sih, pesen warna hitam yang dateng malah merah. Respon seller juga lambat.",
    "Biasa aja, kualitas sesuai sama harga. Gak berharap lebih juga.",
    "Suka banget sama bahannya, lembut dan adem pas dipake. Bakal order lagi nih.",
    "Packing hancur lebur, untung isi barangnya ga rusak. Tolong diperbaiki lagi ya kurirnya."
]

with gr.Blocks(title="E-Commerce Sentiment Analyzer") as demo:
    gr.Markdown("""
    # 🛒 E-Commerce Sentiment Analyzer
    Klasifikasi sentimen ulasan produk e-commerce berbahasa Indonesia menggunakan arsitektur **Deep Learning (BiLSTM)**.
    
    **3 Kelas Sentimen:** Positif · Netral · Negatif
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                lines=4,
                placeholder="Masukkan teks ulasan di sini...",
                label="Teks Ulasan Produk",
            )
            submit_btn   = gr.Button("🔍 Analisis Sentimen", variant="primary")
            example_comp = gr.Examples(
                examples=CONTOH_TEKS,
                inputs=text_input,
                label="Contoh Ulasan",
            )

        with gr.Column(scale=2):
            label_output  = gr.Label(label="Prediksi Sentimen")
            probs_output  = gr.Label(label="Tingkat Keyakinan (Confidence)", num_top_classes=3)

    submit_btn.click(
        fn=predict,
        inputs=[text_input],
        outputs=[label_output, probs_output],
    )

    gr.Markdown("""
    ---
    **Arsitektur Model:** Bidirectional LSTM (BiLSTM)  
    **Tugas Besar NLP (SD25-32202) — Institut Teknologi Sumatera (ITERA)**
    """)

if __name__ == "__main__":
    demo.launch()