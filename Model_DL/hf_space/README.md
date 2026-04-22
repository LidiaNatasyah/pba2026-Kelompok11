# 🛒 E-Commerce Sentiment Analyzer

Klasifikasi sentimen ulasan produk e-commerce berbahasa Indonesia menggunakan arsitektur **Deep Learning (BiLSTM)**. 

Aplikasi/repositori ini menganalisis teks ulasan pelanggan dan memprediksi salah satu dari **3 kelas sentimen**:
- **Positive** — ulasan yang menunjukkan kepuasan pelanggan.
- **Neutral** — ulasan yang bersifat umum atau tidak memihak.
- **Negative** — ulasan yang menunjukkan kekecewaan atau keluhan.

## 🚀 Model & Performa

Model ini dilatih menggunakan **PyTorch** dengan teknik *word embedding* yang dipelajari langsung dari dataset ulasan e-commerce Indonesia untuk menangkap nuansa bahasa lokal dan singkatan khas belanja *online*.

| Komponen | Keterangan |
|-------|-----------|
| **Arsitektur** | Bidirectional LSTM (BiLSTM) - 2 Layers |
| **Framework** | PyTorch |
| **Total Parameter** | ~2.49 Juta |
| **Akurasi (Test)** | **98.87%** |
| **F1-Score** | **0.9887** |

## 📂 Dataset

Dataset yang digunakan adalah [E-Commerce Sentiment Bahasa Indonesia](https://huggingface.co/datasets/AIbnuHibban/e-commerce-sentiment-bahasa-indonesia) yang berisi ribuan ulasan pelanggan dari berbagai platform e-commerce.

## 🛠️ Instalasi & Prasyarat

Pastikan Anda memiliki Python 3.8+ terinstal di sistem Anda.

1. **Clone repositori ini:**
   ```bash
   git clone [https://github.com/username-anda/nama-repo-anda.git](https://github.com/username-anda/nama-repo-anda.git)
   cd nama-repo-anda
