# Proyek PBA 2026 - [Kelompok 11]

## Anggota Kelompok

| Nama | NIM | Username GitHub |
|-----|-----|-----|
| Lidia Natasyah Marpaung | 123450015 | @LidiaNatasyah |
| Vania Claresta | 123450029 | @vaniaclrstaa |
| Iqfina Haula Halika | 123450076 | @iqfinahalika |

## Dataset
Dataset: https://huggingface.co/datasets/AIbnuHibban/e-commerce-sentiment-bahasa-indonesia

Deskripsi Dataset:
Dataset yang digunakan adalah "E-commerce Sentiment Bahasa Indonesia" yang berisi ribuan ulasan pelanggan dari berbagai platform belanja daring. Setiap data telah dilabeli secara manual ke dalam kategori sentimen positif, netral, dan negatif. Dataset ini mencakup variasi bahasa konsumen Indonesia (formal dan non-formal), menjadikannya sumber data yang ideal untuk melatih serta menguji akurasi model klasifikasi teks dalam menangkap opini publik terhadap layanan retail digital.

## Model yang Digunakan
Machine Learning:

LightGBM

Logistic Regression

SVM - Linear Kernel

Deep Learning: BiLSTM


# Module Machine Learning (ML): E-Commerce Sentiment Analysis

## 📊 Perbandingan Performa Model

Dalam eksperimen ini, kami membandingkan tiga algoritma Machine Learning untuk menemukan model klasifikasi sentimen terbaik. Berikut adalah hasil evaluasi berdasarkan data pengujian:

| Model | Accuracy | AUC | Recall | Precision | F1-Score | Kappa | MCC | Waktu Latih (TT) |
|---|---|---|---|---|---|---|---|---|
| **LightGBM** | **0.9823** | **0.9984** | **0.9823** | **0.9823** | **0.9823** | **0.9734** | **0.9734** | **4.932 sec** |
| Logistic Regression | 0.9767 | 0.0000 | 0.9767 | 0.9768 | 0.9767 | 0.9650 | 0.9651 | 9.162 sec |
| SVM - Linear Kernel | 0.9756 | 0.0000 | 0.9756 | 0.9758 | 0.9756 | 0.9633 | 0.9634 | 10.952 sec |

### 🏆 Kesimpulan Pemilihan Model
Berdasarkan tabel di atas, **Light Gradient Boosting Machine (LightGBM)** terpilih sebagai model utama yang diimplementasikan pada aplikasi ini dengan alasan berikut:
1. **Performa Tertinggi:** Mengungguli algoritma lain dengan Akurasi dan F1-Score mencapai **98.23%**.
2. **Kecepatan Komputasi:** Sangat efisien, dengan waktu pelatihan tercepat (hanya 4.9 detik), mengalahkan SVM yang memakan waktu hampir 11 detik.
3. **Kualitas Klasifikasi (AUC):** Memiliki skor *Area Under Curve* (AUC) sebesar 0.9984, menunjukkan kemampuan model yang sangat baik dalam membedakan kelas sentimen positif dan negatif.


🔗 **Live Demo Aplikasi**
Model klasifikasi sentimen ini telah di-deploy menjadi aplikasi web interaktif. Silakan coba langsung aplikasinya di sini:
[Hugging Face Space - Deteksi Sentimen E-Commerce](https://huggingface.co/spaces/lidianat/deteksi-sentimen-ecommerce)

# Module Deep Learning (DL): E-Commerce Sentiment Analysis

Modul ini berisi implementasi arsitektur **Bidirectional LSTM (BiLSTM)** untuk tugas klasifikasi sentimen ulasan produk e-commerce berbahasa Indonesia.

## Arsitektur Model
Model dibangun dari awal (*from scratch*) menggunakan kerangka kerja PyTorch dengan spesifikasi sebagai berikut:
* **Tipe Model**: Bidirectional LSTM (BiLSTM)
* **Jumlah Lapisan**: 2 Lapis
* **Total Parameter**: 2.497.411 (Memenuhi kriteria ToR < 10 Juta parameter)
* **Ukuran Kosakata (Vocabulary)**: 1.003 kata unik

## Detail Dataset & Preprocessing
Data bersumber dari repositori Hugging Face (`AIbnuHibban/e-commerce-sentiment-bahasa-indonesia`).
* **Total Data Awal**: 21.840 baris
* **Data yang Digunakan (Sampel)**: 15.000 baris
* **Proporsi Pemisahan Data**:
  * Data Latih (Train): 12.000
  * Data Validasi (Val): 1.500
  * Data Uji (Test): 1.500
* **Pembersihan Teks (Preprocessing)**: Meliputi *lowercasing*, penghapusan tautan (URL/HTML), pembersihan karakter non-alfanumerik, serta normalisasi singkatan/bahasa gaul (*slang*) yang umum digunakan pada platform e-commerce.

## Hasil Pelatihan & Evaluasi
Pelatihan model dilakukan menggunakan perangkat CPU. Model menunjukkan kemampuan konvergensi yang sangat baik dan proses pelatihan dihentikan lebih awal (*early stopping*) pada Epoch ke-6 untuk mencegah *overfitting*.

* **Durasi Pelatihan**: ~3.7 menit
* **Best Validation Loss**: 0.0353
* **Akurasi (Test Set)**: 98.87%
* **F1 Macro**: 98.87%
* **F1 Weighted**: 98.87%

**Classification Report:**

| Kelas | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative** | 0.99 | 0.99 | 0.99 | 513 |
| **Neutral** | 0.99 | 0.99 | 0.99 | 473 |
| **Positive** | 0.98 | 0.99 | 0.99 | 514 |

## Tautan Penting
* **Hugging Face Spaces (Demo Interaktif)**: https://huggingface.co/spaces/lidianat/analisis-sentimen-ulasan-produk-ecommerce
