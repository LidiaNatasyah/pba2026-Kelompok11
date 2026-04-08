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
LightGBM
Logistic Regression
SVM - Linear Kernel


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
