# Development of the U-Net Model with Lightweight Temporal Attention Encoder (U-LTAE) for Estimating Aboveground Biomass Using Sentinel-2 Multi-Temporal Image Data

Repository ini berisi implementasi Deep Learning *Spatio-Temporal* untuk estimasi *Aboveground Biomass* (AGB) menggunakan model **U-TAE (U-Net with Temporal Attention Encoder)**. Proyek ini menggunakan citra satelit **Sentinel-2** (Time Series Bulanan) dan data referensi **ESA CCI AGB**.

## 🧠 Arsitektur Model

Model didasarkan pada arsitektur U-TAE  yang menggabungkan:
1.  **Spatial Encoder:** Memproses setiap frame waktu Sentinel-2 secara independen.
2.  **L-TAE (Lightweight Temporal Attention Encoder):** Mekanisme atensi efisien untuk mengompresi dimensi waktu ($T$) menjadi representasi fitur tunggal yang kaya informasi.
3.  **Decoder:** Mengembalikan resolusi spasial untuk menghasilkan peta regresi biomassa (pixel-wise).

## 📂 Struktur Folder

```text
biomass-utae-lampung/
├── data/               # Penyimpanan dataset (Raw & Processed)
├── src/
│   ├── data/           # Script download (GEE) & preprocessing
│   └── models/         # Definisi model L-TAE & U-TAE
├── train.py            # Script pelatihan utama
└── requirements.txt    # Dependensi Python
```

## 🚀 Cara Penggunaan

1. Instalasi
Pastikan Anda menggunakan Python 3.8+ dan GPU NVIDIA (Disarankan).
pip install -r requirements.txt
