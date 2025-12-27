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
│
├── data/
│   ├── raw/                  # Tempat file .tif hasil download GEE
│   └── processed/            # Tempat dataset siap pakai (.npz)
│
├── src/
│   ├── __init__.py
│   ├── data/                 # Modul pengolahan data
│   │   ├── __init__.py
│   │   ├── download_gee.py   # Script download dari Google Earth Engine
│   │   ├── preprocess.py     # Script potong patch (tiling) & kompresi
│   │   └── dataset.py        # PyTorch Dataset Loader
│   │
│   └── models/               # Arsitektur Neural Network
│       ├── __init__.py
│       ├── ltae.py           # Modul Lightweight Temporal Attention
│       └── utae.py           # Modul U-TAE untuk Regresi
│
├── train.py                  # Script utama pelatihan (Main Loop)
├── requirements.txt          # Daftar dependensi
└── README.md                 # Dokumentasi Proyek
```

## 🚀 Cara Penggunaan

### 1. Instalasi
Pastikan Anda menggunakan Python 3.8+ dan GPU NVIDIA (Disarankan).
```bash
pip install -r requirements.txt
```
### 2. Download Data
Pastikan Anda memiliki akun Google Earth Engine. Script ini akan mengunduh data Sentinel-2 dan ESA CCI AGB untuk wilayah Lampung dan Kalimantan Selatan.
```bash
cd src/data
python download_gee.py
```


## 📄 References

[1] V. S. F. Garnot and L. Landrieu, "Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks," in *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*, Montreal, QC, Canada, 2021, pp. 4872-4881. [cite: 5, 6, 7, 39]

[2] V. S. F. Garnot and L. Landrieu, "Lightweight Temporal Self-Attention for Classifying Satellite Images Time Series," in *Advanced Analytics and Learning on Temporal Data (AALTD)*, 2020. [cite: 456, 457, 465]

[3] Priamus Lab, "ReUse: Reusing the Derived Features of the Pretext Task for Semantic Segmentation," GitHub Repository, 2024. [Online]. Available: https://github.com/priamus-lab/ReUse.
