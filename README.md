# Warteg-Run: Physics-Informed Machine Learning Pipeline

Warteg-Run adalah sebuah platform cerdas (End-to-End Machine Learning Pipeline) yang dirancang untuk memprediksi durasi aktivitas fisik (kardio/lari) yang dibutuhkan guna membakar kalori dari makanan yang dikonsumsi pengguna. Sistem ini tidak sekadar menggunakan kalkulator kalori statis, melainkan memanfaatkan model regresi cerdas untuk menyimulasikan perhitungan berdasarkan Body Mass Index (BMI), usia, dan batas aman kecepatan berlari pengguna.

## Latar Belakang Masalah
Masyarakat modern seringkali kesulitan menakar seberapa banyak olahraga yang dibutuhkan untuk mengimbangi asupan kalori yang padat. Kalkulator standar di pasaran umumnya memberikan rekomendasi waktu secara universal tanpa mempertimbangkan apakah pengguna tersebut tergolong atletis, memiliki obesitas, atau sudah berusia lanjut. Penggunaan batas kecepatan yang tidak tepat berpotensi membahayakan kesehatan jantung.

## Solusi dan Arsitektur Sistem
Warteg-Run memecahkan masalah tersebut dengan melakukan pengelompokan (grouping) otomatis berdasarkan profil fisik pengguna, sebelum model AI memprediksi waktu secara dinamis.

Arsitektur aplikasi ini dibagi menjadi dua komponen utama (Client-Server):
1. **Backend (FastAPI):** Bertindak sebagai "mesin pemikir" (AutoML Engine) yang menyediakan antarmuka REST API asinkron. API ini memuat model Machine Learning (.pkl), database nutrisi, serta endpoint untuk melakukan pelatihan algoritma secara real-time.
2. **Frontend (Vanilla HTML/CSS/JS):** Antarmuka klien dibangun dengan arsitektur Single Page Application (SPA) mengadopsi tren desain Glassmorphism. Visualisasi data tingkat lanjut (EDA dan Evaluasi Model) diproses menggunakan pustaka Plotly.js.

## Fitur Utama

- **Pipeline Overview:** Pemaparan arsitektur teknis dan landasan bisnis (Product Management) dari sistem yang dibangun.
- **Interactive Exploratory Data Analysis (EDA):** Pengguna dapat melakukan visualisasi silang antar variabel (seperti Berat Badan vs Kecepatan) secara real-time dalam bentuk Scatter Plot dan Histogram.
- **Algorithm Training & Evaluation:** Dashboard pengujian model mandiri. Pengguna dapat memilih jenis algoritma (Random Forest Regressor atau Linear Regression), mengatur hiperparameter, dan memilih fitur (variabel independen) untuk melihat bagaimana model beradaptasi.
- **Live Prediction Engine:** Fitur utama berbasis autocomplete pencarian makanan dari 1340+ dataset nutrisi. Sistem akan memberikan beberapa opsi pace (kecepatan lari) lengkap dengan prediksi menit yang dibutuhkan berdasarkan perhitungan AI.

## Dataset
Proyek ini mengintegrasikan `nutrition.csv`, sebuah dataset yang berisi informasi kalori dan makronutrisi dari 1340 hidangan. Untuk pelatihan regresi, data target waktu disintesis menggunakan rumus turunan Metabolic Equivalent of Task (MET) yang dipadukan dengan toleransi gangguan matematis (noise) guna melatih ketahanan model.

## Penjelasan Metrik Evaluasi
Karena pendekatan sistem ini menggunakan Regresi (memprediksi angka kontinu, bukan kelas/kategori), sistem dievaluasi dengan tiga metrik utama industri:

1. **R-Squared (R2):** Persentase varians pada variabel dependen (Waktu) yang dapat dijelaskan oleh variabel independen (Kalori, Berat, dsb). Semakin mendekati angka 1.0, semakin sempurna model mengenali pola data.
2. **MAE (Mean Absolute Error):** Menghitung rata-rata selisih absolut antara nilai yang diprediksi oleh AI dengan nilai sebenarnya. Jika nilai MAE adalah 2.0, artinya prediksi waktu lari dari AI rata-rata hanya meleset sebesar 2 menit dari kondisi nyata.
3. **RMSE (Root Mean Squared Error):** Mirip dengan MAE, namun metrik ini memberikan penalti yang jauh lebih besar terhadap error (tebakan meleset) yang bernilai ekstrim. Ini sangat penting untuk memastikan tidak ada prediksi waktu yang meleset terlalu jauh (outlier).

## Panduan Instalasi dan Penggunaan

**1. Persiapan Lingkungan (Environment)**
Sangat disarankan menggunakan virtual environment (misalnya Conda) untuk isolasi dependensi.
```bash
conda create -n warteg_run python=3.11
conda activate warteg_run
pip install fastapi uvicorn pandas scikit-learn pydantic joblib