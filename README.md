Warteg-Run: Enterprise AI Runner Prediction Pipeline
Warteg-Run adalah aplikasi Full-stack AI yang memprediksi waktu lari optimal berdasarkan asupan kalori makanan (database nutrisi warteg) dan profil fisik pengguna (BMI, usia, berat badan). Proyek ini mengintegrasikan machine learning model dengan arsitektur modern untuk memberikan analisis kesehatan secara real-time.

Fitur Utama
Interactive EDA: Visualisasi distribusi data sintetis menggunakan Plotly.js langsung di browser.

Live Prediction: Prediksi waktu tempuh lari menggunakan model Random Forest atau Linear Regression yang telah di-deploy.

Dynamic Nutrition Database: Integrasi data nutrisi makanan populer untuk menghitung total beban kalori.

Health Analytics: Klasifikasi kategori fisik (Atletis, Menengah, Senior) berdasarkan BMI dan usia.

Live Model Training: Fitur simulasi training model secara langsung melalui API endpoint.

Tech Stack
Frontend

HTML5 dan CSS3: Menggunakan desain Glassmorphism untuk UI yang responsif.

JavaScript (Vanilla ES6+): Logika asinkronus untuk komunikasi dengan API.

Plotly.js: Library visualisasi data tingkat tinggi untuk grafik interaktif.

Vercel: Platform hosting untuk frontend yang terintegrasi dengan CD/CI.

Backend

Python 3.x: Bahasa utama untuk pemrosesan data dan machine learning.

FastAPI: Framework web berperforma tinggi yang mendukung konkurensi asinkronus.

Scikit-Learn: Library utama untuk implementasi algoritma Random Forest dan Linear Regression.

Pandas dan Numpy: Manipulasi data tabular dan operasi matriks.

Railway: Platform cloud untuk hosting backend dengan sistem otomatisasi port.

Arsitektur Proyek
Aplikasi ini menggunakan model Monorepo yang terbagi menjadi dua bagian utama:

/frontend: Berisi aset statis (HTML/CSS/JS) yang di-deploy ke Vercel.

/backend: Berisi API FastAPI, file model (.pkl), dan dataset (.csv) yang di-deploy ke Railway.

Instalasi Lokal
Clone Repositori:
git clone https://github.com/lewron135/Warteg-Run.git
cd Warteg-Run

Setup Backend:
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

Setup Frontend:
Buka file frontend/index.html dan pastikan API_URL mengarah ke http://127.0.0.1:8000.

Author
Josep Natanael Pasaribu

GitHub: https://github.com/lewron135
