---
title: AttritionApp
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Employee Attrition Intelligence System

Sistem komprehensif berbasis ***Machine Learning*** untuk memprediksi probabilitas keluarnya karyawan (resign/attrition) secara dini dan proaktif. Proyek ini memadukan kemampuan analitik dari Scikit-Learn dengan antarmuka web modern Bootstrap 5, ditenagai oleh *backend* Python Flask yang cepat, dan siap menyajikan repot *Business Intelligence* melalui Google Looker Studio.

## 🌟 Fitur Utama

- **Adaptive Risk Thresholding:** Mengimplementasikan ambang batas (threshold) **0.40** untuk meningkatkan sensitivitas deteksi dini (*High Recall*), memastikan resiko sekecil apapun teridentifikasi sebelum terlambat.
- **Enterprise Experiment Tracking:** Menggunakan **DagsHub** yang terintegrasi dengan **MLflow** untuk pencatatan performa model secara terpusat (Cloud), memungkinkan kolaborasi tim data yang lebih profesional.
- **Model Persistence:** Memanfaatkan **Joblib** untuk penyimpanan model `.pkl` dan pemuatan fitur secara instan demi efisiensi memori.
- **Executive BI Dashboard:** Halaman *dashboard layout* penuh (*full-width*) yang disiapkan untuk penanaman (*embed*) laporan analitis HR secara langsung dari Google Looker Studio.
- **Containerized & Mobile Ready:** Antarmuka Bootstrap 5 yang responsif dan dukungan *Dockerfile* (Python slim image) untuk deployment yang konsisten di platform cloud seperti Hugging Face Spaces.

## 🛠️ Stack & Tools

- **Backend Framework:** Python Flask 3.x
- **Machine Learning Core:** Scikit-Learn, Pandas, NumPy
- **MLOps & Tracking:** DagsHub, MLflow, Joblib
- **Frontend Presentation:** HTML5, Modern Javascript (Fetch API), Bootstrap 5.3
- **Data Visualization:** Google Looker Studio Embeds (`<iframe>` integrasi)
- **Deployment & Scaling:** Docker

### Persyaratan Environment Variabel Berbasis Cloud
Agar sistem pelacakan (Tracking) metrik performa ke **DagsHub / MLflow** berfungsi sempurna, Anda _diwajibkan_ menyediakan kredensial token.
Bagi pengguna lokal (Windows/MacOS), siapkan file `.env` atau atur via *System Environment Variables*. Jika berjalan di **Hugging Face Spaces**, tambahkan ini ke menu **Settings > Variables and secrets**:

*   **`DAGSHUB_TOKEN`**: Berisi API Token dari akun DagsHub Anda.

_Jika variabel ini tidak ada, aplikasi tetap berjalan lancar namun proses latih (training) tidak akan dicatat di DagsHub (Hanya jalan secara offline/lokal)._

## 📂 Struktur Proyek

```text
AttritionProject/
├── .github/
│   └── workflows/
│       └── huggingface-sync.yml # Otomasi Deployment CI/CD ke Hugging Face
├── app/
│   ├── app.py              # Main Flask application & Business Logic (Threshold 0.40)
│   ├── model_loader.py     # Pure MLflow HTTP integration with fast on-the-fly training
│   └── utils.py            # Placeholder untuk fungsionalitas pendukung di masa depan
├── data/
│   └── employee_data_final.csv  # Dataset mentah 
├── model/                  # Hasil training model `.pkl` (Di-ignore oleh git untuk menghindari isu LFS)
├── mlflow.db               # Database SQLite lokal
├── mlruns/                 # Direktori internal log (Otomatis terbuat jika offline)
├── frontend/
│   ├── static/
│   │   ├── script.js       # Asynchronous HTTP POST request handler for ML endpoint
│   │   └── style.css       # Custom cascading stylesheets
│   └── templates/
│       ├── index.html      # Landing Page utama
│       ├── dashboard.html  # Halaman integrasi BI Dashboard HR
│       └── predict.html    # Form interaktif eksekusi prediksi ML 
├── Dockerfile              # Setup environment Docker Container
├── requirements.txt        # Python dependency tracker
└── README.md               # Dokumentasi resmi ini
```

## 🚀 Cara Menjalankan Project

### Opsi A: Berjalan Secara Lokal (Native Python)

1. **Persiapan Direktori:**
   Buka terminal/CMD Anda, posisikan root terminal ke dalam path *startup* ini.
   
2. **Install Dependensi Utama:**
   Sangat disarankan memakai `virtualenv` sebelumnya. Gunakan perintah ini untuk menginstal seluruh utilitas *library*:
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan Aplikasi Web Flask:**
   Bagi pengguna Windows, gunakan *launcher* `py` untuk mencegah *error* bawaan sistem.
   ```bash
   py app/app.py
   ```
   *(Catatan: Model Machine Learning akan otomatis dilatih (on-the-fly) pertama kali lalu log model masuk ke MLflow dan diekspor oleh joblib, tidak perlu step training manual).*

4. **Akses Dashboard Tracking (DagsHub/MLflow):**
   Aplikasi secara otomatis mencatat setiap eksperimen ke repository DagsHub. Anda bisa memantau kurva akurasi dan parameter model melalui UI DagsHub atau menjalankan secara lokal:
   ```bash
   mlflow ui -p 5001
   ```

5. **Akses ke Aplikasi:** 
   Buka peramban anda ke `http://127.0.0.1:7860/`. Aplikasi akan menampilkan prediksi berbasis threshold risiko yang telah dioptimasi.

---

### Opsi B: Berjalan Menggunakan Docker (Isolated Environment)

Pastikan aplikasi **Docker Desktop** atau *daemon* Docker Anda sudah aktif (*Running*).

1. **Jadikan Mesin Aplikasi kedalam Image:**
   ```bash
   docker build -t attrition-app .
   ```
2. **Buka Jalur Eksekusi Container di Background:**
   ```bash
   docker run -d -p 7860:7860 --name running-attrition attrition-app
   ```
3. **Mulai Pengujian Simulasi Web:** 
   Sekarang Anda bisa mengakses `http://localhost:7860/` secara virtual, terisolasi penuh dari operating system Windows/Mac Anda!

---
> *Dikembangkan khusus untuk kebutuhan analitik Employee Attrition modern.*
