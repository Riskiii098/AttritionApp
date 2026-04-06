---
title: AttritionApp
emoji: 📈
colorFrom: blue
colorTo: turquoise
sdk: docker
pinned: false
---

# Employee Attrition Intelligence System

Sistem komprehensif berbasis ***Machine Learning*** untuk memprediksi probabilitas keluarnya karyawan (resign/attrition) secara dini dan proaktif. Proyek ini memadukan kemampuan analitik dari Scikit-Learn dengan antarmuka web modern Bootstrap 5, ditenagai oleh *backend* Python Flask yang cepat, dan siap menyajikan repot *Business Intelligence* melalui Google Looker Studio.

## 🌟 Fitur Utama

- **Predictive ML Endpoint:** REST API *real-time* yang menerima data vektor *array* (JSON) dan melakukan skoring/prediksi menggunakan algoritma *Random Forest*.
- **Model Tracking & Persistence:** Menggunakan **MLflow** untuk pencatatan/logging performa (akurasi & parameter), dan **Joblib** untuk penyimpanan performa model sebagai `file.pkl` secara terstruktur demi memori yang efisien.
- **Executive BI Dashboard:** Halaman *dashboard layout* penuh (*full-width*) yang disiapkan untuk penanaman (*embed*) laporan analitis HR secara langsung dari Google Looker Studio.
- **Interactive UI/UX:** Antarmuka web yang bersih (*clean*), responsif, dan elegan menggunakan Bootstrap 5 bersama Javascript *Fetch API async* untuk memberikan pengalaman pengguna tanpa *loading page* yang patah-patah.
- **Containerized & Siap Produksi:** Dilengkapi dengan *Dockerfile* yang sangat ringan (*python slim image*) memastikan aplikasi *Machine Learning* berjalan persis sama di environment manapun.

## 🛠️ Stack & Tools

- **Backend Framework:** Python Flask 3.x
- **Machine Learning Core:** Scikit-Learn, Pandas, NumPy
- **MLOps Toolkit:** MLflow, Joblib
- **Frontend Presentation:** HTML5, Modern Javascript (Fetch API), Bootstrap 5.3
- **Data Visualization:** Google Looker Studio Embeds (`<iframe>` integrasi)
- **Deployment & Scaling:** Docker

## 📂 Struktur Proyek

```text
AttritionProject/
├── app/
│   ├── app.py              # Main Flask application dan registrasi REST endpoints
│   └── model_loader.py     # Loader otomatis dengan joblib dan track training via MLflow
├── data/
│   └── employee_data_final.csv  # Dataset mentah 
├── model/                  # Cache penyimpanan parameter dan arsitektur mesin `.pkl` via Joblib
├── mlruns/                 # Direktori internal log Tracking model oleh MLflow (otomatis terbuat)
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

4. **Akses Dashboard MLflow (Opsional):**
   ```bash
   mlflow ui -p 5001
   ```

5. **Akses ke Aplikasi:** 
   Buka aplikasi peramban (browser) anda, lalu meluncur ke `http://127.0.0.1:5000/` untuk Flask. Bila ingin mengecek logging model, akses `http://127.0.0.1:5001/` untuk MLflow.

---

### Opsi B: Berjalan Menggunakan Docker (Isolated Environment)

Pastikan aplikasi **Docker Desktop** atau *daemon* Docker Anda sudah aktif (*Running*).

1. **Jadikan Mesin Aplikasi kedalam Image:**
   ```bash
   docker build -t attrition-app .
   ```
2. **Buka Jalur Eksekusi Container di Background:**
   ```bash
   docker run -d -p 5000:5000 --name running-attrition attrition-app
   ```
3. **Mulai Pengujian Simulasi Web:** 
   Sekarang Anda bisa mengakses `http://localhost:5000/` secara virtual, terisolasi penuh dari operating system Windows/Mac Anda!

---
> *Dikembangkan khusus untuk kebutuhan analitik Employee Attrition modern.*
