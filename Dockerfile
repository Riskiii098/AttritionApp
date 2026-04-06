# Menggunakan base image Python resmi versi slim agar ukuran image lebih ringan
FROM python:3.10-slim

# Mencegah Python menulis file .pyc ke disk dan mem-buffer pesan logging (best practice container)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Menentukan Working Directory di dalam container
WORKDIR /app

# Copy file requirements.txt terlebih dahulu (untuk memanfaatkan cache layer Docker
# jika requirements tidak berubah, proses build akan jauh lebih cepat)
COPY requirements.txt .

# Install semua dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh file dan direktori project saat ini ke dalam folder /app container
COPY . .

# Buka akses port 7860 sesuai konfigurasi default Hugging Face Spaces
EXPOSE 7860

# Command spesifik untuk mengeksekusi web server level production
# Menggunakan Gunicorn karena image berbasis Linux Debian
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "app.app:app"]
