# Panduan Deployment ke Streamlit Community Cloud

## Persiapan File untuk Deployment

Aplikasi ini memerlukan beberapa file model untuk dapat berfungsi secara penuh. Jika file-file ini tidak tersedia, aplikasi tetap akan berjalan tetapi fitur prediksi akan dinonaktifkan.

### File yang Dibutuhkan

Aplikasi ini memerlukan file-file berikut. Aplikasi akan mencoba mencari file di beberapa lokasi untuk fleksibilitas deployment:
- Model utama: `Gradient_Boosting_model.pkl` (dicari di folder `output/` dan root)
- Scaler: `scaler.pkl` (dicari di folder `output/` dan root)
- Label encoders: `label_encoders.pkl` (dicari di folder `output/` dan root)
- Dataset: `bpjs antrol.csv` (dicari di folder `database/` dan root)

1. `Gradient_Boosting_model.pkl` - Model machine learning utama
2. `scaler.pkl` - Scaler untuk normalisasi data
3. `label_encoders.pkl` - Encoder untuk variabel kategorikal

### Struktur Folder yang Diperlukan

```
your-app/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── output/
│   ├── Gradient_Boosting_model.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
└── database/
    └── bpjs antrol.csv
```

## Cara Mengunggah File ke Streamlit Community Cloud

### Metode 1: Menggunakan Git (Disarankan)

1. Clone repository aplikasi Anda ke lokal
2. Buat folder `output` jika belum ada
3. Tempatkan file model Anda ke dalam folder `output/`:
   - `Gradient_Boosting_model.pkl`
   - `scaler.pkl`
   - `label_encoders.pkl`
4. Commit dan push perubahan ke repository GitHub Anda
5. Deploy ke Streamlit Community Cloud seperti biasa

### Metode 2: Menggunakan GitHub UI

1. Buka repository GitHub Anda
2. Klik pada folder `output/`
3. Klik "Add file" → "Upload files"
4. Upload ketiga file model ke folder tersebut
5. Commit changes
6. Sinkronkan kembali deployment Anda di Streamlit Community Cloud

## Catatan Penting

- Pastikan ukuran total aplikasi Anda tidak melebihi batas 100MB (dikonfigurasi di `.streamlit/config.toml`)
- File model harus dalam format pickle (`.pkl`) yang dihasilkan dari `joblib.dump()`
- File model harus ditempatkan di folder `output/` dan dataset di folder `database/` dengan struktur sesuai yang ditunjukkan di atas
- Jika Anda melihat pesan "File model tidak ditemukan", berarti file-file di atas belum diunggah ke repository
- Aplikasi tetap dapat berjalan tanpa model, tetapi hanya akan menampilkan data analisis tanpa fitur prediksi

## Troubleshooting

Jika Anda masih mengalami masalah:

1. Pastikan nama file sesuai persis: `Gradient_Boosting_model.pkl`, `scaler.pkl`, `label_encoders.pkl`
2. Pastikan file berada di folder `output/` (case-sensitive)
3. Pastikan repository Anda tidak memiliki file `.gitignore` yang tidak sengaja mengabaikan file model
4. Verifikasi bahwa ukuran file tidak terlalu besar untuk GitHub (perhatikan bahwa GitHub memiliki batas 100MB per file)

## Fitur Aplikasi

Aplikasi ini menyediakan dua tab utama:
- **Prediksi Pendaftaran**: Memprediksi keberhasilan pendaftaran pasien BPJS
- **Data Pasien**: Menampilkan dan menganalisis data pasien BPJS

Jika model tidak tersedia, tab Prediksi Pendaftaran akan menampilkan pesan informasi dan menonaktifkan fitur prediksi.