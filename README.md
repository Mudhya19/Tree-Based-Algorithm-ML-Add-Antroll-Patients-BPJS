# Tree-Based Algorithm ML Analysis for BPJS Antrol Patients

Aplikasi ini merupakan implementasi dari model machine learning berbasis algoritma tree untuk menganalisis dan memprediksi keberhasilan pendaftaran pasien BPJS melalui sistem antrol (antrian online).

## Deskripsi Proyek

Proyek ini bertujuan untuk:
- Menganalisis pola keberhasilan dan kegagalan pendaftaran pasien BPJS melalui dua kanal yaitu Anjungan Pendaftaran Mandiri (APM) dan aplikasi Mobile JKN
- Mengidentifikasi faktor-faktor yang mempengaruhi keberhasilan pendaftaran
- Memberikan prediksi apakah pendaftaran pasien akan berhasil atau gagal berdasarkan fitur-fitur tertentu

## Fitur Aplikasi

- Input pengguna melalui sidebar untuk parameter-parameter penting
- Prediksi keberhasilan pendaftaran BPJS
- Tampilan probabilitas keberhasilan dan kegagalan
- Antarmuka yang ramah pengguna

## Teknologi yang Digunakan

- Python
- Streamlit (untuk antarmuka web)
- Scikit-learn (untuk algoritma machine learning)
- Pandas dan NumPy (untuk pengolahan data)
- Joblib (untuk serialisasi model)

## Fitur-fitur Input

Model menggunakan fitur-fitur berikut untuk prediksi:
- Status Lanjut (Ralan/Ranap)
- Kode Penjamin (UMUM/JKN)
- Cara Bayar (UMUM/JKN)
- Jenis Kunjungan (1/2/3)
- Nama Poli
- User (APM/Mobile JKN)
- Bulan Registrasi
- Hari Registrasi

## Cara Menjalankan Aplikasi Lokal

1. Clone repository ini
2. Install dependensi: `pip install -r requirements.txt`
3. Pastikan file model (Gradient_Boosting_model.pkl, scaler.pkl, label_encoders.pkl) berada di folder output/
4. Jalankan aplikasi: `streamlit run app.py`

## Deployment ke Streamlit Community Cloud

Aplikasi ini siap untuk di-deploy ke Streamlit Community Cloud:

1. Fork repository ini ke GitHub Anda
2. Buka https://share.streamlit.io/
3. Masukkan detail repository Anda
4. Streamlit akan otomatis menginstal dependensi dari requirements.txt
5. Aplikasi akan live dan dapat diakses secara publik

## Struktur Proyek

```
├── app.py                 # Aplikasi Streamlit utama
├── requirements.txt       # Dependensi Python
├── .streamlit/
│   └── config.toml       # Konfigurasi tampilan Streamlit
├── output/
│   ├── Gradient_Boosting_model.pkl    # Model machine learning
│   ├── scaler.pkl                   # Scaler untuk preprocessing
│   └── label_encoders.pkl           # Label encoders untuk variabel kategorikal
├── notebooks/
│   └── colaboratory_tree_based.ipynb # Notebook untuk training model
└── database/
    └── bpjs antrol.csv              # Dataset asli
```

## Catatan untuk Deployment

Pastikan bahwa file model (di folder output/) telah disertakan dalam repository agar aplikasi dapat berjalan dengan benar di lingkungan deployment.
