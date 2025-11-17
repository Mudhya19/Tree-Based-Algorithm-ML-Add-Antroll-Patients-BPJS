Database Server :

host : 192.168.11.5
user : rsds_db
password : rsdsD4t4b4s3
database : rsds_db
port : 3306

Database Struktur Query :

pasien BPJS add antrol :

    SELECT
        rp.no_rawat,
        rp.tgl_registrasi,
        rp.jam_reg,
        rp.kd_dokter,
        d.nm_dokter,
        rp.no_rkm_medis,
        pas.nm_pasien,
        rp.kd_poli,
        p.nm_poli,
        rp.status_lanjut,
        rp.kd_pj,
        pj.png_jawab,
        mar.tanggal_periksa,
        mar.nomor_kartu,
        mar.nomor_referensi,
        mar.kodebooking,
        mar.jenis_kunjungan,
        mar.status_kirim,
        mar.keterangan,
        bs.USER
    FROM
        reg_periksa rp
        JOIN mlite_antrian_referensi mar ON rp.no_rkm_medis = mar.no_rkm_medis
        JOIN poliklinik p ON rp.kd_poli = p.kd_poli
        JOIN dokter d ON rp.kd_dokter = d.kd_dokter
        JOIN penjab pj ON rp.kd_pj = pj.kd_pj
        JOIN pasien pas ON rp.no_rkm_medis = pas.no_rkm_medis
        JOIN bridging_sep bs ON rp.no_rawat = bs.no_rawat
    WHERE
        rp.tgl_registrasi BETWEEN %s AND %s
        AND mar.tanggal_periksa BETWEEN %s AND %s
        AND rp.kd_poli NOT IN ('IGDK', 'HDL', 'BBL', 'IRM', '006', 'U0016')
        AND rp.status_lanjut NOT IN ('Ranap')
    ORDER BY
        rp.no_rawat;

mlite query logs :

    SELECT
        *
    FROM
        mlite_query_logs
    WHERE
        DATE(created_at) BETWEEN %s AND %s
    ORDER BY
        created_at DESC;

logic jikalau gagal dalam mengambil telah disediakan bpjs antrol.csv dalam folder database, buatkan path langsung dalam bpjs_add_antrol.py jika gagal terhubung ke database server

Pilih SATU dari tiga jalur proyek machine learning berikut:
Jalur : Proyek Klasifikasi (Supervised Learning)
•
Tujuan: “Analisis Komperenshif Identifikasi Pendaftaran Pasien BPJS Add Antroll
•
Kasus: Analisis Klasifikasi Pasien BPJS Add Antrol

Persyaratan Proyek
•
Definisi Masalah: Anda harus dapat menjelaskan dengan singkat masalah bisnis
(business case) atau tujuan dari pembuatan model Anda. (Misal: "Membantu tim
marketing menemukan pelanggan potensial" atau "Membantu manajer memprediksi
harga rumah").
•
Kompleksitas Dataset: Dataset yang dipilih harus cukup memadai untuk menunjukkan
proses preprocessing (idealnya memiliki campuran fitur numerik dan kategorikal)
dan bukan dataset "mainan" yang terlalu sederhana (seperti 'Iris' atau 'Titanic' dasar).

# Lakukan langkah-langkah berikut dalam sebuah notebook jupyter, tetapi fokus utama kerjakan dengan format yang sudah disediakan bpjs_add_antrol.py :

1. Definisi Masalah & Pemuatan Data:
   o Jelaskan secara singkat masalah yang ingin Anda selesaikan.
   o Muat dataset Anda dan jelaskan sumbernya.
2. Eksplorasi Data (EDA):
   o Lakukan analisis untuk memahami setiap fitur dan hubungannya dengan target
   (jika ada).
   o Visualisasikan wawasan kunci Anda.
3. Data Preparation & Preprocessing:
   o Tangani missing values dengan metode yang dapat dijustifikasi.
   o Lakukan encoding pada fitur-fitur kategorikal.
   o Lakukan scaling/normalization pada fitur numerik.
4. Pelatihan Model:
   o Bagi dataset menjadi data latih dan data uji (untuk Regresi/Klasifikasi).
   o Latih minimal 2 (dua) model machine learning yang relevan dengan jalur
   proyek Anda.
   ▪
   Klasifikasi: Tree-Based Algorithm menggunakan model Machine Learning (Decision Tree, Random Forest dan Gradient Boosting)
5. Evaluasi Model:
   o Bandingkan performa model Anda menggunakan metrik evaluasi yang sesuai:
   ▪
   Klasifikasi: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
   o Berikan justifikasi model mana yang Anda pilih sebagai model terbaik
   berdasarkan skenario dan metrik Anda.
6. Simpan Model:
   o Simpan (ekspor) model terbaik Anda (bersama dengan preprocessor seperti
   scaler/encoder) ke dalam sebuah file. (Contoh: model_terbaik.pkl menggunakan
   pickle atau joblib).

# buatkan notebook jupyter dengan nama bpjs_add_antrol.ipynb dalam folder notebooks dibawah ini berikan markdown berdasarkan 11 keterangan dibawah, karna saya ingin fokus source code lebih dulu dengan bpjs_add_antrol.py

1. Understanding Business: Memahami kebutuhan bisnis dan tujuan analisis
2. Data Understanding: Menjelajahi dan memahami struktur data
3. Data Preparation / Wrangling: Menyiapkan dan mengolah data
4. Data Cleaning: Membersihkan data dari ketidakkonsistenan
5. Explanatory Data Analysis (EDA Deskriptif): Analisis deskriptif awal
6. Exploratory Data Analysis (EDA Mendalam): Eksplorasi data secara mendalam
7. Data Preprocessing: Pra-pemrosesan data untuk modeling
8. Training Modeling : melatih model
9. Evaluation Modeling : evaluasi model
10. Save Model : menyimpan model terbaik
11. Insight & Conclusion: Penarikan kesimpulan dan rekomendasi
