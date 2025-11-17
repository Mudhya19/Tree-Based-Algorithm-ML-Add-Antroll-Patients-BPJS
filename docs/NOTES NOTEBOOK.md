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

Pilih SATU dari tiga jalur proyek machine learning berikut: 
Jalur : Proyek Klasifikasi (Supervised Learning) 
• 
Tujuan: Memprediksi label kategori (diskrit). 
• 
Contoh Kasus: Prediksi churn pelanggan, deteksi spam email, analisis sentimen 
(positif/negatif), diagnosis medis (sakit/tidak), deteksi fraud. 

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

Lakukan langkah-langkah berikut dalam sebuah notebook jupyter :

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
Contoh Regresi: Linear Regression, Decision Tree Regressor, Random 
Forest Regressor 
▪ 
Contoh Klasifikasi: Logistic Regression, K-Nearest Neighbors, SVM, 
Random Forest Classifier 
▪ 
Contoh Clustering: K-Means, DBSCAN, Agglomerative Clustering 
5. Evaluasi Model: 
o Bandingkan performa model Anda menggunakan metrik evaluasi yang sesuai: 
▪ 
Regresi: MAE, RMSE, R-squared. 
▪ 
Klasifikasi: Accuracy, Precision, Recall, F1-Score, Confusion Matrix. 
▪ 
Clustering: Silhouette Score, Davies-Bouldin Index (atau analisis profil 
cluster secara kualitatif). 
o Berikan justifikasi model mana yang Anda pilih sebagai model terbaik 
berdasarkan skenario dan metrik Anda. 
6. Simpan Model: 
o Simpan (ekspor) model terbaik Anda (bersama dengan preprocessor seperti 
scaler/encoder) ke dalam sebuah file. (Contoh: model_terbaik.pkl menggunakan 
pickle atau joblib). 