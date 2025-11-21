import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Aplikasi ini dibuat untuk menganalisis dan memprediksi keberhasilan pendaftaran pasien BPJS
# melalui kanal APM (Anjungan Pendaftaran Mandiri) dan Mobile JKN menggunakan algoritma Tree-Based

# Set page configuration
st.set_page_config(
    page_title="Aplication Tree-Based Algorithm ML Analysis Prediction for BPJS Antrol Patients",
    page_icon="🏥",
    layout="wide"
)

# Create tabs for different sections
tab1, tab2 = st.tabs(["Prediksi Pendaftaran", "Data Pasien"])

with tab1:
    # Title
    st.title("Aplication Tree-Based Algorithm ML Analysis Prediction for BPJS Antrol Patients")

    # Load model and preprocessing objects
    @st.cache_resource
    def load_model():
        import os
        # Coba beberapa pendekatan untuk membaca file model dengan prioritas di root folder
        model_paths = [
            './Gradient_Boosting_model.pkl',  # Prioritaskan root folder
            'Gradient_Boosting_model.pkl',
            './output/Gradient_Boosting_model.pkl',  # Lalu coba folder output
            'output/Gradient_Boosting_model.pkl',
            '../output/Gradient_Boosting_model.pkl',
            '../../output/Gradient_Boosting_model.pkl',
            '../../../output/Gradient_Boosting_model.pkl'
        ]
        scaler_paths = [
            './scaler.pkl',  # Prioritaskan root folder
            'scaler.pkl',
            './output/scaler.pkl',  # Lalu coba folder output
            'output/scaler.pkl',
            '../output/scaler.pkl',
            '../../output/scaler.pkl',
            '../../../output/scaler.pkl'
        ]
        label_encoder_paths = [
            './label_encoders.pkl',  # Prioritaskan root folder
            'label_encoders.pkl',
            './output/label_encoders.pkl', # Lalu coba folder output
            'output/label_encoders.pkl',
            '../output/label_encoders.pkl',
            '../../output/label_encoders.pkl',
            '../../../output/label_encoders.pkl'
        ]
        
        model = None
        scaler = None
        label_encoders = None
        
        # Coba load model
        for path in model_paths:
            try:
                if os.path.exists(path):
                    model = joblib.load(path)
                    break
            except Exception as e:
                continue  # Lanjutkan ke path berikutnya jika terjadi error
        
        # Coba load scaler
        for path in scaler_paths:
            try:
                if os.path.exists(path):
                    scaler = joblib.load(path)
                    break
            except Exception as e:
                continue  # Lanjutkan ke path berikutnya jika terjadi error
        
        # Coba load label encoders
        for path in label_encoder_paths:
            try:
                if os.path.exists(path):
                    label_encoders = joblib.load(path)
                    break
            except Exception as e:
                continue  # Lanjutkan ke path berikutnya jika terjadi error
        
        if model is None or scaler is None or label_encoders is None:
            # Tampilkan informasi debugging di lingkungan deployment
            missing_files = []
            if model is None: missing_files.append("Gradient_Boosting_model.pkl")
            if scaler is None: missing_files.append("scaler.pkl")
            if label_encoders is None: missing_files.append("label_encoders.pkl")
            raise FileNotFoundError(f"File berikut tidak ditemukan atau tidak dapat dibaca: {', '.join(missing_files)}")
        
        return model, scaler, label_encoders

    try:
        model, scaler, label_encoders = load_model()
        model_loaded = True
    except FileNotFoundError:
        st.warning("File model tidak ditemukan. Fitur prediksi akan dinonaktifkan.")
        st.info("Untuk menggunakan fitur prediksi, unggah file model ke folder 'output': 'Gradient_Boosting_model.pkl', 'scaler.pkl', 'label_encoders.pkl'")
        model_loaded = False
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {str(e)}")
        model_loaded = False

    # Create sidebar for input
    st.sidebar.header("Input Parameters")

    # Define the feature columns
    feature_columns = ['status_lanjut', 'kd_pj', 'png_jawab', 'jenis_kunjungan', 'nm_poli', 'USER', 'bulan_registrasi', 'hari_registrasi']

    # Get unique values from the dataset based on notebook analysis
    # === UPDATED OPTIONS FROM REAL DATA ===

    status_lanjut_options = ['Ralan', 'Ranap']

    kd_pj_options = [
        'BPJS Kesehatan',
        'DPP',
        'UMUM'
    ]

    png_jawab_options = [
        'BPJS Kesehatan',
        'DPP',
        'UMUM'
    ]

    jenis_kunjungan_options = ['1', '2', '3', '4']

    nm_poli_options = [
        'KLINIK ANAK',
        'KLINIK BEDAH',
        'KLINIK DOTS',
        'KLINIK GIGI SPESIALIS',
        'KLINIK JANTUNG',
        'KLINIK JIWA',
        'KLINIK KULIT & KELAMIN',
        'KLINIK MATA',
        'KLINIK OBSTETRI/GYN.',
        'KLINIK PARU',
        'KLINIK PENYAKIT DALAM',
        'KLINIK SARAF',
        'KLINIK THALASEMIA',
        'KLINIK THT'
    ]

    user_options = [
        'Admin Utama',
        'SEP Mandiri',
        'Petugas'
    ]

    # Input widgets for each feature
    status_lanjut = st.sidebar.selectbox("Status Lanjut", options=status_lanjut_options)
    kd_pj = st.sidebar.selectbox("Kode Penjamin", options=kd_pj_options)
    png_jawab = st.sidebar.selectbox("Cara Bayar", options=png_jawab_options)
    jenis_kunjungan = st.sidebar.selectbox("Jenis Kunjungan", options=jenis_kunjungan_options)
    nm_poli = st.sidebar.selectbox("Nama Poli", options=nm_poli_options)
    user = st.sidebar.selectbox("Registration Channel", options=user_options)

    # Date-related inputs
    bulan_registrasi = st.sidebar.slider("Bulan Registrasi", min_value=1, max_value=12, value=datetime.now().month)
    hari_registrasi = st.sidebar.slider("Hari Registrasi (1=Senin, 6=Sabtu)", min_value=1, max_value=6, value=datetime.now().weekday())

    # Create a dataframe with the input values
    input_data = pd.DataFrame({
        'status_lanjut': [status_lanjut],
        'kd_pj': [kd_pj],
        'png_jawab': [png_jawab],
        'jenis_kunjungan': [jenis_kunjungan],
        'nm_poli': [nm_poli],
        'USER': [user],
        'bulan_registrasi': [bulan_registrasi],
        'hari_registrasi': [hari_registrasi]
    })

    # Apply label encoding to categorical features only if model is loaded
    if model_loaded:
        for feature in ['status_lanjut', 'kd_pj', 'png_jawab', 'jenis_kunjungan', 'nm_poli', 'USER']:
            if feature in label_encoders:
                try:
                    input_data[feature] = label_encoders[feature].transform(input_data[feature].astype(str))
                except ValueError:
                    # If a new/unseen value is encountered, use the most frequent label
                    most_frequent_label = label_encoders[feature].classes_[0]
                    input_data[feature] = label_encoders[feature].transform([most_frequent_label])

    # Ensure all columns are in the correct order as expected by the model
    input_data = input_data[feature_columns]

    # Scale the features only if model is loaded
    if model_loaded:
        input_data_scaled = scaler.transform(input_data)

    # Prediction button
    if st.sidebar.button("Prediksi", key="predict"):
        if model_loaded:
            # Make prediction
            with st.spinner('Sedang memproses prediksi...'):
                prediction = model.predict(input_data_scaled)[0]
                prediction_proba = model.predict_proba(input_data_scaled)[0]
            
            # Display results
            st.subheader("Hasil Prediksi")
            
            if prediction == 1:
                st.success("Prediksi: Pendaftaran Berhasil")
                st.write("Probabilitas keberhasilan pendaftaran: {:.2%}".format(prediction_proba[1]))
            else:
                st.error("Prediksi: Pendaftaran Gagal")
                st.write("Probabilitas kegagalan pendaftaran: {:.2%}".format(prediction_proba[0]))
            
            st.subheader("Probabilitas Detil")
            st.write(f"Probabilitas Pendaftaran Gagal: {prediction_proba[0]:.2%}")
            st.write(f"Probabilitas Pendaftaran Berhasil: {prediction_proba[1]:.2%}")
        else:
            st.warning("Fitur prediksi tidak tersedia karena model belum dimuat. Silakan unggah file model yang diperlukan.")

    # Display model information
    st.subheader("Tentang Model")
    st.write("""
    Aplikasi ini menggunakan model Gradient Boosting yang telah dilatih
    untuk memprediksi keberhasilan pendaftaran pasien BPJS melalui
    kanal APM (Anjungan Pendaftaran Mandiri) dan Mobile JKN.
    """)

    # Display analysis results description with expander
    with st.expander("Deskripsi Lengkap Analisis"):
        st.write("""
        **Proyek Klasifikasi (Supervised Learning)**
        
        Tujuan: "Analisis Komperenshif Identifikasi Pendaftaran Pasien BPJS Add Antroll"

        Kasus: Analisis Klasifikasi Pasien BPJS Add Antrol
        
        **Persyaratan Proyek:**
        - Definisi Masalah: Membantu manajemen rumah sakit memahami pola pendaftaran pasien BPJS dan memprediksi jenis pembayaran atau status kunjungan pasien berdasarkan data pendaftaran.
        - Kompleksitas Dataset: Dataset yang diproses memiliki campuran fitur numerik dan kategorikal serta menunjukkan proses preprocessing yang kompleks.
        
        **Permasalahan Utama:**
        Analisis pola keberhasilan dan kegagalan pendaftaran pasien BPJS melalui dua kanal yaitu Anjungan Pendaftaran Mandiri (APM) dan aplikasi Mobile JKN. Melalui pendekatan analisis log dan metode clustering dalam penerapan machine learning dengan menggunakan Tree-Based Algorithm: Decision Tree, Random Forest dan Gradient Boosting. bertujuan untuk mengidentifikasi faktor-faktor penyebab dan memberikan rekomendasi peningkatan efektivitas pelayanan digital rumah sakit.
        
        Secara tidak optimalnya proses pendaftaran pasien BPJS pada kanal APM dan Mobile JKN karena adanya variasi tingkat keberhasilan dan kegagalan pengiriman data (status_kirim) yang menunjukkan pola berbeda, namun belum dianalisis secara komprehensif.
        
        **Kebutuhan Bisnis:**
        1. Memahami faktor-faktor yang mempengaruhi keberhasilan pendaftaran BPJS melalui APM dan Mobile JKN
        2. Mengidentifikasi pola kegagalan pendaftaran untuk perbaikan sistem
        3. Memberikan rekomendasi untuk meningkatkan efektivitas layanan digital rumah sakit
        4. Membantu manajemen dalam pengambilan keputusan terkait layanan pendaftaran pasien BPJS
        
        **Tujuan Analisis:**
        - Mengklasifikasikan keberhasilan/kegagalan pendaftaran pasien BPJS berdasarkan berbagai faktor
        - Menganalisis perbedaan pola pendaftaran antara kanal APM dan Mobile JKN
        - Mengidentifikasi variabel-variabel penting yang mempengaruhi keberhasilan pendaftaran
        - Membangun model prediktif untuk membantu pengambilan keputusan
        
        **Temuan Utama:**
        Berdasarkan analisis komprehensif terhadap data pendaftaran pasien BPJS, berikut adalah temuan utama:
        
        1. **Faktor Penting dalam Keberhasilan Pendaftaran**:
           - Jenis penjamin (BPJS Kesehatan, DPP, UMUM) mempengaruhi keberhasilan pendaftaran
           - Jenis kunjungan (1, 2, 3, 4) memiliki pengaruh signifikan
           - Nama poli tujuan menunjukkan pola keberhasilan yang berbeda
           - Kanal pendaftaran (APM vs Mobile JKN) menunjukkan perbedaan tingkat keberhasilan
        
        2. **Pola Pendaftaran Berdasarkan Kanal**:
           - APM (Anjungan Pendaftaran Mandiri) dan Mobile JKN menunjukkan pola keberhasilan yang berbeda
           - Beberapa poli memiliki tingkat keberhasilan lebih tinggi melalui kanal tertentu
           - Waktu pendaftaran (hari dan bulan) mempengaruhi keberhasilan proses
        
        3. **Rekomendasi Bisnis**:
           - Fokus pada faktor-faktor yang paling mempengaruhi keberhasilan pendaftaran
           - Implementasi monitoring untuk kanal pendaftaran dengan tingkat kegagalan lebih tinggi
           - Penyesuaian strategi pelayanan berdasarkan pola yang teridentifikasi
           - Pengembangan sistem pencegahan kegagalan berdasarkan prediksi model
        """)

    # Display input values
    st.subheader("Parameter Input")
    # Tampilkan input data hanya jika model tidak dimuat untuk memberikan konteks
    if not model_loaded:
        st.write("Parameter yang dimasukkan akan digunakan untuk prediksi ketika model telah dimuat")
    st.write(input_data)

with tab2:
    st.header("Data Pasien BPJS Antrol")
    
    # Load the CSV file
    @st.cache_data
    def load_data():
        import os
        # Coba beberapa pendekatan untuk membaca file dataset dengan prioritas di root folder
        dataset_paths = [
            './bpjs antrol.csv', # Prioritaskan root folder
            'bpjs antrol.csv',
            './database/bpjs antrol.csv',  # Lalu coba folder database
            'database/bpjs antrol.csv',
            '../database/bpjs antrol.csv',
            '../../database/bpjs antrol.csv',
            '../../../database/bpjs antrol.csv'
        ]
        
        df = None
        for path in dataset_paths:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    break
            except Exception as e:
                continue  # Lanjutkan ke path berikutnya jika terjadi error
        
        if df is None:
            raise FileNotFoundError("File dataset 'bpjs antrol.csv' tidak ditemukan di lokasi yang diharapkan")
        
        # Hapus data duplikat berdasarkan kolom no_rawat jika kolom tersebut ada
        if 'no_rawat' in df.columns:
            initial_count = len(df)
            df = df.drop_duplicates(subset=['no_rawat'])
            final_count = len(df)
            if initial_count != final_count:
                st.info(f"Data duplikat ditemukan dan dihapus: {initial_count - final_count} baris dihapus berdasarkan kolom 'no_rawat'")
        
        # Convert date columns to datetime if they exist
        date_columns = ['tgl_registrasi', 'tanggal_periksa']  # Common date columns in the dataset
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("File dataset tidak ditemukan. Pastikan file 'bpjs antrol.csv' telah diunggah ke folder 'database'.")
        st.info("Untuk deployment ke Streamlit Community Cloud, unggah file dataset Anda ke folder 'database' sebelum deployment.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat dataset: {str(e)}")
        st.stop()
    
    # Buat layout dengan expander untuk mengorganisir informasi
    with st.expander("Informasi Dasar Dataset"):
        # Display basic info about the dataset
        st.subheader("Statistik Dasar Dataset")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Jumlah Baris", df.shape[0])
        col2.metric("Jumlah Kolom", df.shape[1])
        col3.metric("Jumlah Missing Values", int(df.isnull().sum().sum()))
        # Tampilkan jumlah data unik jika kolom no_rawat ada
        if 'no_rawat' in df.columns:
            col4.metric("Jumlah Data Unik (no_rawat)", df['no_rawat'].nunique())
        else:
            col4.metric("Kolom no_rawat", "Tidak Tersedia")
    
    # Identify date columns in the dataset
    date_columns = [col for col in df.columns if 'tgl' in col.lower() or 'tanggal' in col.lower() or 'date' in col.lower()]
    
    # Inisialisasi filtered_df sebelum digunakan
    if date_columns:
        selected_date_col = st.selectbox("Pilih kolom tanggal untuk filter", date_columns)
        if selected_date_col:
            # Convert to datetime if not already
            date_series = pd.to_datetime(df[selected_date_col], errors='coerce')
            df[selected_date_col] = date_series
            
            # Get min and max dates
            min_date = date_series.min()
            max_date = date_series.max()
            
            if pd.notna(min_date) and pd.notna(max_date):
                # Date range selector
                date_range = st.date_input(
                    f"Pilih rentang tanggal untuk kolom {selected_date_col}",
                    value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date()
                )
                
                # Filter the data based on selected date range
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    # Convert to datetime for comparison
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(end_date)
                    
                    # Filter the dataframe
                    mask = (date_series >= start_date) & (date_series <= end_date)
                    filtered_df = df[mask]
                    
                    st.success(f"Data difilter dari {start_date.date()} hingga {end_date.date()}. "
                              f"Menampilkan {filtered_df.shape[0]} dari {df.shape[0]} baris.")
                else:
                    filtered_df = df
            else:
                filtered_df = df
                st.warning(f"Tidak ada data tanggal valid dalam kolom {selected_date_col}")
    else:
        # If no date columns found, use the original dataframe
        filtered_df = df
        st.info("Tidak ditemukan kolom tanggal dalam dataset. Menampilkan semua data.")
    
    # Tampilkan informasi tentang struktur data
    with st.expander("Struktur Data"):
        # Show data types
        st.subheader("Tipe Data Kolom")
        # Konversi dtypes ke format string untuk menghindari masalah Arrow
        dtypes_str = filtered_df.dtypes.astype(str)
        st.write(dtypes_str)
    
    # Tampilkan pratinjau data
    with st.expander("Pratinjau Data"):
        # Show data preview
        st.subheader("Pratinjau Data")
        rows_to_show = st.slider("Jumlah baris untuk ditampilkan", min_value=5, max_value=50, value=10)
        df_preview = filtered_df.head(rows_to_show).copy()
        # Konversi data ke format yang kompatibel dengan Arrow
        df_preview_arrow = df_preview.copy()
        for col in df_preview_arrow.columns:
            if df_preview_arrow[col].dtype == 'object':
                df_preview_arrow[col] = df_preview_arrow[col].astype(str)
        st.table(df_preview_arrow)
    
    # Tampilkan filter kolom
    with st.expander("Filter Kolom"):
        # Allow user to select columns to display
        st.subheader("Filter Kolom")
        selected_columns = st.multiselect("Pilih kolom untuk ditampilkan", filtered_df.columns.tolist(), default=filtered_df.columns[:5].tolist())
        if selected_columns:
            df_selected = filtered_df[selected_columns].head(20).copy()
            # Konversi data ke format yang kompatibel dengan Arrow
            df_selected_arrow = df_selected.copy()
            for col in df_selected_arrow.columns:
                if df_selected_arrow[col].dtype == 'object':
                    df_selected_arrow[col] = df_selected_arrow[col].astype(str)
            st.table(df_selected_arrow)
    
    # Tampilkan statistik deskriptif
    with st.expander("Statistik Deskriptif"):
        # Show basic statistics
        st.subheader("Statistik Deskriptif")
        # Only include numeric columns for describe to avoid issues
        numeric_df = filtered_df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            # Konversi describe ke format string untuk menghindari masalah Arrow
            st.write(numeric_df.describe().astype(str))
        else:
            st.info("Tidak ada kolom numerik untuk ditampilkan dalam statistik deskriptif.")
    
    # Tampilkan informasi unik per kolom
    with st.expander("Analisis Kolom Kategorikal"):
        # Show unique values for categorical columns
        st.subheader("Nilai Unik pada Kolom Kategorikal")
        categorical_columns = filtered_df.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            selected_cat_col = st.selectbox("Pilih kolom kategorikal", categorical_columns)
            if selected_cat_col:
                st.write(f"Nilai unik pada kolom {selected_cat_col}:")
                value_counts = filtered_df[selected_cat_col].value_counts()
                # Konversi value_counts ke format string untuk menghindari masalah Arrow
                st.write(value_counts.astype(str))
        else:
            st.info("Tidak ada kolom kategorikal dalam dataset.")
    