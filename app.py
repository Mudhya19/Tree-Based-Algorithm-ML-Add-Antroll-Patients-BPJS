import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Aplication Tree-Based Algorithm ML Analysis Prediction for BPJS Antrol Patients",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("Aplication Tree-Based Algorithm ML Analysis Prediction for BPJS Antrol Patients")

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    model = joblib.load('output/Gradient_Boosting_model.pkl')
    scaler = joblib.load('output/scaler.pkl')
    label_encoders = joblib.load('output/label_encoders.pkl')
    return model, scaler, label_encoders

model, scaler, label_encoders = load_model()

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

kd_poli_options = [
    'JAN',
    'ANA',
    'BED',
    'DOT',
    'ENDO',
    'INT',
    'JIW',
    'KLT',
    'MAT',
    'OBG',
    'PAR',
    'SAR',
    'THT',
    'U0032'
]

user_options = [
    'Admin Utama',
    'SEP Mandiri',
    '51248211',
    '51334232',
    '51382242',
    '51383242',
    '51410241',
    '59030161',
    '59031162',
    '59037162',
    '59341232',
    '71331232'
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
hari_registrasi = st.sidebar.slider("Hari Registrasi (0=Monday, 6=Sunday)", min_value=0, max_value=6, value=datetime.now().weekday())

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

# Apply label encoding to categorical features
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

# Scale the features
input_data_scaled = scaler.transform(input_data)

# Prediction button
if st.sidebar.button("Prediksi", key="predict"):
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

# Display model information
st.subheader("Tentang Model")
st.write("""
Aplikasi ini menggunakan model Gradient Boosting yang telah dilatih 
untuk memprediksi keberhasilan pendaftaran pasien BPJS melalui 
kanal APM (Anjungan Pendaftaran Mandiri) dan Mobile JKN.
""")

# Display input values
st.subheader("Parameter Input")
st.write(input_data)