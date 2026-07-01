import streamlit as st
import numpy as np
import pandas as pd
from pickle import load
import os

# Set halaman Streamlit
st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")

st.title("Aplikasi Prediksi Customer Churn")
st.markdown("""
Aplikasi ini digunakan untuk memprediksi apakah seorang pelanggan berpotensi churn (berhenti berlangganan) atau tidak 
berdasarkan data aktivitas dan transaksi mereka.
""")

# Load Model dan Scaler yang sudah disimpan dari Notebook
@st.cache_resource
def load_models():
model = load(open("model_terbaik.pkl", "rb"))
scaler = load(open("scaler.pkl", "rb"))
    return model, scaler

try:
    model, scaler = load_models()
    st.success("Model Machine Learning Berhasil Dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan folder 'models/' berisi file .pkl yang benar. Error: {e}")

# Membuat Form Input Fitur Utama (Berdasarkan Top Feature Importance)
st.subheader("Masukkan Data Aktivitas Pelanggan")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Usia Pelanggan (Age)", min_value=1, max_value=100, value=30)
    total_visits = st.number_input("Total Kunjungan (Total Visits)", min_value=0, value=10)
    avg_session_time = st.number_input("Rata-rata Waktu Sesi (Menit)", min_value=0.0, value=15.0)

with col2:
    pages_per_session = st.number_input("Halaman per Sesi (Pages/Session)", min_value=0.0, value=5.0)
    total_spent = st.number_input("Total Pengeluaran ($)", min_value=0.0, value=500.0)
    avg_order_value = st.number_input("Rata-rata Nilai Transaksi ($)", min_value=0.0, value=50.0)

with col3:
    support_tickets = st.number_input("Jumlah Tiket Dukungan (Support Tickets)", min_value=0, value=1)
    satisfaction_score = st.slider("Skor Kepuasan (Satisfaction Score)", min_value=1.0, max_value=5.0, value=4.0)
    nps_score = st.slider("Net Promoter Score (NPS)", min_value=1, max_value=10, value=8)

# Fitur tambahan tiruan untuk melengkapi sisa dimensi data X (26 fitur total)
# Diisi nilai default agar dimensi input sesuai dengan scaler saat training
fitur_dummy = [0] * 17

# Tombol Prediksi
if st.button("Prediksi Status Churn"):
    # Gabungkan semua input menjadi satu array
    input_fitur = [
        age, total_visits, avg_session_time, pages_per_session, 
        total_spent, avg_order_value, support_tickets, satisfaction_score, nps_score
    ] + fitur_dummy
    
    # Lakukan scaling pada data input
    input_array = np.array([input_fitur])
    input_scaled = scaler.transform(input_array)
    
    # Prediksi menggunakan model terbaik (Random Forest)
    prediksi = model.predict(input_scaled)[0]
    probabilitas = model.predict_proba(input_scaled)[0][1]
    
    st.subheader("Hasil Analisis Prediksi:")
    if prediksi == 1:
        st.error(f"Pelanggan Berpotensi CHURN! (Probabilitas: {probabilitas*100:.2f}%)")
        st.markdown("Rekomendasi: Berikan penawaran khusus atau diskon melalui tim marketing untuk mempertahankan pelanggan.")
    else:
        st.success(f"Pelanggan Tetap AKTIF (Tidak Churn) (Probabilitas Tetap: {(1-probabilitas)*100:.2f}%)")