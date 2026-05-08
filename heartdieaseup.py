import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. KONFIGURASI HALAMAN UTAMA
st.set_page_config(page_title="Heart-Sync", page_icon="💙", layout="centered")

# Load the trained model
model = joblib.load('heart_model.pkl')

# Sidebar navigation
st.sidebar.title("Heart-Sync Navigation")
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Prediksi Risiko", "Info Model", "Profil Tim"])

# 2. SUNTIKAN CUSTOM CSS (Tema Biru Muda)
st.markdown("""
    <style>
    /* Mengubah warna background halaman */
    .stApp {
        background-color: #F0F8FF; /* Warna Alice Blue yang sangat muda dan bersih */
    }
    
    /* Judul Utama Gradient Biru Muda ke Biru */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #56CCF2, #2F80ED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .sub-title {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 40px;
    }

    /* Membuat Kotak Container */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* Tombol Prediksi Utama (Biru Muda) */
    .stButton>button {
        background: linear-gradient(90deg, #56CCF2 0%, #2F80ED 100%);
        color: white;
        border-radius: 30px;
        width: 100%;
        padding: 15px;
        font-weight: bold;
        font-size: 18px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(47, 128, 237, 0.4); /* Bayangan biru */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Page logic
if page == "Beranda":
    st.markdown("<div class='main-title'>Heart-Sync</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Pengembangan Model Machine Learning untuk Deteksi Dini Risiko Penyakit Jantung</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ## Selamat Datang di Heart-Sync
    
    Aplikasi cerdas berbasis data kesehatan dasar yang mampu memberikan peringatan dini mengenai tingkat risiko jantung secara instan.
    
    ### Fitur Utama:
    - **Prediksi Risiko Real-time**: Masukkan data kesehatan Anda dan dapatkan hasil analisis segera
    - **Algoritma Random Forest**: Model machine learning dengan akurasi tinggi
    - **Antarmuka User-Friendly**: Desain yang mudah digunakan dengan tema biru menenangkan
    
    ### Cara Penggunaan:
    1. Pilih menu "Prediksi Risiko" di sidebar
    2. Isi formulir dengan data kesehatan Anda
    3. Klik "Mulai Prediksi" untuk mendapatkan hasil
    
    ### Penting:
    Hasil prediksi ini bukan pengganti diagnosis medis profesional. Selalu konsultasikan dengan dokter untuk pemeriksaan lebih lanjut.
    """)

elif page == "Prediksi Risiko":
    # 3. HEADER & HERO SECTION
    st.markdown("<div class='main-title'>Cek Risiko Jantung Anda</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Isi formulir medis di bawah untuk mendapatkan prediksi risiko penyakit kardiovaskular berdasarkan teknologi Machine Learning.</div>", unsafe_allow_html=True)

    # 4. CONTOH PROFIL
    st.markdown("**🪄 Contoh Profil untuk Testing:**")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("💚 Profil Sehat"):
            st.info("**Contoh Profil Sehat:**\n- Usia: 25\n- Jenis Kelamin: Perempuan\n- Tekanan Darah: 120 mmHg\n- Kolesterol: 200 mg/dl\n- Gula Darah: Tidak >120")
    with col_b:
        if st.button("⚠️ Profil Sedang"):
            st.warning("**Contoh Profil Sedang:**\n- Usia: 45\n- Jenis Kelamin: Laki-laki\n- Tekanan Darah: 135 mmHg\n- Kolesterol: 220 mg/dl\n- Gula Darah: Tidak >120")
    with col_c:
        if st.button("💔 Profil Tinggi"):
            st.error("**Contoh Profil Tinggi:**\n- Usia: 60\n- Jenis Kelamin: Laki-laki\n- Tekanan Darah: 150 mmHg\n- Kolesterol: 250 mg/dl\n- Gula Darah: Ya >120")

    st.write("---")

    # 5. FORM INPUT DATA MEDIS - DIBAGI MENJADI 3 KATEGORI
    st.markdown("### 👤 1. Informasi Pribadi")
    
    # Baris 1: Jenis kelamin dan usia
    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    with col2:
        age = st.number_input("Kategori Usia", min_value=1, max_value=120, value=25)
    
    # Baris 2: Tinggi, Berat, BMI otomatis
    col3, col4, col5 = st.columns(3)
    with col3:
        height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=170)
    with col4:
        weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=70)
    with col5:
        bmi = weight / ((height/100) ** 2) if height > 0 else 0
        st.metric("BMI (otomatis)", f"{bmi:.1f}")
    
    # Status kesehatan umum
    health_status = st.selectbox("Status Kesehatan Umum", ["Sangat Baik", "Baik", "Sedang", "Buruk"])
    
    st.markdown("---")
    st.markdown("### 🏃‍♂️ 2. Gaya Hidup")
    
    # Kebiasaan merokok dan alkohol
    col6, col7 = st.columns(2)
    with col6:
        smoking = st.selectbox("Status Merokok", ["Tidak Pernah", "Kadang-kadang", "Sering", "Berhenti"])
    with col7:
        alcohol = st.selectbox("Konsumsi Alkohol", ["Tidak Pernah", "Kadang-kadang", "Sering"])
    
    # Frekuensi konsumsi makanan
    col8, col9, col10 = st.columns(3)
    with col8:
        fruits_veg = st.slider("Frekuensi Konsumsi Buah/Sayur (per hari)", 0, 10, 3)
    with col9:
        fried_food = st.slider("Frekuensi Konsumsi Gorengan (per minggu)", 0, 20, 2)
    with col10:
        exercise = st.selectbox("Rutinitas Olahraga", ["Tidak Pernah", "1-2x/minggu", "3-4x/minggu", "Hampir Setiap Hari"])
    
    st.markdown("---")
    st.markdown("### 🏥 3. Riwayat Medis")
    
    # Kondisi komorbid
    st.markdown("**Kondisi Komorbid (centang jika ada):**")
    col11, col12 = st.columns(2)
    with col11:
        diabetes = st.checkbox("Diabetes")
        arthritis = st.checkbox("Radang Sendi")
    with col12:
        cancer = st.checkbox("Kanker")
        depression = st.checkbox("Depresi")
    
    # Riwayat pemeriksaan
    last_checkup = st.selectbox("Riwayat Pemeriksaan Kesehatan Terakhir", ["< 1 tahun", "1-2 tahun", "2-5 tahun", "> 5 tahun", "Tidak pernah"])
    
    st.markdown("---")
    
    # Input medis utama (seperti sebelumnya)
    st.markdown("### 🩺 Data Medis Utama")
    col13, col14, col15 = st.columns(3)
    with col13:
        trestbps = st.number_input("Tekanan Darah (mm Hg)", min_value=50, max_value=250, value=120)
    with col14:
        chol = st.number_input("Kolesterol (mg/dl)", min_value=100, max_value=600, value=200)
    with col15:
        fbs = st.selectbox("Gula Darah Puasa > 120?", ["Tidak", "Ya"])

    # 6. TOMBOL PREDIKSI & HASIL
    st.write("---")
    if st.button("🔮 Mulai Prediksi Sekarang"):
        
        # Menampilkan animasi loading
        with st.spinner("AI sedang menganalisis data Anda..."):
            
            # Prediksi ML utama
            sex_val = 1 if sex == "Laki-laki" else 0
            fbs_val = 1 if fbs == "Ya" else 0
            input_data = np.array([[age, sex_val, trestbps, chol, fbs_val]])
            ml_prediction = model.predict_proba(input_data)[0][1]  # Probabilitas risiko tinggi
            
            # Hitung skor risiko tambahan
            additional_risk = 0
            max_risk = 15  # Total skor maksimal
            
            # BMI
            if bmi > 30:
                additional_risk += 2
            elif bmi > 25:
                additional_risk += 1
            
            # Merokok
            if smoking == "Sering":
                additional_risk += 2
            elif smoking == "Kadang-kadang":
                additional_risk += 1
            
            # Alkohol
            if alcohol == "Sering":
                additional_risk += 1
            
            # Konsumsi buah/sayur
            if fruits_veg < 2:
                additional_risk += 1
            
            # Gorengan
            if fried_food > 5:
                additional_risk += 1
            
            # Olahraga
            if exercise == "Tidak Pernah":
                additional_risk += 2
            elif exercise == "1-2x/minggu":
                additional_risk += 1
            
            # Komorbid
            if diabetes:
                additional_risk += 1
            if arthritis:
                additional_risk += 1
            if cancer:
                additional_risk += 1
            if depression:
                additional_risk += 1
            
            # Riwayat checkup
            if last_checkup == "> 5 tahun" or last_checkup == "Tidak pernah":
                additional_risk += 1
            
            # Gabungkan prediksi
            lifestyle_risk = additional_risk / max_risk
            final_risk = (ml_prediction * 0.7) + (lifestyle_risk * 0.3)
            
            hasil_berisiko = final_risk > 0.5

        # 7. TAMPILAN HASIL
        if hasil_berisiko:
            st.error("### ⚠️ Risiko Tinggi Terdeteksi!")
            st.markdown(f"""
            **Hasil Prediksi: BERISIKO**
            
            Berdasarkan analisis komprehensif data medis dan gaya hidup Anda, sistem mendeteksi risiko penyakit kardiovaskular yang **TINGGI** ({final_risk:.1%}).
            
            **Faktor Kontribusi:**
            - Prediksi ML: {ml_prediction:.1%}
            - Risiko Gaya Hidup: {lifestyle_risk:.1%}
            
            **Rekomendasi:** Sangat disarankan untuk segera berkonsultasi dengan dokter atau spesialis jantung untuk pemeriksaan lebih lanjut.
            """)
        else:
            st.success("### 💙 Risiko Rendah")
            st.markdown(f"""
            **Hasil Prediksi: TIDAK BERISIKO**
            
            Berdasarkan analisis komprehensif data medis dan gaya hidup Anda, risiko penyakit kardiovaskular terdeteksi **RENDAH** ({final_risk:.1%}).
            
            **Faktor Kontribusi:**
            - Prediksi ML: {ml_prediction:.1%}
            - Risiko Gaya Hidup: {lifestyle_risk:.1%}
            
            Tetap jaga kesehatan jantung Anda! Terus pertahankan gaya hidup sehat dengan rutin berolahraga, makan makanan bergizi, dan kelola stres dengan baik.
            """)

elif page == "Info Model":
    st.markdown("<div class='main-title'>Informasi Model</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Performa dan Analisis Model Random Forest</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ## Algoritma Random Forest Classifier
    
    Model ini menggunakan algoritma Random Forest dengan 100 pohon keputusan untuk klasifikasi risiko penyakit jantung.
    
    ### Performa Model:
    - **Akurasi**: 99%
    - **Precision**: 99% (untuk kedua kelas)
    - **Recall**: 99% (untuk kedua kelas)
    - **F1-Score**: 99%
    
    ### Fitur yang Digunakan:
    1. **Usia (Age)**: Faktor risiko utama penyakit jantung
    2. **Jenis Kelamin (Sex)**: Perbedaan risiko antara laki-laki dan perempuan
    3. **Tekanan Darah (Blood Pressure)**: Indikator kesehatan kardiovaskular
    4. **Kolesterol (Cholesterol)**: Kadar lemak dalam darah
    5. **Gula Darah Puasa (Fasting Blood Sugar)**: Indikator diabetes
    
    ### Feature Importance:
    """)
    
    # Load and display feature importance
    try:
        with open('model_metrics.txt', 'r') as f:
            content = f.read()
        st.code(content, language='text')
    except:
        st.write("Data metrik model tidak tersedia.")
    
    st.markdown("""
    ### Interpretasi:
    - **Kolesterol** memiliki pengaruh terbesar (34%)
    - **Usia** merupakan faktor kedua terpenting (30%)
    - **Tekanan Darah** berkontribusi sekitar 24%
    
    Model ini telah dilatih pada dataset kesehatan jantung dan menunjukkan performa yang sangat baik dalam mendeteksi risiko penyakit jantung.
    """)

elif page == "Profil Tim":
    st.markdown("<div class='main-title'>Profil Tim Pengembang</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Mahasiswa Pendidikan Teknik Informatika UNY</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ## Heart-Sync Development Team
    
    ### 👩‍💻 Inna Putri Meida (23051130027)
    **Peran**: Frontend Developer & Project Coordinator
    
    Mahasiswa Pendidikan Teknik Informatika Universitas Negeri Yogyakarta yang bertanggung jawab atas pengembangan antarmuka aplikasi dan koordinasi proyek.
    
    ### 👩‍💻 Ngafifah Rahma Syadza (23051130030)
    **Peran**: Machine Learning Developer
    
    Mahasiswa Pendidikan Teknik Informatika Universitas Negeri Yogyakarta yang mengembangkan model machine learning dan algoritma prediksi.
    
    ### 📚 Program Studi
    **Pendidikan Teknik Informatika**  
    Fakultas Teknik  
    Universitas Negeri Yogyakarta
    
    ### 🎯 Visi Proyek
    Mengembangkan aplikasi kesehatan berbasis AI yang dapat diakses masyarakat luas untuk deteksi dini risiko penyakit jantung, menggabungkan keilmuan informatika dengan sektor kesehatan (Health-Tech).
    
    ### 📞 Kontak
    Untuk pertanyaan lebih lanjut, dapat menghubungi tim pengembang melalui email universitas atau platform akademik UNY.
    """)