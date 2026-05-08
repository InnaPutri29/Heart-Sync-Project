# Heart-Sync: Deteksi Dini Risiko Penyakit Jantung

## Deskripsi Proyek

Heart-Sync adalah aplikasi web berbasis Machine Learning untuk deteksi dini risiko penyakit jantung. Aplikasi ini menggunakan algoritma Random Forest untuk menganalisis data kesehatan dasar dan memberikan prediksi risiko secara real-time.

## Tim Pengembang

- **Inna Putri Meida** (23051130027) - Frontend Developer & Project Coordinator
- **Ngafifah Rahma Syadza** (23051130030) - Machine Learning Developer

**Program Studi**: Pendidikan Teknik Informatika  
**Institusi**: Universitas Negeri Yogyakarta

## Fitur Utama

### 🏠 Beranda
- Pengantar aplikasi dan latar belakang proyek
- Penjelasan cara penggunaan
- Informasi penting tentang keterbatasan diagnosis

### 🔮 Prediksi Risiko
- **Input Komprehensif** dibagi menjadi 3 kategori:
  - **Informasi Pribadi**: Jenis kelamin, usia, tinggi badan, berat badan, BMI (otomatis), status kesehatan umum
  - **Gaya Hidup**: Status merokok, konsumsi alkohol, frekuensi buah/sayur/gorengan, rutinitas olahraga
  - **Riwayat Medis**: Kondisi komorbid (diabetes, radang sendi, kanker, depresi), riwayat pemeriksaan kesehatan
- **Prediksi Hibrid**: Menggabungkan Machine Learning (70%) dengan analisis gaya hidup (30%)
- **Data Medis Utama**: Tekanan darah, kolesterol, gula darah puasa
- Hasil klasifikasi: Risiko Tinggi atau Risiko Rendah dengan persentase

### 📊 Info Model
- Performa model (akurasi 99%)
- Penjelasan algoritma Random Forest
- Feature Importance analysis
- Confusion Matrix dan metrics evaluasi

### 👥 Profil Tim
- Informasi pengembang
- Visi dan misi proyek

## Teknologi yang Digunakan

- **Machine Learning**: Scikit-learn Random Forest Classifier
- **Frontend**: Streamlit dengan tema Light Blue
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Streamlit Community Cloud

## Dataset

Dataset berasal dari Heart Disease Dataset di Kaggle, berisi data kesehatan pasien dengan fitur medis dan label target penyakit jantung.

## Instalasi dan Menjalankan

1. **Clone repository** (jika diperlukan)
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Jalankan aplikasi**:
   ```bash
   streamlit run heartdieaseup.py
   ```
4. **Buka browser** dan akses `http://localhost:8501`

## Model Machine Learning

### Training
Model dilatih menggunakan:
- 100 decision trees (Random Forest)
- 5 fitur utama: age, sex, trestbps, chol, fbs
- Test size: 20%
- Random state: 42

### Performa
- **Accuracy**: 99%
- **Precision**: 99%
- **Recall**: 99%
- **F1-Score**: 99%

### Prediksi Hibrid
Aplikasi menggunakan pendekatan hibrid:
- **70%** dari prediksi Machine Learning (berdasarkan data medis)
- **30%** dari analisis gaya hidup dan riwayat medis
- Total risiko = (ML_prediction × 0.7) + (lifestyle_risk × 0.3)

### Feature Importance
1. Kolesterol (34.2%)
2. Usia (30.4%)
3. Tekanan Darah (23.8%)
4. Jenis Kelamin (8.5%)
5. Gula Darah (3.2%)

## Deployment

Aplikasi dapat di-deploy ke Streamlit Community Cloud untuk akses online.

## Disclaimer

**PENTING**: 
- Hasil prediksi ini TIDAK menggantikan diagnosis medis profesional dari dokter spesialis jantung
- Aplikasi ini menggunakan kombinasi Machine Learning dan analisis gaya hidup untuk estimasi risiko awal
- Selalu konsultasikan hasil ini dengan tenaga medis yang berkualifikasi untuk pemeriksaan dan diagnosis yang akurat
- Data yang Anda masukkan bersifat pribadi dan tidak disimpan dalam sistem ini

## Lisensi

Lihat file LICENSE untuk informasi lisensi.

## Kontak

Untuk pertanyaan atau feedback, hubungi tim pengembang melalui platform akademik UNY.