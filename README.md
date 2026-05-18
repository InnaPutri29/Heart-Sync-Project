# 🫀 Heart-Sync: Deteksi Dini Risiko Penyakit Jantung

Aplikasi web berbasis Machine Learning untuk deteksi dini risiko penyakit jantung. Heart-Sync menggunakan algoritma **Random Forest** yang dilatih pada dataset klinis jantung dan memberikan prediksi risiko secara real-time melalui antarmuka web yang modern.

---

## 👥 Tim Pengembang

| Nama | NIM | 
|------|-----|
| Inna Putri Meida | 23051130027 | 
| Ngafifah Rahma Syadza | 23051130030 | 

**Mata Kuliah** : Artificial Intelligence 
**Program Studi** :Teknologi Informasi  
**Fakultas** : Fakultas Teknik  
**Universitas** : Universitas Negeri Yogyakarta  
**Dosen Pengampu** : Dr. Ir. Fatchul Arifin, M.T.  
**Tahun** : 2026  

---

## ✨ Fitur Utama

### 🏠 Beranda
- Pengantar aplikasi dan latar belakang proyek
- Statistik performa model (akurasi, jumlah fitur)
- Navigasi ke fitur prediksi dan info model

### 🔮 Prediksi Risiko
- Input **9 parameter klinis** yang dipilih berdasarkan feature importance
- Klasifikasi **Risiko Tinggi / Risiko Rendah** dengan persentase probabilitas
- Form sticky — data tetap tersimpan setelah submit

### 📊 Info Model
- Metrik evaluasi model RF Final (Accuracy, Recall, Precision, F1, ROC-AUC)
- Tabel perbandingan Random Forest vs Decision Tree
- Penjelasan 9 fitur terpilih beserta nilai importance
- Alur kerja prediksi dari input hingga output

### 👥 Profil Tim
- Informasi anggota tim dan dosen pengampu
- Info akademik lengkap
  
---

## 🛠️ Teknologi yang Digunakan

| Komponen | Teknologi |
|----------|-----------|
| Machine Learning | Scikit-learn `RandomForestClassifier` |
| Backend | Flask (Python) |
| Frontend | HTML, Tailwind CSS, Jinja2 |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| Dataset | Heart Disease Dataset (1.025 baris, 14 kolom) |

---

## ⚙️ Instalasi dan Menjalankan

1. **Clone repository**
   ```bash
   git clone <url-repository>
   cd Heart-Sync
   ```

2. **Buat virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi**
   ```bash
   flask run
   ```

5. **Buka browser** dan akses `http://127.0.0.1:5000`

> **Catatan**: Pastikan versi scikit-learn yang diinstall sama dengan versi yang digunakan saat menyimpan `heart_model.pkl` untuk menghindari `InconsistentVersionWarning`.

---

## 🤖 Model Machine Learning

### Dataset
- **Sumber**: Heart Disease Dataset (UCI / Kaggle)
- **Ukuran**: 1.025 baris × 14 kolom
- **Preprocessing**: 18 baris anomali (ca=4) dihapus → 1.007 baris bersih
- **Split**: 80% training (805 baris) / 20% test (202 baris), stratified

### Alur Training
1. Exploratory Data Analysis (EDA) & deteksi outlier IQR
2. Preprocessing (handling missing values, hapus anomali)
3. RF Baseline — 13 fitur
4. Feature Importance & seleksi fitur (threshold ≥ 0.05) → 9 fitur terpilih
5. Retrain RF dengan 9 fitur
6. Hyperparameter Tuning (RandomizedSearchCV, n_iter=50, scoring=recall)
7. Evaluasi final & perbandingan dengan Decision Tree
8. Simpan model terbaik (`heart_model.pkl`)

### 9 Fitur Terpilih

| # | Fitur | Nama | Importance |
|---|-------|------|-----------|
| 1 | `cp` | Tipe Nyeri Dada | 0.167 |
| 2 | `thalach` | Detak Jantung Maks | 0.133 |
| 3 | `ca` | Jumlah Pembuluh Utama | 0.130 |
| 4 | `thal` | Thalassemia | 0.119 |
| 5 | `oldpeak` | Depresi ST | 0.100 |
| 6 | `age` | Usia | 0.080 |
| 7 | `exang` | Angina saat Olahraga | 0.066 |
| 8 | `chol` | Kolesterol | 0.057 |
| 9 | `trestbps` | Tekanan Darah Istirahat | 0.056 |

### Hyperparameter Terbaik (RF Final)
```
n_estimators     : 100
max_depth        : 10
min_samples_split: 5
min_samples_leaf : 1
max_features     : sqrt
class_weight     : balanced
```

### Performa Model

| Metrik | RF Final (9 Fitur) | DT Final (9 Fitur) |
|--------|-------------------|-------------------|
| Accuracy | **0.9703** | 0.9802 |
| Recall | **0.9806** | 1.0000 |
| Precision | **0.9619** | 0.9626 |
| F1-Score | **0.9712** | 0.9810 |
| ROC-AUC | **0.9949** ✅ | 0.9798 |
| CV Recall (mean) | **0.9829** ✅ | 0.9804 |

> **RF dipilih untuk deployment** karena ROC-AUC dan CV Recall lebih tinggi, menunjukkan stabilitas lebih baik pada data baru dibanding Decision Tree yang cenderung overfit.

---

## ⚠️ Disclaimer

**PENTING**:
- Hasil prediksi ini **TIDAK menggantikan diagnosis medis profesional** dari dokter spesialis
- Aplikasi ini ditujukan untuk **skrining mandiri dan edukasi kesehatan** berbasis machine learning
- Selalu konsultasikan hasil dengan tenaga medis yang berkualifikasi
- Data yang dimasukkan **tidak disimpan** dalam sistem
