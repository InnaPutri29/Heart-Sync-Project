from flask import Flask, render_template, request
import joblib
import numpy as np
from pathlib import Path
import time

# Inisialisasi Path dan Flask
app_dir = Path(__file__).parent
app = Flask(__name__, 
            static_folder=str(app_dir / "static"), 
            template_folder=str(app_dir / "templates"))

# Cache-busting untuk CSS
app.config['TIMESTAMP'] = str(int(time.time()))

# Load model Random Forest (9 fitur terpilih)
model_path = app_dir / "heart_model.pkl"
model = joblib.load(model_path)

# Urutan fitur HARUS sama persis dengan saat training
# selected_features = ['cp', 'thalach', 'ca', 'thal', 'oldpeak', 'age', 'exang', 'chol', 'trestbps']
FEATURE_ORDER = ['cp', 'thalach', 'ca', 'thal', 'oldpeak', 'age', 'exang', 'chol', 'trestbps']

# Fungsi pembantu untuk konversi input aman
def get_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def get_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default

@app.route("/")
def home():
    return render_template("index.html", active_page="home")

@app.route("/prediksi", methods=["GET", "POST"])
def prediksi():
    result = None
    data = {}

    if request.method == "POST":
        try:
            # 1. Ambil semua input dari form
            cp       = get_int(request.form.get("cp"), 0)         # 0-3
            thalach  = get_float(request.form.get("thalach"), 150) # detak jantung maks
            ca       = get_int(request.form.get("ca"), 0)          # 0-3
            thal     = get_int(request.form.get("thal"), 1)        # 1-3
            oldpeak  = get_float(request.form.get("oldpeak"), 0.0) # depresi ST
            age      = get_int(request.form.get("age"), 45)        # usia
            exang    = get_int(request.form.get("exang"), 0)       # 0 atau 1
            chol     = get_float(request.form.get("chol"), 200)    # kolesterol
            trestbps = get_float(request.form.get("trestbps"), 120) # tekanan darah

            # 2. Susun fitur sesuai FEATURE_ORDER
            # ['cp', 'thalach', 'ca', 'thal', 'oldpeak', 'age', 'exang', 'chol', 'trestbps']
            features = np.array([[
                cp,
                thalach,
                ca,
                thal,
                oldpeak,
                age,
                exang,
                chol,
                trestbps
            ]])  # shape: (1, 9)

            # 3. Eksekusi Prediksi
            probability = float(model.predict_proba(features)[0][1])

            # 4. Tentukan Label dan Warna Hasil
            label = "Risiko Tinggi" if probability > 0.5 else "Risiko Rendah"
            color = "danger" if probability > 0.5 else "success"
            message = (
                "⚠️ Pasien menunjukkan indikator risiko penyakit jantung yang tinggi. Segera konsultasikan dengan spesialis."
                if probability > 0.5 else
                "✅ Indikator klinis menunjukkan risiko penyakit jantung yang rendah untuk saat ini."
            )

            result = {
                "label"      : label,
                "color"      : color,
                "probability": f"{probability:.1%}",
                "message"    : message
            }

            # 5. Simpan data input agar form tidak kosong setelah submit (Sticky Form)
            data = {
                "age"     : age,
                "thalach" : thalach,
                "chol"    : chol,
                "trestbps": trestbps,
                "oldpeak" : oldpeak,
                "cp"      : str(cp),
                "ca"      : str(ca),
                "thal"    : str(thal),
                "exang"   : str(exang),
            }

        except Exception as e:
            print(f"Error Prediksi: {e}")
            return f"Terjadi kesalahan pada sistem: {e}", 500

    return render_template("predict.html", active_page="prediksi", result=result, data=data)

@app.route("/info")
def info():
    return render_template("info.html", active_page="info")

@app.route("/team")
def team():
    return render_template("team.html", active_page="team")

if __name__ == "__main__":
    app.run(debug=True)