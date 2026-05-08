from flask import Flask, render_template, request
import joblib
import numpy as np
from pathlib import Path
import os
import time

# Tentukan path direktori aplikasi
app_dir = Path(__file__).parent
static_dir = app_dir / "static"
template_dir = app_dir / "templates"

# Inisialisasi Flask dengan path static dan templates yang eksplisit
app = Flask(__name__, 
            static_folder=str(static_dir), 
            static_url_path="/static",
            template_folder=str(template_dir))

# Konfigurasi untuk cache-busting CSS dengan timestamp
app.config['TIMESTAMP'] = str(int(time.time()))

model_path = app_dir / "heart_model.pkl"
model = joblib.load(model_path)


def calculate_lifestyle_risk(smoking, alcohol, fruits_veg, fried_food, exercise, diabetes, arthritis, cancer, depression, last_checkup):
    """Hitung skor risiko gaya hidup berdasarkan faktor-faktor kesehatan"""
    score = 0
    # Risiko BMI dan gaya hidup dasar
    if smoking == "Sering":
        score += 2
    elif smoking == "Kadang-kadang":
        score += 1

    if alcohol == "Sering":
        score += 1

    if fruits_veg < 2:
        score += 1

    if fried_food > 5:
        score += 1

    if exercise == "Tidak Pernah":
        score += 2
    elif exercise == "1-2x/minggu":
        score += 1

    if diabetes:
        score += 1
    if arthritis:
        score += 1
    if cancer:
        score += 1
    if depression:
        score += 1

    if last_checkup in ["> 5 tahun", "Tidak pernah"]:
        score += 1

    max_score = 15
    return round(score / max_score, 2)


@app.route("/")
def home():
    return render_template("index.html", active_page="home")


@app.route("/prediksi", methods=["GET", "POST"])
def prediksi():
    result = None
    data = {}

    if request.method == "POST":
        sex = request.form.get("sex", "Laki-laki")
        age = int(request.form.get("age", 25))
        height = int(request.form.get("height", 170))
        weight = int(request.form.get("weight", 70))
        health_status = request.form.get("health_status", "Baik")
        smoking = request.form.get("smoking", "Tidak Pernah")
        alcohol = request.form.get("alcohol", "Tidak Pernah")
        fruits_veg = int(request.form.get("fruits_veg", 3))
        fried_food = int(request.form.get("fried_food", 2))
        exercise = request.form.get("exercise", "3-4x/minggu")
        diabetes = request.form.get("diabetes") == "on"
        arthritis = request.form.get("arthritis") == "on"
        cancer = request.form.get("cancer") == "on"
        depression = request.form.get("depression") == "on"
        last_checkup = request.form.get("last_checkup", "1-2 tahun")
        trestbps = int(request.form.get("trestbps", 120))
        chol = int(request.form.get("chol", 200))
        fbs = request.form.get("fbs", "Tidak")

        bmi = weight / ((height / 100) ** 2) if height > 0 else 0
        sex_val = 1 if sex == "Laki-laki" else 0
        fbs_val = 1 if fbs == "Ya" else 0

        ml_input = np.array([[age, sex_val, trestbps, chol, fbs_val]])
        ml_probability = float(model.predict_proba(ml_input)[0][1])
        lifestyle_risk = calculate_lifestyle_risk(
            smoking,
            alcohol,
            fruits_veg,
            fried_food,
            exercise,
            diabetes,
            arthritis,
            cancer,
            depression,
            last_checkup,
        )
        final_risk = round((ml_probability * 0.7) + (lifestyle_risk * 0.3), 2)
        label = "Risiko Tinggi" if final_risk > 0.5 else "Risiko Rendah"
        color = "danger" if final_risk > 0.5 else "success"
        message = (
            "⚠️ Risiko penyakit jantung terdeteksi tinggi. Segera konsultasikan ke dokter untuk pemeriksaan lebih lanjut."
            if final_risk > 0.5
            else "✅ Risiko penyakit jantung rendah. Tetap pertahankan dan tingkatkan gaya hidup sehat."
        )

        result = {
            "label": label,
            "color": color,
            "message": message,
            "final_risk": f"{final_risk:.0%}",
            "ml_probability": f"{ml_probability:.0%}",
            "lifestyle_risk": f"{lifestyle_risk:.0%}",
        }

        data = {
            "sex": sex,
            "age": age,
            "height": height,
            "weight": weight,
            "bmi": f"{bmi:.1f}",
            "health_status": health_status,
            "smoking": smoking,
            "alcohol": alcohol,
            "fruits_veg": fruits_veg,
            "fried_food": fried_food,
            "exercise": exercise,
            "diabetes": diabetes,
            "arthritis": arthritis,
            "cancer": cancer,
            "depression": depression,
            "last_checkup": last_checkup,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
        }

    return render_template("predict.html", active_page="prediksi", result=result, data=data)


@app.route("/info")
def info():
    return render_template("info.html", active_page="info")


@app.route("/team")
def team():
    return render_template("team.html", active_page="team")


if __name__ == "__main__":
    app.run(debug=True)
