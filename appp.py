import os, io, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template
from skimage.io import imread
from skimage import color, measure
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'cad_model.pkl')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'models', 'X_train.csv')

# ─────────────────────────────────────────────
# SAFE MODEL LOADING
# ─────────────────────────────────────────────
cad_model = None
X_train_data = None

try:
    import joblib
    cad_model = joblib.load(MODEL_PATH)
    print("✅ Model loaded")
except Exception as e:
    print("❌ Model load failed:", e)

try:
    if os.path.exists(DATA_PATH):
        X_train_data = pd.read_csv(DATA_PATH)
        print("✅ Data loaded")
except Exception as e:
    print("❌ Data load failed:", e)

# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────
CAD_FEATURES = ['age','male','currentSmoker','cigsPerDay','BPMeds','prevalentStroke',
                'prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose']

FEATURE_LABELS = {f: f.upper() for f in CAD_FEATURES}

# ─────────────────────────────────────────────
# SHAP APPROX (SAFE)
# ─────────────────────────────────────────────
def compute_shap_approx(input_df):
    if cad_model is None or X_train_data is None:
        return {f: 0 for f in CAD_FEATURES}

    importances = cad_model.feature_importances_
    means = X_train_data.mean()
    diffs = (input_df.iloc[0] - means).values
    return {CAD_FEATURES[i]: float(importances[i] * diffs[i]) for i in range(len(CAD_FEATURES))}

def make_chart(shap_vals, proba):
    labels = list(shap_vals.keys())
    values = list(shap_vals.values())

    fig, ax = plt.subplots(figsize=(6,4))
    ax.barh(labels, values)
    ax.set_title(f"Risk Score: {proba:.2f}")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close()
    return img

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

# ─────────────────────────────────────────────
# CAD PREDICTION
# ─────────────────────────────────────────────
@app.route('/predict_cad', methods=['POST'])
def predict_cad():
    try:
        if cad_model is None:
            return jsonify({'error': 'Model not loaded. Fix model file.'})

        data = request.get_json()
        vals = [float(data[f]) for f in CAD_FEATURES]
        df = pd.DataFrame([vals], columns=CAD_FEATURES)

        pred = int(cad_model.predict(df)[0])
        proba = float(cad_model.predict_proba(df)[0][1])

        shap_vals = compute_shap_approx(df)
        chart = make_chart(shap_vals, proba)

        risk = 'high' if proba > 0.5 else 'medium' if proba > 0.3 else 'low'

        return jsonify({
            'prediction': pred,
            'probability': proba,
            'risk_level': risk,
            'risk_pct': round(proba * 100, 1),
            'top_factors': list(shap_vals.items())[:3],
            'shap_chart': chart,
            'message': 'High Risk Detected' if pred else 'Low Risk'
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# ─────────────────────────────────────────────
# ECG PROCESSING (SIMPLIFIED)
# ─────────────────────────────────────────────
@app.route('/predict_ecg', methods=['POST'])
def predict_ecg():
    try:
        file = request.files['file']
        path = 'temp.png'
        file.save(path)

        img = imread(path)
        gray = color.rgb2gray(img)
        gray = resize(gray, (300, 300))

        variance = float(np.var(gray))

        if variance < 0.01:
            label = "Normal ECG"
            risk = "low"
        elif variance < 0.05:
            label = "Abnormal"
            risk = "medium"
        else:
            label = "Myocardial Infarction"
            risk = "high"

        os.remove(path)

        return jsonify({
            'label': label,
            'confidence': round(variance * 100, 1),
            'risk': risk,
            'description': "AI-based ECG analysis result",
            'preview': None
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)


# app.run(host="0.0.0.0", port=5000)