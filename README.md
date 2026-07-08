# CardiXAI — Coronary Artery Disease Prediction with Explainable AI

## Project Structure

```
flask_app/
├── app.py                  ← Main Flask backend
├── requirements.txt        ← Python dependencies
├── templates/
│   └── index.html          ← Full single-page UI
└── models/
    ├── cad_model.pkl       ← Trained Random Forest (Framingham dataset)
    └── X_train.pkl         ← Training data for SHAP approximation
```

---

## Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Your Model Files
Ensure these files are in the `models/` folder:
- `cad_model.pkl` — trained RandomForestClassifier on Framingham dataset
- `X_train.pkl` — pandas DataFrame of training features

### 3. Run the App
```bash
python app.py
```
Then open: **http://127.0.0.1:5000**

---

## Features

### 🫀 CAD Prediction (Manual Entry)
- Input 14 clinical parameters from Framingham dataset
- Outputs: 10-year CHD risk percentage, risk level (Low/Medium/High)
- SHAP-style feature importance bar chart
- Top 3 contributing risk factors

### 📈 ECG Scan
- Upload a standard 12-lead ECG image (PNG/JPG)
- Automatically segments all 12 leads
- Extracts 1D signal via contour detection
- Classifies into: Normal / Abnormal Heartbeat / Myocardial Infarction / History of MI
- Shows extracted leads visualization

---

## Model Details

| Aspect | Detail |
|--------|--------|
| Dataset | Framingham Heart Study |
| Algorithm | Random Forest (300 trees, class_weight='balanced') |
| Features | 14 clinical features |
| Target | TenYearCHD (10-year coronary heart disease risk) |
| XAI Method | SHAP feature attribution (TreeExplainer) |

---

## ECG Classification Labels
| Code | Meaning |
|------|---------|
| 0 | Abnormal Heartbeat |
| 1 | Myocardial Infarction |
| 2 | Normal ECG |
| 3 | History of Myocardial Infarction |

---

## Notes on ECG Model
The ECG `.pkl` file was trained on sklearn 1.0.1 and is incompatible with newer sklearn versions.
The app uses the ECG signal processing pipeline from `Ecg.py` (lead segmentation → contour extraction → signal scaling)
and applies heuristic-based classification on signal statistics. For full model accuracy, retrain
the ECG classifier using the [GitHub dataset](https://github.com/rameshavinash94/Cardiovascular-Detection-using-ECG-images).

---

## Disclaimer
This tool is for **educational and research purposes only**.  
It is NOT a substitute for professional medical advice or diagnosis.
