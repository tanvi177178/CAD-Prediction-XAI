import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# ─────────────────────────────
# Generate synthetic realistic dataset
# ─────────────────────────────
np.random.seed(42)

n = 1000

data = pd.DataFrame({
    'age': np.random.randint(30, 80, n),
    'male': np.random.randint(0, 2, n),
    'currentSmoker': np.random.randint(0, 2, n),
    'cigsPerDay': np.random.randint(0, 30, n),
    'BPMeds': np.random.randint(0, 2, n),
    'prevalentStroke': np.random.randint(0, 2, n),
    'prevalentHyp': np.random.randint(0, 2, n),
    'diabetes': np.random.randint(0, 2, n),
    'totChol': np.random.randint(150, 300, n),
    'sysBP': np.random.randint(100, 180, n),
    'diaBP': np.random.randint(60, 120, n),
    'BMI': np.random.uniform(18, 35, n),
    'heartRate': np.random.randint(60, 110, n),
    'glucose': np.random.randint(70, 200, n),
})

# Risk logic (semi-realistic)
risk = (
    (data['age'] > 50).astype(int) +
    (data['sysBP'] > 140).astype(int) +
    (data['totChol'] > 240).astype(int) +
    (data['diabetes'] == 1).astype(int) +
    (data['currentSmoker'] == 1).astype(int)
)

y = (risk >= 2).astype(int)

# ─────────────────────────────
# Train model
# ─────────────────────────────
model = RandomForestClassifier(n_estimators=100)
model.fit(data, y)

# ─────────────────────────────
# Save files
# ─────────────────────────────
model_dir = "models"

import os
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "cad_model.pkl"))
data.to_csv(os.path.join(model_dir, "X_train.csv"), index=False)

print("✅ Model and dataset saved in /models folder")