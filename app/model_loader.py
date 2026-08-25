import json
import pickle
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

with open(MODELS_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODELS_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(MODELS_DIR / "feature_names.json", "r", encoding="utf-8") as f:
    feature_names = json.load(f)

with open(MODELS_DIR / "threshold.json", "r") as f:
    threshold_data = json.load(f)
    threshold = threshold_data["seuil"]


def predict(input_dict: dict) -> dict:
    df = pd.DataFrame([input_dict])[feature_names]
    df_scaled = scaler.transform(df)
    proba = model.predict_proba(df_scaled)[0][1]
    prediction = int(proba >= threshold)

    return {
        "probability": round(float(proba), 4),
        "prediction": prediction,
        "threshold_used": threshold
    }