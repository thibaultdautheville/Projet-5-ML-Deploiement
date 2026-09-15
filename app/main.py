import pandas as pd
from fastapi import FastAPI, HTTPException

from app.model_loader import load_model, predict
from app.schemas import EmployeeInput, PredictionOutput


app = FastAPI(
    title="API Prediction Attrition RH",
    description="POC de deploiement d'un modele ML - Projet 5 OpenClassrooms",
    version="0.1.0",
)

MODEL = load_model()


@app.get("/")
def root():
    return {"message": "API operationnelle. Voir /docs pour Swagger."}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True,
        "n_features": MODEL.n_features_in_,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(input_data: EmployeeInput):
    try:
        df = pd.DataFrame([input_data.model_dump()])
        result = predict(MODEL, df)

        return {
            "prediction": int(result["predictions"][0]),
            "probabilite_attrition": float(result["probabilities"][0]),
        }

    except Exception as erreur:
        raise HTTPException(status_code=400, detail=str(erreur)) from erreur
