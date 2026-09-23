import pandas as pd
from fastapi import FastAPI, HTTPException

from app.model_loader import load_model, predict
from app.schemas import EmployeeInput, PredictionOutput
from app.database import SessionLocal
from app.database_models import PredictionInput, PredictionOutput as PredictionOutputDB
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
        input_dict = input_data.model_dump()

        # 1. Enregistrement de l'input
        with SessionLocal() as session:
            prediction_input = PredictionInput(**input_dict)

            session.add(prediction_input)
            session.commit()
            session.refresh(prediction_input)

            input_id = prediction_input.id

        # 2. Prédiction
        df = pd.DataFrame([input_dict])
        result = predict(MODEL, df)

        prediction = int(result["predictions"][0])
        probabilite = float(result["probabilities"][0])

        # 3. Enregistrement de l'output
        with SessionLocal() as session:
            prediction_output = PredictionOutputDB(
                input_id=input_id,
                prediction=prediction,
                probabilite_attrition=probabilite,
            )

            session.add(prediction_output)
            session.commit()

        # 4. Réponse API
        return {
            "prediction": prediction,
            "probabilite_attrition": probabilite,
        }

    except Exception as erreur:
        raise HTTPException(
            status_code=400,
            detail=str(erreur)
        ) from erreur