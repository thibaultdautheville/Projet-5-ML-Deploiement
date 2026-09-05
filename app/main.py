# app/main.py
from fastapi import FastAPI, HTTPException

from app.model_loader import predict
from app.schemas import PredictionInput, PredictionOutput

app = FastAPI(
    title="API Prédiction Attrition RH",
    description="POC de déploiement d'un modèle ML (Projet 5 - OpenClassrooms)",
    version="0.1.0",
)


@app.get("/")
def root():
    return {"message": "API opérationnelle. Voir /docs pour la documentation Swagger."}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(input_data: PredictionInput):
    try:
        input_dict = input_data.model_dump(by_alias=True)
        result = predict(input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
