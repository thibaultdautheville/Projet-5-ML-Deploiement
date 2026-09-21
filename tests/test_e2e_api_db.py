import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.database import SessionLocal
from app.database_models import (
    PredictionInput,
    PredictionOutput as PredictionOutputDB,
)
from app.main import app
from app.schemas import EmployeeInput


DATASET_PATH = "data/DataFrameRH_clean.csv"


def test_prediction_api_postgresql_bout_en_bout():
    # ---------------------------------------------------------
    # 1. Préparation d'un vrai individu du dataset
    # ---------------------------------------------------------
    df = pd.read_csv(DATASET_PATH)

    premier_individu = df.head(1)

    colonnes_api = list(EmployeeInput.model_fields.keys())

    ligne_api = premier_individu[colonnes_api].iloc[0]

    payload = json.loads(
        ligne_api.to_json(force_ascii=False)
    )

    # ---------------------------------------------------------
    # 2. Etat de la base AVANT l'appel API
    # ---------------------------------------------------------
    with SessionLocal() as session:
        nb_inputs_avant = session.scalar(
            select(func.count()).select_from(PredictionInput)
        )

        nb_outputs_avant = session.scalar(
            select(func.count()).select_from(PredictionOutputDB)
        )

        dernier_input_id_avant = session.scalar(
            select(func.max(PredictionInput.id))
        ) or 0

    # ---------------------------------------------------------
    # 3. Appel réel de POST /predict
    # ---------------------------------------------------------
    client = TestClient(app)

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    resultat_api = response.json()

    # ---------------------------------------------------------
    # 4. Etat de la base APRES l'appel API
    # ---------------------------------------------------------
    with SessionLocal() as session:
        nb_inputs_apres = session.scalar(
            select(func.count()).select_from(PredictionInput)
        )

        nb_outputs_apres = session.scalar(
            select(func.count()).select_from(PredictionOutputDB)
        )

        nouvel_input = session.scalars(
            select(PredictionInput)
            .where(PredictionInput.id > dernier_input_id_avant)
            .order_by(PredictionInput.id.desc())
        ).first()

        assert nouvel_input is not None

        nouvel_output = session.scalars(
            select(PredictionOutputDB)
            .where(PredictionOutputDB.input_id == nouvel_input.id)
        ).first()

        assert nouvel_output is not None

        # -----------------------------------------------------
        # 5. Vérifications
        # -----------------------------------------------------

        # Une nouvelle entrée a été enregistrée.
        assert nb_inputs_apres == nb_inputs_avant + 1

        # Une nouvelle sortie a été enregistrée.
        assert nb_outputs_apres == nb_outputs_avant + 1

        # Quelques champs de l'entrée correspondent bien
        # au salarié envoyé à l'API.
        assert nouvel_input.age == payload["age"]
        assert nouvel_input.genre == payload["genre"]
        assert nouvel_input.poste == payload["poste"]

        # L'output PostgreSQL est relié au bon input.
        assert nouvel_output.input_id == nouvel_input.id

        # La prédiction enregistrée est exactement celle
        # retournée par l'API.
        assert nouvel_output.prediction == resultat_api["prediction"]

        assert nouvel_output.probabilite_attrition == pytest.approx(
            resultat_api["probabilite_attrition"],
            rel=1e-7,
        )

    print("\n=== TEST E2E API + POSTGRESQL ===")
    print("Input PostgreSQL ID :", nouvel_input.id)
    print("Prediction :", resultat_api["prediction"])
    print(
        "Probabilite :",
        resultat_api["probabilite_attrition"],
    )
    print("Input enregistré : OK")
    print("Output enregistré : OK")
    print("Lien input/output : OK")
    print("Cohérence API/BDD : OK")
    print("\nTEST E2E COMPLET : OK")