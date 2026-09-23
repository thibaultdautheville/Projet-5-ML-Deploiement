from pathlib import Path
import json
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import EmployeeInput

DATASET_PATH = Path("data/DataFrameRH_clean.csv")
DATASET_1_LIGNE_PATH = Path("data/DataFrameRH_clean_1ligne.csv")


def executer_test_premier_individu():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATASET_PATH.resolve()}")

    df = pd.read_csv(DATASET_PATH)
    if df.empty:
        raise ValueError("Le dataset est vide.")

    print("\n=== DATASET SOURCE ===")
    print("Fichier :", DATASET_PATH)
    print("Dimensions :", df.shape)
    print("Nombre de colonnes :", len(df.columns))

    premier_individu = df.head(1).copy()
    premier_individu.to_csv(DATASET_1_LIGNE_PATH, index=False)

    print("\n=== DATASET DE TEST ===")
    print("Fichier créé :", DATASET_1_LIGNE_PATH)
    print("Dimensions :", premier_individu.shape)

    assert len(premier_individu) == 1
    assert list(premier_individu.columns) == list(df.columns)

    colonnes_api = list(EmployeeInput.model_fields.keys())

    print("\n=== ENTREE API ===")
    print("Nombre de champs EmployeeInput :", len(colonnes_api))

    colonnes_manquantes = [
        col for col in colonnes_api
        if col not in premier_individu.columns
    ]
    if colonnes_manquantes:
        raise ValueError(
            "Colonnes nécessaires à l'API absentes du dataset : "
            f"{colonnes_manquantes}"
        )

    ligne_api = premier_individu[colonnes_api].iloc[0]
    payload = json.loads(ligne_api.to_json(force_ascii=False))

    print("Colonnes nécessaires trouvées : OK")

    client = TestClient(app)
    response = client.post("/predict", json=payload)

    print("\n=== REPONSE API ===")
    print("Code HTTP :", response.status_code)

    if response.status_code != 200:
        print("Erreur :", response.text)

    assert response.status_code == 200

    resultat = response.json()
    assert "prediction" in resultat
    assert "probabilite_attrition" in resultat

    print("Prediction :", resultat["prediction"])
    print("Probabilite d'attrition :", resultat["probabilite_attrition"])

    if "a_quitte_l_entreprise" in premier_individu.columns:
        valeur_reelle = premier_individu["a_quitte_l_entreprise"].iloc[0]
        print("Valeur réelle a_quitte_l_entreprise :", valeur_reelle)

    print("\nTEST FONCTIONNEL PREMIER INDIVIDU : OK")
    return resultat


def test_prediction_premier_individu(
    db_session_factory,
):
    resultat = executer_test_premier_individu()

    assert resultat["prediction"] == 1

    assert resultat[
        "probabilite_attrition"
    ] == pytest.approx(
        0.9106922149658203,
        rel=1e-6,
    )

if __name__ == "__main__":
    executer_test_premier_individu()
