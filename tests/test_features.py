import json

import pandas as pd
import pytest

from app.model_loader import (
    load_feature_names,
    load_model,
    load_threshold,
    predict,
    preprocess_input,
)
from app.schemas import EmployeeInput


DATASET_PATH = "data/DataFrameRH_clean.csv"


@pytest.fixture(scope="module")
def premier_individu():
    df = pd.read_csv(DATASET_PATH)

    colonnes_api = list(EmployeeInput.model_fields.keys())

    return df[colonnes_api].head(1).copy()


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_nombre_features():
    feature_names = load_feature_names()

    assert len(feature_names) == 54
    assert len(set(feature_names)) == 54


def test_preprocessing_produit_54_features(premier_individu):
    feature_names = load_feature_names()

    X = preprocess_input(premier_individu)

    assert X.shape == (1, 54)
    assert len(set(X.columns)) == 54
    assert set(X.columns) == set(feature_names)

    X_aligne = X[feature_names]

    assert list(X_aligne.columns) == feature_names


def test_seuil_metier_valide():
    seuil = load_threshold()

    assert 0 <= seuil <= 1


def test_prediction_reproductible(model, premier_individu):
    resultat = predict(
        model,
        premier_individu,
    )

    assert resultat["predictions"] == [1]

    assert resultat["probabilities"][0] == pytest.approx(
        0.9106922149658203,
        rel=1e-6,
    )


def test_experience_hors_bornes_declenche_erreur(
    premier_individu,
):
    individu = premier_individu.copy()

    individu["nombre_experiences_precedentes"] = 100

    with pytest.raises(ValueError):
        preprocess_input(individu)


def test_augmentation_hors_bornes_declenche_erreur(
    premier_individu,
):
    individu = premier_individu.copy()

    individu["augementation_salaire_precedente"] = 100

    with pytest.raises(ValueError):
        preprocess_input(individu)

def test_modele_attend_54_features(model):
    assert model.n_features_in_ == 54
