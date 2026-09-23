import joblib
import json
import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def model():
    return joblib.load("models/model.pkl")


@pytest.fixture
def feature_names():
    with open("models/feature_names.json") as f:
        return json.load(f)


def test_feature_count(feature_names):
    """Verifie qu'on a bien le nombre de features attendu."""
    assert len(feature_names) > 0


def test_model_predicts_with_expected_features(model, feature_names):
    """Verifie que le modele accepte un DataFrame construit avec les features attendues."""
    fake_row = pd.DataFrame(
        [np.random.rand(len(feature_names))],
        columns=feature_names
    )
    pred = model.predict(fake_row)
    assert pred is not None
    assert len(pred) == 1