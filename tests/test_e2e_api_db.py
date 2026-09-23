from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.database_models import (
    PredictionInput,
    PredictionOutput as PredictionOutputDB,
)
from app.main import app


def test_prediction_api_bdd_bout_en_bout(
    db_session_factory,
    valid_payload,
):
    client = TestClient(app)

    with db_session_factory() as session:
        nb_inputs_avant = session.scalar(
            select(func.count()).select_from(PredictionInput)
        )
        nb_outputs_avant = session.scalar(
            select(func.count()).select_from(PredictionOutputDB)
        )

    assert nb_inputs_avant == 0
    assert nb_outputs_avant == 0

    response = client.post(
        "/predict",
        json=valid_payload,
    )

    assert response.status_code == 200

    resultat_api = response.json()

    with db_session_factory() as session:
        nb_inputs_apres = session.scalar(
            select(func.count()).select_from(PredictionInput)
        )
        nb_outputs_apres = session.scalar(
            select(func.count()).select_from(PredictionOutputDB)
        )

        nouvel_input = session.scalars(
            select(PredictionInput)
            .order_by(PredictionInput.id.desc())
        ).first()

        assert nouvel_input is not None

        nouvel_output = session.scalars(
            select(PredictionOutputDB)
            .where(PredictionOutputDB.input_id == nouvel_input.id)
        ).first()

        assert nouvel_output is not None

        assert nb_inputs_apres == 1
        assert nb_outputs_apres == 1

        assert nouvel_input.age == valid_payload["age"]
        assert nouvel_input.genre == valid_payload["genre"]
        assert nouvel_input.poste == valid_payload["poste"]

        assert nouvel_output.input_id == nouvel_input.id
        assert nouvel_output.prediction == resultat_api["prediction"]
        assert (
            nouvel_output.probabilite_attrition
            == resultat_api["probabilite_attrition"]
        )