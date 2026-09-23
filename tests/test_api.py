from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    assert "message" in response.json()


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    resultat = response.json()

    assert resultat["status"] == "ok"


def test_age_invalide_retourne_422(valid_payload):
    payload = valid_payload.copy()

    payload["age"] = -10

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_genre_invalide_retourne_422(valid_payload):
    payload = valid_payload.copy()

    payload["genre"] = "X"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_champ_obligatoire_manquant_retourne_422(
    valid_payload,
):
    payload = valid_payload.copy()

    payload.pop("revenu_mensuel")

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422