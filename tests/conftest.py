import json

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.main as main_module
from app.database_models import Base
from app.schemas import EmployeeInput


DATASET_PATH = "data/DataFrameRH_clean.csv"


@pytest.fixture
def db_session_factory(monkeypatch):
    """
    Base SQLite temporaire utilisée pendant les tests.

    Elle remplace PostgreSQL uniquement pendant le test afin de rendre
    la suite reproductible sur toutes les machines.
    """

    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(bind=test_engine)

    TestingSessionLocal = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
    )

    # app.main utilise SessionLocal importé au chargement du module.
    # On le remplace temporairement par notre base de test.
    monkeypatch.setattr(
        main_module,
        "SessionLocal",
        TestingSessionLocal,
    )

    yield TestingSessionLocal

    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


@pytest.fixture
def valid_payload():
    """
    Payload réel construit à partir du premier individu
    de DataFrameRH_clean.csv.
    """

    df = pd.read_csv(DATASET_PATH)

    colonnes_api = list(EmployeeInput.model_fields.keys())

    ligne = df[colonnes_api].iloc[0]

    return json.loads(
        ligne.to_json(force_ascii=False)
    )