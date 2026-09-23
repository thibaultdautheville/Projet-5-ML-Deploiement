import os

import pytest
from sqlalchemy import create_engine, text

from app.database import DATABASE_URL


@pytest.mark.integration
def test_connexion_postgresql_reelle():
    """
    Test d'intégration facultatif avec une vraie instance PostgreSQL.

    Il n'est exécuté que si RUN_POSTGRES_TESTS=1.
    """

    if os.getenv("RUN_POSTGRES_TESTS") != "1":
        pytest.skip(
            "PostgreSQL réel non demandé sur cette machine."
        )

    test_engine = create_engine(
        DATABASE_URL,
        connect_args={"connect_timeout": 3},
    )

    with test_engine.connect() as connection:
        resultat = connection.execute(text("SELECT 1"))

        assert resultat.scalar() == 1