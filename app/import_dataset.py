from pathlib import Path

import pandas as pd
from sqlalchemy import func, select

from app.database import SessionLocal
from app.database_models import Employee


DATASET_PATH = Path("data/DataFrameRH_clean.csv")


def import_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATASET_PATH.resolve()}")

    df = pd.read_csv(DATASET_PATH)

    print("Dataset chargé :", df.shape)

    with SessionLocal() as session:
        nombre_existant = session.scalar(
            select(func.count()).select_from(Employee)
        )

        if nombre_existant > 0:
            print(
                f"Import annulé : la table employees contient déjà "
                f"{nombre_existant} lignes."
            )
            return

        employees = [
            Employee(**ligne)
            for ligne in df.to_dict(orient="records")
        ]

        session.add_all(employees)
        session.commit()

        nombre_final = session.scalar(
            select(func.count()).select_from(Employee)
        )

    print("Import PostgreSQL : OK")
    print("Nombre de lignes insérées :", nombre_final)


if __name__ == "__main__":
    import_dataset()