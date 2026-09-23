from app.database import engine
from app.database_models import Base


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("Tables PostgreSQL créées avec succès.")


if __name__ == "__main__":
    create_tables()