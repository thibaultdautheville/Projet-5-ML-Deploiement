from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg://thibaultdautheville@localhost/projet5_ml"

engine = create_engine(DATABASE_URL)

with engine.connect() as connection:
    result = connection.execute(text("SELECT 1"))
    value = result.scalar()

print("Connexion PostgreSQL : OK")
print("Résultat SELECT 1 :", value)
