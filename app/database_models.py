from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Employee(Base):
    """
    Dataset historique issu du Projet 4.
    Une ligne = un salarié du dataset DataFrameRH_clean.csv.
    """

    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    satisfaction_employee_environnement: Mapped[int] = mapped_column(Integer)
    note_evaluation_precedente: Mapped[int] = mapped_column(Integer)
    niveau_hierarchique_poste: Mapped[int] = mapped_column(Integer)
    satisfaction_employee_nature_travail: Mapped[int] = mapped_column(Integer)
    satisfaction_employee_equipe: Mapped[int] = mapped_column(Integer)
    satisfaction_employee_equilibre_pro_perso: Mapped[int] = mapped_column(Integer)
    note_evaluation_actuelle: Mapped[int] = mapped_column(Integer)

    heure_supplementaires: Mapped[str] = mapped_column(String(20))

    # Conservé sous forme texte dans la table historique car le CSV
    # contient la valeur source telle qu'elle existe dans le dataset.
    augementation_salaire_precedente: Mapped[str] = mapped_column(String(20))

    age: Mapped[int] = mapped_column(Integer)
    genre: Mapped[str] = mapped_column(String(10))
    revenu_mensuel: Mapped[int] = mapped_column(Integer)

    statut_marital: Mapped[str] = mapped_column(String(50))
    departement: Mapped[str] = mapped_column(String(100))
    poste: Mapped[str] = mapped_column(String(100))

    nombre_experiences_precedentes: Mapped[int] = mapped_column(Integer)
    annee_experience_totale: Mapped[int] = mapped_column(Integer)
    annees_dans_l_entreprise: Mapped[int] = mapped_column(Integer)
    annees_dans_le_poste_actuel: Mapped[int] = mapped_column(Integer)

    a_quitte_l_entreprise: Mapped[str] = mapped_column(String(10))

    nombre_participation_pee: Mapped[int] = mapped_column(Integer)
    nb_formations_suivies: Mapped[int] = mapped_column(Integer)
    nombre_employee_sous_responsabilite: Mapped[int] = mapped_column(Integer)
    distance_domicile_travail: Mapped[int] = mapped_column(Integer)
    niveau_education: Mapped[int] = mapped_column(Integer)

    domaine_etude: Mapped[str] = mapped_column(String(100))
    frequence_deplacement: Mapped[str] = mapped_column(String(50))

    annees_depuis_la_derniere_promotion: Mapped[int] = mapped_column(Integer)
    annes_sous_responsable_actuel: Mapped[int] = mapped_column(Integer)


class PredictionInput(Base):
    """
    Données brutes reçues par POST /predict.
    Elles correspondent aux 26 champs de EmployeeInput.
    """

    __tablename__ = "prediction_inputs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    satisfaction_employee_environnement: Mapped[int] = mapped_column(Integer)
    note_evaluation_precedente: Mapped[int] = mapped_column(Integer)
    niveau_hierarchique_poste: Mapped[int] = mapped_column(Integer)
    satisfaction_employee_nature_travail: Mapped[int] = mapped_column(Integer)
    satisfaction_employee_equipe: Mapped[int] = mapped_column(Integer)
    satisfaction_employee_equilibre_pro_perso: Mapped[int] = mapped_column(Integer)
    note_evaluation_actuelle: Mapped[int] = mapped_column(Integer)

    heure_supplementaires: Mapped[str] = mapped_column(String(20))
    augementation_salaire_precedente: Mapped[float] = mapped_column(Float)

    age: Mapped[int] = mapped_column(Integer)
    genre: Mapped[str] = mapped_column(String(10))
    revenu_mensuel: Mapped[int] = mapped_column(Integer)

    statut_marital: Mapped[str] = mapped_column(String(50))
    departement: Mapped[str] = mapped_column(String(100))
    poste: Mapped[str] = mapped_column(String(100))

    nombre_experiences_precedentes: Mapped[int] = mapped_column(Integer)
    annee_experience_totale: Mapped[int] = mapped_column(Integer)
    annees_dans_l_entreprise: Mapped[int] = mapped_column(Integer)

    nombre_participation_pee: Mapped[int] = mapped_column(Integer)
    nb_formations_suivies: Mapped[int] = mapped_column(Integer)
    nombre_employee_sous_responsabilite: Mapped[int] = mapped_column(Integer)
    distance_domicile_travail: Mapped[int] = mapped_column(Integer)
    niveau_education: Mapped[int] = mapped_column(Integer)

    domaine_etude: Mapped[str] = mapped_column(String(100))
    frequence_deplacement: Mapped[str] = mapped_column(String(50))

    annees_depuis_la_derniere_promotion: Mapped[int] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )


class PredictionOutput(Base):
    """
    Résultat généré par le modèle pour un PredictionInput donné.
    """

    __tablename__ = "prediction_outputs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    input_id: Mapped[int] = mapped_column(
        ForeignKey("prediction_inputs.id"),
        nullable=False,
        unique=True,
    )

    prediction: Mapped[int] = mapped_column(Integer)
    probabilite_attrition: Mapped[float] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )

