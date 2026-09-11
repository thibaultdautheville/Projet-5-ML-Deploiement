# app/schemas.py

from pydantic import BaseModel, Field
from typing import Optional


class EmployeeInput(BaseModel):
    """
    Schéma d'entrée pour la prédiction d'attrition.
    Représente les données brutes d'un employé, telles que saisies
    par l'utilisateur de l'API (avant feature engineering).
    """

    age: int = Field(..., ge=18, le=70, description="Âge de l'employé")
    annees_dans_l_entreprise: int = Field(
        ..., ge=0, description="Ancienneté totale dans l'entreprise (années)"
    )
    nombre_experiences_precedentes: int = Field(
        ..., ge=0, description="Nombre d'expériences professionnelles précédentes"
    )
    annee_experience_totale: int = Field(
        ..., ge=0, description="Nombre total d'années d'expérience (toutes entreprises confondues)"
    )
    revenu_mensuel: float = Field(
        ..., gt=0, description="Revenu mensuel de l'employé"
    )
    annees_depuis_la_derniere_promotion: int = Field(
        ..., ge=0, description="Nombre d'années depuis la dernière promotion"
    )
    niveau_hierarchique_poste: int = Field(
        ..., ge=1, le=5, description="Niveau hiérarchique du poste (1 à 5)"
    )

    satisfaction_employee_environnement: int = Field(
        ..., ge=1, le=4, description="Satisfaction vis-à-vis de l'environnement (1-4)"
    )
    satisfaction_employee_nature_travail: int = Field(
        ..., ge=1, le=4, description="Satisfaction vis-à-vis de la nature du travail (1-4)"
    )
    satisfaction_employee_equipe: int = Field(
        ..., ge=1, le=4, description="Satisfaction vis-à-vis de l'équipe (1-4)"
    )
    satisfaction_employee_equilibre_pro_perso: int = Field(
        ..., ge=1, le=4, description="Satisfaction vis-à-vis de l'équilibre pro/perso (1-4)"
    )

    poste: str = Field(..., description="Intitulé du poste occupé")
    statut_marital: str = Field(..., description="Statut marital de l'employé")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 34,
                "annees_dans_l_entreprise": 5,
                "nombre_experiences_precedentes": 2,
                "annee_experience_totale": 8,
                "revenu_mensuel": 4200.0,
                "annees_depuis_la_derniere_promotion": 1,
                "niveau_hierarchique_poste": 2,
                "satisfaction_employee_environnement": 3,
                "satisfaction_employee_nature_travail": 4,
                "satisfaction_employee_equipe": 3,
                "satisfaction_employee_equilibre_pro_perso": 2,
                "poste": "Cadre Commercial",
                "statut_marital": "Célibataire",
            }
        }


class PredictionOutput(BaseModel):
    """
    Schéma de sortie retourné par l'API après prédiction.
    """

    prediction: int = Field(..., description="0 = reste, 1 = quitte l'entreprise")
    probabilite_attrition: float = Field(
        ..., ge=0, le=1, description="Probabilité que l'employé quitte l'entreprise"
    )
    message: Optional[str] = Field(
        None, description="Message informatif (ex: champs manquants ignorés)"
    )