from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmployeeInput(BaseModel):
    """Donnees brutes necessaires a la prediction d'attrition."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "satisfaction_employee_environnement": 3,
                "note_evaluation_precedente": 3,
                "niveau_hierarchique_poste": 2,
                "satisfaction_employee_nature_travail": 4,
                "satisfaction_employee_equipe": 3,
                "satisfaction_employee_equilibre_pro_perso": 2,
                "note_evaluation_actuelle": 3,
                "heure_supplementaires": "Non",
                "augementation_salaire_precedente": 13,
                "age": 34,
                "genre": "M",
                "revenu_mensuel": 4200,
                "nombre_experiences_precedentes": 2,
                "annee_experience_totale": 8,
                "annees_dans_l_entreprise": 5,
                "nombre_participation_pee": 1,
                "nb_formations_suivies": 3,
                "nombre_employee_sous_responsabilite": 0,
                "distance_domicile_travail": 10,
                "niveau_education": 3,
                "frequence_deplacement": "Occasionnel",
                "annees_depuis_la_derniere_promotion": 1,
                "statut_marital": "C\u00e9libataire",
                "departement": "Commercial",
                "poste": "Cadre Commercial",
                "domaine_etude": "Marketing"
            }
        },
    )

    satisfaction_employee_environnement: int = Field(..., ge=1, le=4)
    note_evaluation_precedente: int = Field(..., ge=1, le=4)
    niveau_hierarchique_poste: int = Field(..., ge=1, le=5)
    satisfaction_employee_nature_travail: int = Field(..., ge=1, le=4)
    satisfaction_employee_equipe: int = Field(..., ge=1, le=4)
    satisfaction_employee_equilibre_pro_perso: int = Field(..., ge=1, le=4)
    note_evaluation_actuelle: int = Field(..., ge=1, le=4)

    heure_supplementaires: Literal["Non", "Oui"]
    augementation_salaire_precedente: float = Field(..., ge=10, le=25)

    age: int = Field(..., ge=18, le=70)
    genre: Literal["F", "M"]
    revenu_mensuel: float = Field(..., gt=0)

    nombre_experiences_precedentes: int = Field(..., ge=0)
    annee_experience_totale: int = Field(..., ge=0)
    annees_dans_l_entreprise: int = Field(..., ge=0)
    nombre_participation_pee: int = Field(..., ge=0)
    nb_formations_suivies: int = Field(..., ge=0)
    nombre_employee_sous_responsabilite: int = Field(..., ge=0)
    distance_domicile_travail: int = Field(..., ge=0)
    niveau_education: int = Field(..., ge=1)
    annees_depuis_la_derniere_promotion: int = Field(..., ge=0)

    frequence_deplacement: Literal["Aucun", "Occasionnel", "Frequent"]

    statut_marital: Literal[
        "C\u00e9libataire",
        "Mari\u00e9(e)",
        "Divorc\u00e9(e)",
    ]

    departement: Literal[
        "Commercial",
        "Consulting",
        "Ressources Humaines",
    ]

    poste: Literal[
        "Assistant de Direction",
        "Cadre Commercial",
        "Consultant",
        "Directeur Technique",
        "Manager",
        "Repr\u00e9sentant Commercial",
        "Ressources Humaines",
        "Senior Manager",
        "Tech Lead",
    ]

    domaine_etude: Literal[
        "Autre",
        "Entrepreunariat",
        "Infra & Cloud",
        "Marketing",
        "Ressources Humaines",
        "Transformation Digitale",
    ]

    @field_validator("augementation_salaire_precedente", mode="before")
    @classmethod
    def nettoyer_augmentation(cls, valeur):
        if isinstance(valeur, str):
            valeur = valeur.replace("%", "").strip()
        return valeur


class PredictionOutput(BaseModel):
    prediction: int = Field(..., ge=0, le=1)
    probabilite_attrition: float = Field(..., ge=0, le=1)
