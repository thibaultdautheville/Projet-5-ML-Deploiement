"""
model_loader.py

Charge le modèle de ML entraîné (P4) et applique les transformations
nécessaires pour produire une prédiction à partir de données brutes
(format proche du CSV original).

Toutes les valeurs "figées" ci-dessous (médiane d'âge, mapping de genre,
ordre des déplacements) proviennent du notebook d'entraînement
S_intermediare_2.ipynb et NE DOIVENT PAS être recalculées dynamiquement
en production, sous peine de "train-serving skew".
"""

import joblib
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# CONSTANTES FIGÉES (issues du notebook d'entraînement)
# ------------------------------------------------------------------

# Médiane de l'âge calculée sur le jeu d'entraînement (P4)
AGE_MEDIAN_TRAIN = 36.0

# Mapping binaire du genre
GENRE_MAPPING = {'F': 0, 'M': 1}

# Modalités one-hot à générer (drop_first=True → la modalité de référence n'est PAS dans la liste)
CATEGORIES_ONE_HOT = {
    'statut_marital': ['Divorcé(e)', 'Marié(e)'],  # référence : Célibataire
    'departement': ['Consulting', 'Ressources Humaines'],  # référence : Commercial
    'poste': [
        'Cadre Commercial', 'Consultant', 'Directeur Technique',
        'Manager', 'Représentant Commercial', 'Ressources Humaines',
        'Senior Manager', 'Tech Lead'
    ],  # référence : Assistant de Direction
    'domaine_etude': [
        'Entrepreunariat', 'Infra & Cloud', 'Marketing',
        'Ressources Humaines', 'Transformation Digitale'
    ],  # référence : Autre
}

# Mapping ordinal de la fréquence de déplacement
ORDRE_DEPLACEMENT = {'Aucun': 0, 'Occasionnel': 1, 'Frequent': 2}

# Colonnes utilisées pour l'indice de satisfaction globale
COLS_SATISFACTION = [
    'satisfaction_employee_environnement',
    'satisfaction_employee_nature_travail',
    'satisfaction_employee_equipe',
    'satisfaction_employee_equilibre_pro_perso',
]

# Chemin par défaut vers le modèle sérialisé
MODEL_PATH = Path(__file__).parent / "model" / "model.joblib"


# ------------------------------------------------------------------
# CHARGEMENT DU MODÈLE (une seule fois, au démarrage de l'API)
# ------------------------------------------------------------------

def load_model(model_path: Path = MODEL_PATH):
    """
    Charge le modèle sérialisé (joblib) depuis le disque.
    À appeler une seule fois au démarrage de l'application FastAPI,
    pas à chaque requête (coûteux en I/O).
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable à l'emplacement : {model_path}. "
            "Vérifie que le fichier a bien été exporté depuis le notebook "
            "(joblib.dump(model, 'model.joblib'))."
        )
    return joblib.load(model_path)


# ------------------------------------------------------------------
# FEATURE ENGINEERING (réplique exacte du notebook)
# ------------------------------------------------------------------

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique EXACTEMENT les mêmes transformations que celles faites
    dans le notebook d'entraînement, dans le même ordre.

    Paramètres
    ----------
    df : pd.DataFrame
        Un DataFrame contenant une ou plusieurs lignes, avec les colonnes
        brutes attendues (mêmes noms que le CSV original).

    Retourne
    --------
    pd.DataFrame
        Le DataFrame transformé, prêt à être passé à model.predict().
    """
    X2 = df.copy()

    # --- Encodage des variables catégorielles ---
    X2['genre'] = X2['genre'].map(GENRE_MAPPING)
    X2['frequence_deplacement'] = X2['frequence_deplacement'].map(ORDRE_DEPLACEMENT)

    # --- Feature engineering (7 features, ordre du notebook) ---

    # 1. Ratio d'ancienneté relative à l'âge
    X2['anciennete_ratio'] = X2['annees_dans_l_entreprise'] / X2['age']

    # 2. Stabilité dans le poste
    X2['stabilite_poste'] = X2['annees_dans_l_entreprise'] / (
        X2['nombre_experiences_precedentes'] + 1
    )

    # 3. Ecart entre expérience totale et ancienneté chez l'entreprise
    X2['ecart_experience_totale'] = (
        X2['annee_experience_totale'] - X2['annees_dans_l_entreprise']
    )

    # 4. Revenu rapporté à l'ancienneté
    X2['revenu_par_anciennete'] = X2['revenu_mensuel'] / (
        X2['annees_dans_l_entreprise'] + 1
    )

    # 5. Indice de satisfaction globale
    X2['indice_satisfaction_global'] = X2[COLS_SATISFACTION].mean(axis=1)

    # 6. Stagnation par rapport à la dernière promotion
    X2['stagnation_promotion'] = (
        X2['annees_dans_l_entreprise'] - X2['annees_depuis_la_derniere_promotion']
    )

    # 7. Profil "jeune à poste hiérarchique élevé"
    #    ATTENTION : on utilise la médiane FIGÉE de l'entraînement,
    #    PAS X2['age'].median() qui n'aurait aucun sens sur une seule ligne.
    X2['jeune_dans_poste_haut'] = (
        (X2['age'] <= AGE_MEDIAN_TRAIN) &
        (X2['niveau_hierarchique_poste'] >= 3)
    ).astype(int)

    return X2


# ------------------------------------------------------------------
# PRÉDICTION
# ------------------------------------------------------------------

def predict(model, df: pd.DataFrame):
    """
    Applique le preprocessing puis retourne la prédiction du modèle.

    Paramètres
    ----------
    model : objet modèle scikit-learn (ou compatible) déjà chargé
    df : pd.DataFrame brut (une ou plusieurs lignes)

    Retourne
    --------
    dict avec la/les prédiction(s) et la/les probabilité(s) associée(s)
    """
    X2 = preprocess_input(df)

    # ⚠️ Il faudra vérifier ici que X2 contient EXACTEMENT les colonnes
    # attendues par le modèle, dans le bon ordre. On le fera à l'étape
    # suivante avec model.feature_names_in_ (scikit-learn >= 1.0).
    predictions = model.predict(X2)
    probabilities = model.predict_proba(X2)[:, 1]  # proba classe positive

    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }