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

from xgboost import XGBClassifier
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
MODEL_PATH = Path(__file__).parent.parent / "models" / "xgb_model_v2.json"

# ------------------------------------------------------------------
# CHARGEMENT DU MODÈLE (une seule fois, au démarrage de l'API)
# ------------------------------------------------------------------

def load_model(model_path: Path = MODEL_PATH):
    """Charge le modele natif et controle les noms et leur ordre."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Modele introuvable : {model_path}")

    modele = XGBClassifier()
    modele.load_model(model_path)

    if modele.get_booster().feature_names != load_feature_names():
        raise ValueError("Les variables du modele et du JSON different.")

    return modele


# ------------------------------------------------------------------
# FEATURE ENGINEERING (réplique exacte du notebook)
# ------------------------------------------------------------------

def _one_hot_manual(df: pd.DataFrame, col: str, modalites: list) -> pd.DataFrame:
    """
    Réplique pd.get_dummies(df[col], drop_first=True) mais de façon
    déterministe : les colonnes générées sont TOUJOURS les mêmes,
    même si une modalité est absente du batch actuel (ex: 1 seule ligne
    en prédiction temps réel).
    """
    for modalite in modalites:
        nom_colonne = f"{col}_{modalite}"
        df[nom_colonne] = (df[col] == modalite).astype(int)
    return df.drop(columns=[col])


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
    X2['heure_supplementaires'] = X2['heure_supplementaires'].map({'Non': 0, 'Oui': 1})
    X2['frequence_deplacement'] = X2['frequence_deplacement'].map(ORDRE_DEPLACEMENT)

    for col, modalites in CATEGORIES_ONE_HOT.items():
        X2 = _one_hot_manual(X2, col, modalites)

    # --- Classes dérivées (expérience, augmentation) ---
    X2 = _add_experience_classe(X2)
    X2 = _add_augmentation_classe(X2)   


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



import json

FEATURE_NAMES_PATH = Path(__file__).parent.parent / "models" / "feature_names.json"

def load_feature_names(path: Path = FEATURE_NAMES_PATH) -> list:
    """
    Charge la liste ordonnée des features attendues par le modèle.
    Remplace model.feature_names_in_ car le modèle XGBoost a été
    entraîné sur un array numpy (sans noms de colonnes), pas un DataFrame.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def predict(model, df: pd.DataFrame, feature_names: list = None):
    X2 = preprocess_input(df)

    if feature_names is None:
        feature_names = load_feature_names()

    colonnes_manquantes = set(feature_names) - set(X2.columns)
    if colonnes_manquantes:
        raise ValueError(f"Colonnes manquantes après preprocessing : {colonnes_manquantes}")

    X2 = X2[feature_names]  # force l'ordre exact attendu par le modèle

    predictions = model.predict(X2)
    probabilities = model.predict_proba(X2)[:, 1]

    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }

# ------------------------------------------------------------------
# CLASSES DÉRIVÉES (bins définis dans le notebook d'entraînement)
# ------------------------------------------------------------------

BINS_EXPERIENCE = [-1, 0, 2, 5, 9]
LABELS_EXPERIENCE = ['Aucune_experience', 'Junior', 'Confirme', 'Senior']

BINS_AUGMENTATION = [10, 14, 18, 22, 25]
LABELS_AUGMENTATION = ['Faible_10-14%', 'Moderee_15-18%', 'Forte_19-22%', 'Exceptionnelle_23-25%']


def _add_experience_classe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dérive 'experience_classe' à partir de 'nombre_experiences_precedentes',
    puis génère les colonnes one-hot correspondantes (sans drop_first,
    car dans le notebook toutes les modalités sont gardées ici).
    """
    classe = pd.cut(
        df['nombre_experiences_precedentes'],
        bins=BINS_EXPERIENCE,
        labels=LABELS_EXPERIENCE,
    )

    if classe.isna().any():
        raise ValueError(
            "nombre_experiences_precedentes hors bornes attendues "
            f"(bins={BINS_EXPERIENCE}). Valeur reçue : "
            f"{df['nombre_experiences_precedentes'].tolist()}"
        )

    for label in LABELS_EXPERIENCE:
        df[f"experience_classe_{label}"] = (classe == label).astype(int)

    return df


def _add_augmentation_classe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie 'augementation_salaire_precedente' (retire le '%' si présent),
    dérive 'augmentation_classe', puis génère les colonnes one-hot.
    """
    valeurs = (
        df['augementation_salaire_precedente']
        .astype(str)
        .str.replace('%', '', regex=False)
        .str.strip()
        .astype(float)
    )
    df['augementation_salaire_precedente'] = valeurs  # remplace la colonne brute par la version numérique

    classe = pd.cut(
        valeurs,
        bins=BINS_AUGMENTATION,
        labels=LABELS_AUGMENTATION,
    )

    if classe.isna().any():
        raise ValueError(
            "augementation_salaire_precedente hors bornes attendues "
            f"(bins={BINS_AUGMENTATION}). Valeur reçue : {valeurs.tolist()}"
        )

    for label in LABELS_AUGMENTATION:
        df[f"augmentation_classe_{label}"] = (classe == label).astype(int)

    return df