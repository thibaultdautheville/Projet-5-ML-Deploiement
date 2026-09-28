import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from app.model_loader import load_model, predict
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DATASET_PATH = "data/DataFrameRH_clean.csv"


def test_performance_modele_sur_jeu_test():
    # 1. Charger les données brutes
    df = pd.read_csv(DATASET_PATH)

    # 2. Reconstituer la cible binaire du Projet 4
    y = df["a_quitte_l_entreprise"].map({"Non": 0, "Oui": 1})

    # 3. Reconstituer exactement le split historique
    indices = df.index

    _, indices_test, _, y_test = train_test_split(
        indices,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. Récupérer uniquement les 294 salariés du jeu de test
    X_test = df.loc[indices_test].drop(columns=["a_quitte_l_entreprise"])

    assert len(X_test) == 294
    assert len(y_test) == 294
    assert int(y_test.sum()) == 47

    # 5. Faire passer ces données dans le pipeline réellement déployé
    model = load_model()
    resultat = predict(model, X_test)

    y_pred = resultat["predictions"]

    # 6. Calculer les métriques métier
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nRecall classe 1   : {recall:.3f}")
    print(f"Precision classe 1: {precision:.3f}")
    print(f"F1-score classe 1 : {f1:.3f}")        
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\nMatrice de confusion :")
    print(f"Vrais négatifs  : {tn}")
    print(f"Faux positifs   : {fp}")
    print(f"Faux négatifs   : {fn}")
    print(f"Vrais positifs  : {tp}")
    print(f"Prédictions 1   : {fp + tp} / {len(y_pred)}")


# 7. Garde-fous de non-régression du modèle déployé
#
# Référence observée sur le jeu de test historique :
# recall ≈ 0.915, precision ≈ 0.194, F1 ≈ 0.320.
# Les seuils ci-dessous servent à détecter une dégradation importante
# lors d'une modification future du modèle ou du preprocessing.
    assert recall >= 0.85
    assert precision >= 0.17
    assert f1 >= 0.28   # 7. Garde-fous de non-régr   assert recall >   assert precision >   assert f1 >= 0.40