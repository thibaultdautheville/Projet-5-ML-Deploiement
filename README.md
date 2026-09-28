# Projet 5 - Déploiement d'un modèle de Machine Learning

Ce projet a pour objectif de rendre opérationnel un modèle de Machine Learning de prédiction du départ des employés.

Le modèle XGBoost entraîné lors d'un projet précédent est exposé à travers une API REST développée avec FastAPI.

Le projet intègre également :

- le prétraitement des données nécessaires au modèle ;
- la validation des données d'entrée avec Pydantic ;
- la persistance des prédictions dans une base PostgreSQL ;
- des tests unitaires, fonctionnels et d'intégration avec Pytest ;
- un contrôle de non-régression des performances du modèle ;
- une gestion reproductible des dépendances avec `uv`.

La mise en place de la CI/CD et du déploiement sur Hugging Face Spaces complète l'architecture du projet.

---

## Architecture générale

Le fonctionnement de l'application est le suivant :

```text
Utilisateur
    |
    v
API FastAPI
    |
    +--> Validation Pydantic
    |
    +--> Prétraitement des variables
    |
    +--> Modèle XGBoost
    |
    +--> Prédiction
    |
    v
PostgreSQL