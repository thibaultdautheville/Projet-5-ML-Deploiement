# Projet-5-ML-Déploiement
Je suis freelance spécialisé en machine learning et j'ai reçu une demande de la part de votre client Futurisys, une entreprise innovante qui souhaite rendre ses modèles de machine learning opérationnels et accessibles via une API performante. Je suis chargé de déployer un modèle de machine learning en production.



## 📁 Structure du dépôt

\```
.
├── app/            # Code source de l'API (FastAPI, modèle, BDD)
├── db/             # Scripts de création et peuplement de la BDD
├── tests/          # Tests unitaires et fonctionnels (Pytest)
├── docs/           # Documentation technique complémentaire
├── models/         # Modèle ML entraîné (.pkl)
└── .github/        # Pipeline CI/CD (GitHub Actions)


## ⚙️ Installation

### Prérequis
- Python 3.10+
- PostgreSQL 15+ (ou Docker)
- Git

### Étapes

1. Cloner le dépôt
\```bash
git clone https://github.com/thibaultdautheville/projet5-ml-deployment.git
cd projet5-ml-deployment
\```

2. Créer un environnement virtuel
\```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
\```

3. Installer les dépendances
\```bash
pip install -r requirements.txt
\```

4. Configurer les variables d'environnement
\```bash
cp .env.example .env
# Puis renseigner vos identifiants PostgreSQL dans .env
\```
