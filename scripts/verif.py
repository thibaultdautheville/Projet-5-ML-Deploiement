import json
with open("models/feature_names.json") as f:
    features_modele = set(json.load(f))

for prefix in ["statut_marital", "departement", "poste", "domaine_etude"]:
    print(prefix, "->", sorted([f for f in features_modele if f.startswith(prefix)]))