import os
import json
import pandas as pd
import time
from boite import Boite
from build_boite import BoitePropagator
from stable import StabilityChecker
from monotonicity_checker import MonotonicityChecker

def tester_un_modele(dataset_path, model_path, order_classes):
    """
    Teste stabilité + monotonie pour un modèle et retourne les résultats.
    """
    print(f"--- Test sur modèle {os.path.basename(model_path)} ---")

    # Charger dataset et modèle
    boite_init = Boite.creer_boite_initiale_depuis_dataset(dataset_path)
    propagateur = BoitePropagator(model_path, boite_init)
    resultats = propagateur.run()

    # Nombre de features
    df = pd.read_csv(dataset_path)
    nb_features = df.shape[1] - 1  # dernière colonne = label

    # Nombre de boîtes finales
    final_boites = BoitePropagator.regrouper_boites_par_classe(resultats)
    nb_boites = sum(len(lst) for lst in final_boites.values())

    # Taille du fichier modèle
    model_size = os.path.getsize(model_path) / 1024  # en Ko

    # Vérification stabilité
    stable_checker = StabilityChecker(final_boites, model_path)
    stable, _ = stable_checker.verif_stable()

    # Vérification monotonie
    monotone_checker = MonotonicityChecker(final_boites, model_path, order_classes)
    monotone = monotone_checker.verif_monotone()

    return {
        "dataset": os.path.basename(dataset_path),
        "model": os.path.basename(model_path),
        "stable": stable,
        "monotone": monotone,
        "features": nb_features,
        "boites": nb_boites,
        "model_size_kb": round(model_size, 2)
    }


def experimentation_batch(dossier_datasets, dossier_models, ordre_classes, chemin_resultat="resultats_experimentation.txt"):
    """
    Lance l'expérimentation sur tous les datasets et modèles correspondants.
    """
    fichiers_datasets = sorted([f for f in os.listdir(dossier_datasets) if f.endswith('.csv')])
    fichiers_models = sorted([f for f in os.listdir(dossier_models) if f.endswith('.json')])

    resultats = []

    for dataset_file, model_file in zip(fichiers_datasets, fichiers_models):
        dataset_path = os.path.join(dossier_datasets, dataset_file)
        model_path = os.path.join(dossier_models, model_file)

        resultat = tester_un_modele(dataset_path, model_path, ordre_classes)
        resultats.append(resultat)

    # Enregistrement dans un fichier
    with open(chemin_resultat, "w") as f:
        f.write("==== Résultats de l'expérimentation ====\n\n")
        for r in resultats:
            f.write(f"Dataset : {r['dataset']}\n")
            f.write(f"Modèle  : {r['model']}\n")
            f.write(f"- Stabilité : {'OUI' if r['stable'] else 'NON'}\n")
            f.write(f"- Monotonie : {'OUI' if r['monotone'] else 'NON'}\n")
            f.write(f"- Nombre de features : {r['features']}\n")
            f.write(f"- Nombre de boîtes : {r['boites']}\n")
            f.write(f"- Taille du modèle : {r['model_size_kb']} Ko\n")
            f.write("-" * 40 + "\n")

    print(f"\n✅ Résultats sauvegardés dans {chemin_resultat}")

