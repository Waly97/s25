import sys, os
# Ajouter la RACINE du projet (deux niveaux au-dessus de ce fichier) au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import json
import pandas as pd
import time
from collections import defaultdict, Counter
import sys
from Abstraction.src.verification.boite import Boite
from src.verification.propagator import BoitePropagator
from Abstraction.src.verification.monotonicity_checker import MonotonicityChecker
from src.experiences.verif_c_exemple import predict


"""
Usage:

python3 src/experiences/experience_monotonie.py 'dossier jeux de données' 'dossier model'

Exemple : python3 src/experience_monotonie.py Dataset model
"""

def tester_un_modele(dataset_path, model_path):
    """
    Teste stabilité + monotonie pour un modèle et retourne les résultats.
    """
    print(f"--- Test sur modèle {os.path.basename(model_path)} ---")
    import time
    star =  time.time()
    # Charger dataset et modèle
    boite_init = Boite.creer_boite_initiale_depuis_dataset(dataset_path)
    propagateur = BoitePropagator(model_path, boite_init)
    resultats = propagateur.run()

    # Nombre de features
    df = pd.read_csv(dataset_path)
    nb_features = df.shape[1] - 1  # dernière colonne = label

    # Nombre de boîtes finales
    final_boites = BoitePropagator.regrouper_boites_par_classe(resultats[0])
    nb_boites = propagateur.nb_boite

    order = {i: i for i in range(len(final_boites))}
    # Taille du fichier modèle
    model_size = os.path.getsize(model_path) / 1024  # en Ko
    # Vérification de la monotonie
    monotonie_checker = MonotonicityChecker(final_boites,propagateur,order,model_path)
    is_monotone = monotonie_checker.verif_monotone()
    end = time.time()

    if monotonie_checker.c_exemple:
        c_exemple = monotonie_checker.c_exemple
        s1= predict(model_path,c_exemple[0][0])
        s2= predict(model_path,c_exemple[0][1])
        s3= predict(model_path,c_exemple[1][0])
        s3= predict(model_path,c_exemple[1][1])


    return {
        "dataset": os.path.basename(dataset_path),
        "model": os.path.basename(model_path),
        "monotone": is_monotone,
        "features": nb_features,
        "time_execution": (end - star),
        "boites": nb_boites,
        "model_size_kb": round(model_size, 2)
    }


def experimentation_batch(dossier_datasets, dossier_models,chemin_resultat="LightGBM/resultats/monotonie"):
    """
    Lance l'expérimentation sur tous les datasets et modèles correspondants.
    """
    fichiers_datasets = sorted([f for f in os.listdir(dossier_datasets) if f.endswith('.csv')])
    fichiers_models = sorted([f for f in os.listdir(dossier_models) if f.endswith('.txt')])

    resultats = []

    for dataset_file, model_file in zip(fichiers_datasets, fichiers_models):
        dataset_path = os.path.join(dossier_datasets, dataset_file)
        model_path = os.path.join(dossier_models, model_file)

        resultat = tester_un_modele(dataset_path, model_path)
        resultats.append(resultat)

    # Enregistrement dans un fichier
    with open(chemin_resultat, "w") as f:
        f.write("==== Résultats de l'expérimentation ====\n\n")
        for r in resultats:
            f.write(f"Dataset : {r['dataset']}\n")
            f.write(f"Modèle  : {r['model']}\n")
            f.write(f"- Monotonie : {'OUI' if r['monotone'] else 'NON'}\n")
            f.write(f"- Nombre de features : {r['features']}\n")
            f.write(f"- Nombre de boîtes : {r['boites']}\n")
            f.write(f"Temps d'execution : {r['time_execution']}\n")
            f.write(f"- Taille du modèle : {r['model_size_kb']} Ko\n")
            f.write("-" * 40 + "\n")

    print(f"\n✅ Résultats sauvegardés dans {chemin_resultat}")


if sys.argv[1] and sys.argv[2] :
    experimentation_batch(sys.argv[1],sys.argv[2])


