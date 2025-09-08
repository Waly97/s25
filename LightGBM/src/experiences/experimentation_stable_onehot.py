import sys, os
# Ajouter la RACINE du projet (deux niveaux au-dessus de ce fichier) au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pandas as pd
import time
import argparse

from Abstraction.src.verification.boite import Boite
from src.verification.propagator import BoitePropagator
from Abstraction.src.verification.stable_improve import StabilityChecker
from Abstraction.src.verification.utils import detect_onehot_groups_from_dataset

"""
Usage:

python3 src/experiences/experience_one_hot.py 'dossier jeux de données one hot' 'dossier model'

Exemple : python3 src/experiences/experimentation_stable_onehot.py dataset model_one_hot
"""


def extract_instance_from_boite(boite: Boite, mode='min'):
    """
    Extrait une instance (vecteur) à partir d'une boîte.
    mode: 'min' ou 'max'
    """
    if mode == 'min':
        return [bounds[0] for bounds in boite.bornes.values()]
    elif mode == 'max':
        return [bounds[1] for bounds in boite.bornes.values()]
    else:
        raise ValueError("mode must be 'min' or 'max'")

# Affichage contre exemple
def printCE(c_exemple):
    if c_exemple is not None:
        class_id = c_exemple[0]
        broken = c_exemple[1]
        boite = broken[0]
        result = broken[1]

        fmin = extract_instance_from_boite(boite)
        fmax = extract_instance_from_boite(boite, "max")
        between = extract_instance_from_boite(result['boite'])

        s1 = "instance min, a = " + str((fmin, class_id)) + "\n"
        s2 = "instance between, b = " + str((between, result['prediction'])) + "\n"
        s3 = "instance max, c = " + str((fmax, class_id)) + "\n"
        s4 = "CONCLUSION \n"
        s5 = "We have a < b < c and k(a) = k(c) or k(a)!= k(b) so we conclude that the model isn't stable"
        return s1 + s2 + s3 + s4 + s5
    return "All is Ok "


def tester_un_modele(dataset_path, model_path):
    """
    Teste stabilité + monotonie pour un modèle et retourne les résultats.
    """
    print(f"--- Test sur modèle {os.path.basename(model_path)} ---")
    star = time.time()

    # Charger dataset et modèle
    groupe_one_hot = detect_onehot_groups_from_dataset(dataset_path)
    boite_init = Boite.creer_boite_initiale_depuis_dataset(dataset_path)
    propagateur = BoitePropagator(model_path, boite_init, group_one_hot=groupe_one_hot)
    resultats = propagateur.run()

    taux_stabilite = 0.0
    # Nombre de features
    df = pd.read_csv(dataset_path)
    nb_features = df.shape[1] - 1  # dernière colonne = label

    c_exemple = None
    is_stable = True
    nb_boites = 0

    for k in range(len(resultats)):
        # Nombre de boîtes finales
        final_boites = BoitePropagator.regrouper_boites_par_classe(resultats[k])
        nb_boites += len(resultats[k])
        # Vérification stabilité
        stable_checker = StabilityChecker(final_boites, propagateur, model_path)
        stable, _ = stable_checker.verif_stable()
        taux_stabilite += stable_checker.taux_stability
        if not stable:
            is_stable = False
            if stable_checker.contre_exemple:
                for class_id, cex in stable_checker.contre_exemple.items():
                    c_exemple = (class_id, cex[0])
                    break

    # Taille du fichier modèle
    model_size = os.path.getsize(model_path) / 1024.0  # en Ko
    end = time.time()

    return {
        "dataset": os.path.basename(dataset_path),
        "model": os.path.basename(model_path),
        "stable": is_stable,
        "taux_stability": (taux_stabilite / max(1, len(resultats))),
        "c_exemple": c_exemple,
        "features": nb_features,
        "time_execution": (end - star),
        "boites": nb_boites,
        "model_size_kb": round(model_size, 2),
    }


def experimentation_batch(dossier_datasets, dossier_models, chemin_resultat="resultas_one_hot"):
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

    # Enregistrement dans un fichier (UTF-8 pour la compatibilité)
    with open(chemin_resultat, "w", encoding="utf-8") as f:
        f.write("==== Résultats de l'expérimentation ====\n\n")
        for r in resultats:
            f.write(f"Dataset : {r['dataset']}\n")
            f.write(f"Modèle  : {r['model']}\n")
            f.write(f"- Stabilité : {'OUI' if r['stable'] else 'NON'}\n")
            f.write(f"- Taux Stabilité : {r['taux_stability']}\n")
            f.write(f"- Nombre de features : {r['features']}\n")
            f.write(f"- Nombre de boîtes : {r['boites']}\n")
            f.write(f"Temps d'execution : {r['time_execution']}\n")
            f.write(f"- Taille du modèle : {r['model_size_kb']} Ko\n")
            f.write(f"contre exemple : \n {printCE(r['c_exemple'])} \n")
            f.write("-" * 40 + "\n")

    print(f"\n✅ Résultats sauvegardés dans {chemin_resultat}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Tester stabilité/monotonie de modèles")
    parser.add_argument("datasets_dir", help="Dossier contenant les .csv")
    parser.add_argument("models_dir", help="Dossier contenant les .json")
    parser.add_argument("-o", "--output", default="resultas_one_hot",
                        help="Chemin du fichier de résultats (par défaut: resultas_one_hot)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    experimentation_batch(args.datasets_dir, args.models_dir, args.output)


if __name__ == "__main__":
    # Recommandé pour compat Windows / freeze. Inoffensif ailleurs.
    import multiprocessing as mp
    mp.freeze_support()
    # Laisser spawn (par défaut sur macOS). Si tu veux forcer :
    # mp.set_start_method("spawn", force=True)
    main()
