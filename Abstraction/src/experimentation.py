import os
import json
import pandas as pd
import time
import sys
from boite import Boite
from build_boite import BoitePropagator
from stable import StabilityChecker
from monotonicity_checker import MonotonicityChecker


def extract_instance_from_boite(boite:Boite, mode='min'):
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

def printCE(c_exemple):
    if c_exemple is not None :
        class_id = c_exemple[0]
        broken = c_exemple[1]
        boite = broken[0]
        result =broken[1]

    
        fmin= extract_instance_from_boite(boite)
        fmax = extract_instance_from_boite(boite,"max")
        between = extract_instance_from_boite(result['boite'])

        s1= "instance min, a = "+ str((fmin,class_id)) + "\n"
        s2 = "instance between, b = " + str((between,result['prediction']))+"\n"
        s3= "instance max, c = " +str((fmax,class_id))+"\n"
        s4= "CONCLUSION \n"
        s5="We have a < b < c and k(a) = k(c) or k(a)!= k(b) so we conclude that the model isn't stable"
        return s1 + s2 + s3 + s4 + s5
    return "All is Ok "




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

    # Taille du fichier modèle
    model_size = os.path.getsize(model_path) / 1024  # en Ko

    # Vérification stabilité
    stable_checker = StabilityChecker(final_boites, propagateur,model_path)
    stable, _ = stable_checker.verif_stable()
    end = time.time()

    # # Vérification monotonie
    # monotone_checker = MonotonicityChecker(final_boites, model_path, order_classes)
    # monotone = monotone_checker.verif_monotone()

    c_exemple = None
    if stable_checker.contre_exemple:
        for class_id,cex in stable_checker.contre_exemple.items():
            c_exemple = (class_id,cex[0])
            break


    return {
        "dataset": os.path.basename(dataset_path),
        "model": os.path.basename(model_path),
        "stable": stable,
        "taux_stability":stable_checker.taux_stability,
        "c_exemple": c_exemple,
        # "monotone": monotone,
        "features": nb_features,
        "time_execution": (end -star),
        "boites": nb_boites,
        "model_size_kb": round(model_size, 2)
    }


def experimentation_batch(dossier_datasets, dossier_models,chemin_resultat="resultas"):
    """
    Lance l'expérimentation sur tous les datasets et modèles correspondants.
    """
    fichiers_datasets = sorted([f for f in os.listdir(dossier_datasets) if f.endswith('.csv')])
    fichiers_models = sorted([f for f in os.listdir(dossier_models) if f.endswith('.json')])

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
            f.write(f"- Stabilité : {'OUI' if r['stable'] else 'NON'}\n")
            f.write(f"- Taux Stabilité : {r['taux_stability']}\n")
            # f.write(f"- Monotonie : {'OUI' if r['monotone'] else 'NON'}\n")
            f.write(f"- Nombre de features : {r['features']}\n")
            f.write(f"- Nombre de boîtes : {r['boites']}\n")
            f.write(f"Temps d'execution : {r['time_execution']}\n")
            f.write(f"- Taille du modèle : {r['model_size_kb']} Ko\n")
            f.write(f"contre exemple : \n {printCE(r['c_exemple'])} \n")
            f.write("-" * 40 + "\n")

    print(f"\n✅ Résultats sauvegardés dans {chemin_resultat}")


if sys.argv[1] and sys.argv[2] :
    experimentation_batch(sys.argv[1],sys.argv[2])


