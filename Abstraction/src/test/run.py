import sys, os
# Ajouter la RACINE du projet (deux niveaux au-dessus de ce fichier) au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from collections import defaultdict
from src.verification.boite import Boite
from src.verification.build_boite import  BoitePropagator
from src.verification.stable_improve import StabilityChecker
from src.verification.boite_model import CorrectedBoxClassifier
from src.verification.monotonicity_checker import MonotonicityChecker
import sys
import pandas as pd
import os
from src.verification.utils import detect_onehot_groups_from_dataset


"""
Usage :
python3 src/test/run_verif_stable.py   model  données 

Exemple :

python3 src/test/run_verif_stable.py model/CPU.json Dataset/CPU.py
"""

def get_model_path_from_dataset(dataset_path, model_dir="model_boite"):
    basename = os.path.basename(dataset_path)         
    name, _ = os.path.splitext(basename)              
    os.makedirs(model_dir, exist_ok=True)            
    return os.path.join(model_dir, f"{name}.json")   

def verif_stable(model,data,verbose=False):
    print(f"Vérification de la stabilité pour le modèle : {model}")
    boite_init = Boite.creer_boite_initiale_depuis_dataset(data)
    propagateur = BoitePropagator(model, boite_init)
    resultats = propagateur.run()
    boites_intermediaires = defaultdict(list)
    i=0
    valid=True
    if len(resultats) < 2:
        final_boites = BoitePropagator.regrouper_boites_par_classe(resultats[0])
        stable_checker = StabilityChecker(final_boites, propagateur,model)
        valid, _, _ = stable_checker.verif_stable()
    else:
        for k in range(len(resultats)) :
            final_boites = propagateur.regrouper_boites_par_classe(resultats[k])
            stability_checker = StabilityChecker(final_boites,propagateur,model)
            is_stable, boxes, f = stability_checker.verif_stable()
            if is_stable:
                for b in boxes:
                    boites_intermediaires[i].append(boxes)
                valid=True
            else:
                valid= False
                break
                i+=1
            if valid:
                taux= stability_checker.taux_stability
                if verbose and taux < 1:
                    data = pd.read_csv(dataset)
                    dataset = dataset.iloc[:, :-1]                     # Enlève la colonne "Class"
                    dataset = dataset.astype(float).values.tolist()
                    new_model = CorrectedBoxClassifier(boites_intermediaires, f)
                    #Génère le bon chemin
                    model_path = get_model_path_from_dataset(data)
                    # Sauvegarde
                    new_model.save_to_json(model)
                    print(f"✅ Modèle sauvegardé dans : {model}")
    return valid


def verif_mono(model,data):
    print(f"Verification de la monotonie pour le modele : {model}")
    # Charger dataset et modèle
    boite_init = Boite.creer_boite_initiale_depuis_dataset(data)
    propagateur = BoitePropagator(model, boite_init)
    resultats = propagateur.run()
    is_monotone = False
    if len(resultats) < 2:
        # Nombre de boîtes finales
        final_boites = BoitePropagator.regrouper_boites_par_classe(resultats[0])
        order = {i: i for i in range(len(final_boites))}
        # Vérification de la monotonie
        monotonie_checker = MonotonicityChecker(final_boites,propagateur,order,model)
        is_monotone = monotonie_checker.verif_monotone()
    else:
        for k in range(len(resultats)) :
            final_boites = propagateur.regrouper_boites_par_classe(resultats[k])
            stability_checker = StabilityChecker(final_boites,propagateur,model)
            is_stable, boxes, f = stability_checker.verif_stable()
            if is_stable:
                order = {i: i for i in range(len(final_boites))}
                monotonie_checker = MonotonicityChecker(final_boites,propagateur,order,model)
                if monotonie_checker.verif_monotone():
                    is_monotone= True
                else:
                    is_monotone=False
                    break
            else:
                is_monotone=False
                break
    return is_monotone

def my_Help():
    print("If you want to test stability tape S \n")
    print("If you want to test the montony tape M \n")
    print("If you encoding is in one-hot your answer is Y; If not it's N\n")
    print("If your answer is diferrent of what you see in check list the program stop\n")

def main():
    df = sys.argv[2]
    model =sys.argv[1]
    task = input("if you want to test the stabilility tape S; if it's for monotony tape M !  \n")
    encode = input("Is your encoding in one-hot (Y/N) \n")
    if encode=="Y":
        gr_one_hot = detect_onehot_groups_from_dataset(df)
    else:
        gr_one_hot= None
    if (task!="S" and task!="M") or (encode!="Y" and encode!="N"):
        my_Help()
    elif task=="M":
        verif_mono(model,df)
    elif task=="S":
        verif_stable(model,df,True)
    

if __name__ == "__main__":
    """
    profiling pour observer les fonction qui prend plus de temps pour l'optimisation du code 
    """
    main()
    #cProfile.run('main()','profiling_stats')
    #p=pstats.Stats('profiling_stats')
    # p.sort_stats('cumtime').print_stats(30)
