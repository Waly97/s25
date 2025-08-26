import sys, os
# Ajouter la RACINE du projet (deux niveaux au-dessus de ce fichier) au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from collections import defaultdict
from xgboost import XGBClassifier
from src.verification.boite import Boite
from src.verification.build_boite import  BoitePropagator
from src.verification.stable_improve import StabilityChecker
from src.verification.boite_model import CorrectedBoxClassifier
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
def main():
    df = sys.argv[2]
    model =sys.argv[1]
    gr_one_hot = detect_onehot_groups_from_dataset(df)
#     groupes_onehot = [
#     ["f3","f4","f5","f6"],
#     ["f7","f8","f9","f10","f11"]
# ]

    bt = Boite.creer_boite_initiale_depuis_dataset(df)
    
    print("boite initiale ", bt)
    buil_prop = BoitePropagator(model,bt,group_one_hot=gr_one_hot)

    result_final= buil_prop.run()


    i=1
    valid=False
    boite_intermediaire = defaultdict(list)
    for k in range(len(result_final)) :
        final_boites = buil_prop.regrouper_boites_par_classe(result_final[k])
        print("boite init ",i)
        stability_checker = StabilityChecker(final_boites,buil_prop,model)
        is_stable, boxes, f = stability_checker.verif_stable()
        
        if is_stable:
            # for cls,box in boxes.items():
            #     for b in box:
            #         boite_intermediaire[cls].append(b)
            valid=True
        else:
            valid= False
            break
        i+=1
    if valid:
        print("le modele est stable")
    #     dataset = pd.read_csv(df)
    #     dataset = dataset.iloc[:, :-1]                     # Enlève la colonne "Class"
    #     dataset = dataset.astype(float).values.tolist()
    #     new_model = CorrectedBoxClassifier(boite_intermediaire, f)
    #     #Génère le bon chemin
    #     model_path = get_model_path_from_dataset(df)

    #     # Sauvegarde
    #     new_model.save_to_json(model_path)
    #     print(f"✅ Modèle sauvegardé dans : {model_path}") 
    else :
        print("le modele n'est pas stable")

if __name__ == "__main__":
    """
    profiling pour observer les fonction qui prend plus de temps pour l'optimisation du code 
    """
    main()
    #cProfile.run('main()','profiling_stats')
    #p=pstats.Stats('profiling_stats')
    # p.sort_stats('cumtime').print_stats(30)
