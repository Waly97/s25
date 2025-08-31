import json
import numpy as np
from math import ceil
from tqdm import tqdm
from collections import defaultdict, Counter
from src.verification.boite import Boite
from src.verification.arbre import propagate_boites_in_tree 
import xgboost as xgb
from itertools import product
import pandas as pd
from concurrent.futures import ProcessPoolExecutor



def pred_boite_extremes_xgb(model, boite):
    """
    Prédiction pour les extrêmes (f_min et centre) d'une boîte.
    Utilise les features directement du modèle.
    """
    # Récupérer les noms de features depuis le modèle
    features = model.feature_names

    # Points f_min et centre, mappés aux bons noms
    fmin_point = {f: boite.bornes[int(i)][0] for i, f in enumerate(features)}
    fmax_point = {f: boite.bornes[int(i)][1] for i, f in enumerate(features)}

    # DataFrames pour les prédictions
    fmin_vec = pd.DataFrame([fmin_point])
    center_vec = pd.DataFrame([fmax_point])
    pred_classes = []
    for df in [fmin_vec, center_vec]:
        dmat = xgb.DMatrix(df, feature_names=features)
        pred = model.predict(dmat)
        if pred.ndim == 1:
            pred_class = int(pred[0] > 0.5)
        else:
            pred_class = int(np.argmax(pred))
        pred_classes.append(pred_class)

    if pred_classes[0] == pred_classes[1]:
        counts = {pred_classes[0]: 2}
        return pred_classes[0], counts
    else:
        print(f"⚠️ Incohérence f_min et f_max: {pred_classes}")
        print(boite)
        print("minimum ", fmin_point)
        print( "maximum",fmax_point)
        counts = {pred_classes[0]: 1, pred_classes[1]: 1}
        return None, counts

def generer_boites_init_avec_onehot(boite_initiale, groupes_onehot):
    """
    Génère toutes les boîtes initiales valides à partir d'une boîte initiale (avec bornes) et des groupes one-hot.
    🔄 Conversion des clés entières en chaînes si besoin (ex: 0 -> 'f0').
    """
    # Étape 1 — Conversion des clés entières en 'f0', 'f1'...
    bornes_init = {f"f{int(k)}" if isinstance(k, int) else k: v for k, v in boite_initiale.bornes.items()}
    
    # Étape 2 — Préparer les boîtes "activées" pour chaque groupe one-hot
    groupes_valeurs = []
    for groupe in groupes_onehot:
        valeurs = []
        for f in groupe:
            bornes = {c: [0, 0] for c in groupe}  # tout désactivé
            bornes[f] = [1, 1]                   # seul f activé
            valeurs.append(bornes)
        groupes_valeurs.append(valeurs)

    # Étape 3 — Récupérer les autres bornes (hors one-hot)
    autres_features = {
        k: v for k, v in bornes_init.items()
        if all(k not in groupe for groupe in groupes_onehot)
    }

    # Étape 4 — Générer toutes les combinaisons valides des groupes one-hot
    boites_valides = []
    for combinaison in product(*groupes_valeurs):
        merged = dict(autres_features)  # Copie des bornes hors one-hot
        for d in combinaison:
            merged.update(d)            # Mettre à jour avec un "choix" one-hot
        # ⚠️ Conserver l'ordre initial des bornes
        ordered_bornes = {int(k[1:]): merged[k] for k in bornes_init.keys()}
        boites_valides.append(Boite(ordered_bornes))

    return boites_valides


def _parallel_propagate(args):
    boite, model_bin_path, batch_size, group_one_hot, verbose = args

    # Créer un nouvel objet dans chaque processus
    propagator = BoitePropagator(
        model_bin_path=model_bin_path,
        boite_init=boite, 
        batch_size=batch_size,
        group_one_hot=group_one_hot,
        verbose=verbose
    )
    propagator.model_predictor.load_model(model_bin_path)
    return propagator.propagate([boite])

class BoitePropagator:
    def __init__(self, model_bin_path, boite_init, batch_size=2000,group_one_hot=None,verbose=True):
        self.model_bin_path = model_bin_path
        self.boite_init = boite_init
        self.batch_size = batch_size
        self.verbose = verbose
        self.group_one_hot=group_one_hot

        self.model_predictor = xgb.Booster()
        self.model_predictor.load_model(self.model_bin_path)
        self.nb_boite = 0
        self.arbres = None

    def run(self):
        import time
        start = time.time()

        if self.group_one_hot is not None:
            boite_list = generer_boites_init_avec_onehot(self.boite_init,self.group_one_hot)
        else:
            boite_list = [self.boite_init]

        print("le(s) boite(s) initiale(s) ", boite_list)

        result_final = []

        if len(boite_list) == 1 :

            results = self.propagate(boite_list)
            result_final.append(results)
        else:
             # 🧠 Préparer les arguments pour chaque processus
            args_list = [
                (boite, self.model_bin_path, self.batch_size, self.group_one_hot, self.verbose)
                for boite in boite_list
            ]

            with ProcessPoolExecutor() as executor:
                result_final = list(executor.map(_parallel_propagate, args_list))

        end = time.time()
        print(f"✅ Terminé en {end - start:.2f} secondes")
        return result_final
            


    def propagate_boite(self, boite):
        import time
        start = time.time()

        boite_list = [boite]

        print("Boite intermediaire : ", boite)

        # Charger la structure JSON du modèle
        model_json_path = self.model_bin_path.replace(".bin", ".json")
        with open(model_json_path, "r") as f:
            model_json = json.load(f)
        arbres = model_json["learner"]["gradient_booster"]["model"]["trees"]
        self.arbres = arbres

        for idx, arbre_json in enumerate(tqdm(self.arbres, desc="🔁 Itérations", ncols=80)):
            input_batch = boite_list
            num_batches = ceil(len(input_batch) / self.batch_size)
            new_boxes = []

            for i in range(num_batches):
                batch = input_batch[i * self.batch_size:(i + 1) * self.batch_size]
                for boite in batch:
                    new_boxes.extend(propagate_boites_in_tree(arbre_json, [(boite, None)], class_id=None))

            boite_list = [b for b, _ in new_boxes]

            if self.verbose:
                tqdm.write(f"🔁 Itération {idx + 1}/{len(arbres)} — {len(boite_list)} boîtes finales")

        results = []
        prediction_counter = Counter()

        for boite in boite_list:
            pred_class, class_counts = pred_boite_extremes_xgb(self.model_predictor, boite)
            if pred_class == None :
                continue
            prediction_counter[pred_class] += 1
            results.append({
                "boite": boite,
                "prediction": pred_class,
                "details": class_counts
            })

        end = time.time()
        print(f"\n⏱ Temps total d'exécution : {end - start:.2f} secondes")
        print("\n📊 Répartition des prédictions par classe (réelles par points) :")
        for cls, count in sorted(prediction_counter.items()):
            print(f"  Classe {cls} : {count} boîtes")

        return results
    
    def propagate(self,boite_list):
             # Charger la structure JSON du modèle
            model_json_path = self.model_bin_path.replace(".bin", ".json")
            with open(model_json_path, "r") as f:
                model_json = json.load(f)
            arbres = model_json["learner"]["gradient_booster"]["model"]["trees"]
            self.arbres = arbres
        # Propagation : partitionner les boîtes
            for idx, arbre_json in enumerate(tqdm(arbres, desc="🔁 Itérations", ncols=80)):
                input_batch = boite_list
                num_batches = ceil(len(input_batch) / self.batch_size)
                new_boxes = []

                for i in range(num_batches):
                    batch = input_batch[i * self.batch_size:(i + 1) * self.batch_size]
                    for boite in batch:
                        new_boxes.extend(propagate_boites_in_tree(arbre_json, [(boite, None)], class_id=None))

                boite_list = [b for b, _ in new_boxes]

                if self.verbose:
                    tqdm.write(f"🔁 Itération {idx + 1}/{len(arbres)} — {len(boite_list)} boîtes finales")

            self.nb_boite = len(boite_list)
            results = []
            prediction_counter = Counter()

            for boite in boite_list:
                pred_class, class_counts = pred_boite_extremes_xgb(self.model_predictor, boite)
                prediction_counter[pred_class] += 1
                results.append({
                    "boite": boite,
                    "prediction": pred_class,
                    "details": class_counts
                })
            return results

    @staticmethod
    def regrouper_boites_par_classe(resultats):
        boites_par_classe = defaultdict(list)
        for resultat in resultats:
            classe = resultat["prediction"]
            boite = resultat["boite"]
            boites_par_classe[classe].append(boite)
        return dict(boites_par_classe)
