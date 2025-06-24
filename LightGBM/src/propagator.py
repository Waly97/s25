import json
import numpy as np
from math import ceil
from tqdm import tqdm
from collections import defaultdict, Counter
from boite import Boite
from arbre import propagate_boites_in_tree  # Ce module doit faire uniquement la partition des boîtes
import lightgbm as lgb
from itertools import product

def pred_boite_reelle(model, boite, n_samples=500):
    """
    Échantillonne aléatoirement des points dans la boîte et prédit leur classe réelle.
    """
    features = list(boite.bornes.keys())
    data = []
    for _ in range(n_samples):
        point = [np.random.uniform(*boite.bornes[f]) for f in features]
        data.append(point)
    data = np.array(data)
    y_pred = model.predict(data)
    if y_pred.ndim == 1:
        y_pred_classes = (y_pred > 0.5).astype(int)
    else:
        y_pred_classes = np.argmax(y_pred, axis=1)
    unique, counts = np.unique(y_pred_classes, return_counts=True)
    classe_dominante = unique[np.argmax(counts)]
    return classe_dominante, dict(zip(unique, counts))

import lightgbm as lgb
import pandas as pd
import numpy as np

def pred_boite_extremes_lgb(model, boite):
    """
    Prédiction pour les extrêmes (f_min et f_max) d'une boîte avec LightGBM.
    Utilise les features directement du modèle.
    """
    # Récupérer les noms de features depuis le modèle
    features = model.feature_name()

     # Points f_min et centre, mappés aux bons noms
    fmin_point = {f: boite.bornes[int(i)][0] for i, f in enumerate(features)}
    fmax_point = {f: boite.bornes[int(i)][1] for i, f in enumerate(features)}

    # Créer les DataFrames
    fmin_df = pd.DataFrame([fmin_point])
    fmax_df = pd.DataFrame([fmax_point])

    pred_classes = []
    for df in [fmin_df, fmax_df]:
        pred = model.predict(df)
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
        print("maximum", fmax_point)
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

class BoitePropagator:
    def __init__(self, model_txt_path, boite_init,group_one_hot=None,batch_size=2000, verbose=True):
        self.model_txt_path = model_txt_path
        self.boite_init = boite_init
        self.batch_size = batch_size
        self.verbose = verbose
        self.group_one_hot=group_one_hot

        self.model_predictor = lgb.Booster(model_file=self.model_txt_path)
        self.nb_boite = 0
        self.arbres=None

    def run(self):
        """
        Propagation de la boîte initiale dans les arbres pour obtenir les boîtes finales.
        Puis, pour chaque boîte finale, échantillonne des points et prédit la classe réelle.
        """
        import time
        start = time.time()

        result_final = []

        if self.group_one_hot is not None :
             boite_list = generer_boites_init_avec_onehot(self.boite_init,self.group_one_hot)
        else:
            boite_list = [self.boite_init]
        if len(boite_list) == 1 :

            results = self.propagate(boite_list)
            result_final.append(results)
        else:
            for boite in boite_list:
                batch_boite =[]
                batch_boite.append(boite)
                results =self.propagate(batch_boite)
                result_final.append(results)
        return  result_final 

        
       
    
    def propagate_boite(self, boite):
        """
        Propagation de la boîte initiale dans les arbres pour obtenir les boîtes finales.
        Puis, pour chaque boîte finale, échantillonne des points et prédit la classe réelle.
        """
        import time
        start= time.time()
        boite_list = [boite]

        # Charger la structure des arbres (JSON)
        with open(self.model_txt_path.replace(".txt", ".json"), "r") as f:
            model_json = json.load(f)
        arbres = model_json["tree_info"]
        self.arbres=arbres

        # Propagation: partitionner les boîtes uniquement
        for idx, arbre_json in enumerate(tqdm(self.arbres, desc="🔁 Itérations", ncols=80)):
            input_batch = boite_list
            num_batches = ceil(len(input_batch) / self.batch_size)
            new_boxes = []

            for i in range(num_batches):
                batch = input_batch[i * self.batch_size:(i + 1) * self.batch_size]
                for boite in batch:
                    new_boxes.extend(propagate_boites_in_tree(arbre_json, [(boite, None)], class_id=None))  # la propagation ne touche plus aux scores

            boite_list = [b for b, _ in new_boxes]  # On ne garde que les boîtes partitionnées

            if self.verbose:
                tqdm.write(f"🔁 Itération {idx + 1}/{len(arbres)} — {len(boite_list)} boîtes finales")
        self.nb_boite = len(boite_list)
        results = []
        prediction_counter = Counter()

        for boite in boite_list:
            pred_class, class_counts = pred_boite_extremes_lgb(self.model_predictor, boite)
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
        import time
        start = time.time()
         # Charger la structure des arbres (JSON)
        with open(self.model_txt_path.replace(".txt", ".json"), "r") as f:
            model_json = json.load(f)
        arbres = model_json["tree_info"]
        self.arbres=arbres

        # Propagation: partitionner les boîtes uniquement
        for idx, arbre_json in enumerate(tqdm(arbres, desc="🔁 Itérations", ncols=80)):
            input_batch = boite_list
            num_batches = ceil(len(input_batch) / self.batch_size)
            new_boxes = []

            for i in range(num_batches):
                batch = input_batch[i * self.batch_size:(i + 1) * self.batch_size]
                for boite in batch:
                    new_boxes.extend(propagate_boites_in_tree(arbre_json, [(boite, None)], class_id=None))  # la propagation ne touche plus aux scores

            boite_list = [b for b, _ in new_boxes]  # On ne garde que les boîtes partitionnées

            if self.verbose:
                tqdm.write(f"🔁 Itération {idx + 1}/{len(arbres)} — {len(boite_list)} boîtes finales")

        self.nb_boite = len(boite_list)
        results = []
        prediction_counter = Counter()

        for boite in boite_list:
            pred_class, class_counts = pred_boite_extremes_lgb(self.model_predictor, boite)
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


    @staticmethod
    def regrouper_boites_par_classe(resultats):
        boites_par_classe = defaultdict(list)
        for resultat in resultats:
            classe = resultat["prediction"]
            boite = resultat["boite"]
            boites_par_classe[classe].append(boite)
        return dict(boites_par_classe)
