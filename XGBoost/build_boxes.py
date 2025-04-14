import json
import numpy as np
from math import ceil
from tqdm import tqdm
from collections import defaultdict
from boite import Boite
from arbre import propagate_boites_in_tree

def softmax(logits):
    logits = np.array(logits)
    e = np.exp(logits - np.max(logits))
    return e / np.sum(e)

class BoitePropagator:
    def __init__(self, model_json_path, boite_init, batch_size=1000, verbose=True):
        self.model_json_path = model_json_path
        self.boite_init = boite_init
        self.batch_size = batch_size
        self.verbose = verbose

        self.model = self._load_model()
        self.arbres, self.num_classes = self._extract_trees()

    def _load_model(self):
        with open(self.model_json_path, "r") as f:
            return json.load(f)

    def _extract_trees(self):
        arbres = self.model["learner"]["gradient_booster"]["model"]["trees"]
        tree_info = self.model["learner"]["gradient_booster"]["model"]["tree_info"]
        num_classes = max(tree_info) + 1
        return list(zip(arbres, tree_info)), num_classes

    def run(self):
        import time
        start = time.time()

        boite_logit_pairs = [{
            "boite": self.boite_init,
            "logits": np.zeros(self.num_classes)
        }]

        for idx, (arbre_json, class_id) in enumerate(tqdm(self.arbres, desc="🌲 Arbres")):
            input_batch = [(pair["boite"], pair["logits"]) for pair in boite_logit_pairs]
            new_pairs = []

            for i in range(0, len(input_batch), self.batch_size):
                batch = input_batch[i:i + self.batch_size]
                result = propagate_boites_in_tree(arbre_json, batch, class_id)
                new_pairs.extend([{"boite": b, "logits": logits} for b, logits in result])

            boite_logit_pairs = new_pairs

            if self.verbose:
                tqdm.write(f"🔁 Étape {idx + 1}/{len(self.arbres)} — {len(boite_logit_pairs)} boîtes actives")

        results = []
        for pair in boite_logit_pairs:
            logits = pair["logits"]
            probas = softmax(logits)
            pred_class = int(np.argmax(probas))
            results.append({
                "boite": pair["boite"],
                "logits": logits.tolist(),
                "probas": probas.tolist(),
                "prediction": pred_class
            })

        end = time.time()
        print(f"\n⏱ Temps total d'exécution : {end - start:.2f} secondes")
        return results

    @staticmethod
    def regrouper_boites_par_classe(resultats):
        boites_par_classe = defaultdict(list)
        for resultat in resultats:
            classe = resultat["prediction"]
            boite = resultat["boite"]
            boites_par_classe[classe].append(boite)
        return dict(boites_par_classe)



