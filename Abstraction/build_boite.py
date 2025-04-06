from boite import Boite
from arbre import propagate_boites_in_tree
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, cpu_count

def softmax(logits):
    logits = np.array(logits)
    e = np.exp(logits - np.max(logits))
    return e / np.sum(e)

def charger_arbres_et_classes(model):
    arbres = model["learner"]["gradient_booster"]["model"]["trees"]
    tree_info = model["learner"]["gradient_booster"]["model"]["tree_info"]
    num_classes = max(tree_info) + 1
    return list(zip(arbres, tree_info)), num_classes

def propagate_boites_cascade(model_json_path, boite_init, verbose=True):
    import time
    start = time.time()
    with open(model_json_path, "r") as f:
        model = json.load(f)

    arbres, num_classes = charger_arbres_et_classes(model)

    boite_logit_pairs = [
        {
            "boite": boite_init,
            "logits": np.zeros(num_classes)
        }
    ]

    pool = Pool(cpu_count())

    for idx, (arbre_json, class_id) in enumerate(tqdm(arbres, desc="🌲 Arbres")):
        input_batch = [(pair["boite"], pair["logits"]) for pair in boite_logit_pairs]
        tasks = [(arbre_json, [item], class_id) for item in input_batch]
        all_results = pool.starmap(propagate_boites_in_tree, tasks)
        boite_logit_pairs = [{"boite": b, "logits": logits} for sublist in all_results for b, logits in sublist]

        if verbose:
            tqdm.write(f"🔁 Étape {idx+1}/{len(arbres)} — {len(boite_logit_pairs)} boîtes actives")

    pool.close()
    pool.join()

    # ➕ Analyse des boîtes finales
    def boite_figee(boite):
        return all(a == b for a, b in boite.bornes.values())

    nb_figees = sum(1 for pair in boite_logit_pairs if boite_figee(pair["boite"]))
    nb_total = len(boite_logit_pairs)

    print(f"\n📦 Boîtes finales : {nb_total} au total")
    print(f"🤪 Dont figées (a == b partout) : {nb_figees}")

    for i, pair in enumerate(boite_logit_pairs):
        boite = pair["boite"]
        figées = [f"f{f}" for f, (a, b) in boite.bornes.items() if a == b]
        print(f"  ▪️ Boîte {i} — {len(figées)} features figées : {figées}")

    # ➕ Retour des résultats finaux
    final_results = []
    for pair in boite_logit_pairs:
        logits = pair["logits"]
        probas = softmax(logits)
        pred_class = int(np.argmax(probas))
        final_results.append({
            "boite": pair["boite"],
            "logits": logits.tolist(),
            "probas": probas.tolist(),
            "prediction": pred_class
        })
    end = time.time()
    print(f"\n⏱ Temps total d'exécution : {end - start:.2f} secondes")

    return final_results

def regrouper_boites_par_classe(resultats):
    boites_par_classe = defaultdict(list)

    for resultat in resultats:
        classe = resultat["prediction"]
        boite = resultat["boite"]
        boites_par_classe[classe].append(boite)

    return dict(boites_par_classe)
