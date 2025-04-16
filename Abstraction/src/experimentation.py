import time
from boite import Boite
from build_boite import BoitePropagator
from stable import StabilityChecker
import os

def tester_stabilite(model_path, boite_init, description):
    print(f"\n--- Test du modèle : {description} ---")
    start = time.time()

    propagateur = BoitePropagator(model_path, boite_init)
    resultats = propagateur.run()
    boxes_by_class = BoitePropagator.regrouper_boites_par_classe(resultats)

    stable = StabilityChecker(boxes_by_class, model_path)
    stable_intra = stable.verif_stable()

    duree = time.time() - start
    nb_arbres = len(propagateur.arbres)
    nb_boites = sum(len(bx) for bx in boxes_by_class.values())

    print(f"Stabilité intra-classe   : {'OK' if stable_intra else 'NON'}")
    print(f"Nombre d'arbres          : {nb_arbres}")
    print(f"Nombre total de boîtes   : {nb_boites}")
    print(f"Durée totale             : {duree:.2f} sec")

    return {
        "description": description,
        "fichier": os.path.basename(model_path),
        "nb_arbres": nb_arbres,
        "nb_boites": nb_boites,
        "intra_classe": stable_intra,
        "duree_sec": round(duree, 2)
    }

def test_models_batch(modeles, boites_inits):
    resume = []
    for i, (path, description) in enumerate(modeles):
        boite_init = Boite.creer_boite_initiale_depuis_dataset(boites_inits[i])
        resultat = tester_stabilite(path, boite_init, description)
        resume.append(resultat)
    return resume

def enregistrer_resultats(resultats, chemin_fichier="Abstraction/src/resultats/resultats_stabilite.txt"):
    with open(chemin_fichier, "w") as f:
        f.write("Résumé des vérifications de stabilité\n")
        f.write("="*40 + "\n")
        for res in resultats:
            f.write(f"Modèle : {res['description']}\n")
            f.write(f" - Fichier        : {res['fichier']}\n")
            f.write(f" - Nb arbres      : {res['nb_arbres']}\n")
            f.write(f" - Nb boîtes      : {res['nb_boites']}\n")
            f.write(f" - Stable         : {'OUI' if res['intra_classe'] else 'NON'}\n")
            f.write(f" - Durée (s)      : {res['duree_sec']}\n")
            f.write("-"*40 + "\n")

if __name__ == "__main__":
    modeles = [
        ("Abstraction/src/models/car_evaluation.json", "Modèle car_evaluation (encodé)"),
        ("Abstraction/src/models/placement.json", "Modèle placement (encodé)"),
        ("Abstraction/src/models/titanic_train.json", "Modèle titanic"),
    ]

    boites_inits = [
        "Abstraction/src/datasets_train_encoded/car_evaluation.csv",
        "Abstraction/src/datasets_train_encoded/placement.csv",
        "Abstraction/src/datasets_train_encoded/titanic_train.csv",
    ]

    res = test_models_batch(modeles, boites_inits)

    print("\n====== Récapitulatif ======")
    for r in res:
        print(r)

    enregistrer_resultats(res)
    print("\nRésultats enregistrés dans 'resultats_stabilite.txt'")

