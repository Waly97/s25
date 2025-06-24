import xgboost as xgb
import matplotlib.pyplot as plt
import sys

# # Charger un modèle sauvegardé au format .json ou .bin
# model = xgb.Booster()
# model.load_model(sys.argv[1])  # <-- adapte le chemin ici

# # Afficher le premier arbre (index 0)
# xgb.plot_tree(model, num_trees=2)
# # plt.rcParams['figure.figsize'] = [15, 10]
# plt.show()


from boite import Boite

from itertools import product

def generer_boites_init_avec_onehot(boite_init, groupes_onehot):
    """
    Génère toutes les boîtes initiales valides à partir d'une boîte initiale
    en respectant les contraintes one-hot et l'ordre des features.
    """
    # Crée les "états valides" de chaque groupe
    groupes_valeurs = []
    for groupe in groupes_onehot:
        valeurs = []
        for f_actif in groupe:
            valeurs.append(f_actif)
        groupes_valeurs.append(valeurs)

    # Autres features
    autres_features = [
        f for f in boite_init.bornes.keys()
        if all(f not in groupe for groupe in groupes_onehot)
    ]

    # Génération des boîtes
    boites_valides = []
    for combinaison in product(*groupes_valeurs):
        # Nouveau dictionnaire bornes avec ordre correct
        nouvelles_bornes = {}
        for f in boite_init.bornes.keys():
            # Si feature est dans un groupe one-hot, on active l'élément choisi
            for i, groupe in enumerate(groupes_onehot):
                if f in groupe:
                    if f == combinaison[i]:
                        nouvelles_bornes[f] = [1, 1]
                    else:
                        nouvelles_bornes[f] = [0, 0]
                    break
            else:
                # Sinon, borne inchangée
                nouvelles_bornes[f] = boite_init.bornes[f]
        boites_valides.append(Boite(nouvelles_bornes))

    return boites_valides
boite_init = Boite({
    "f0": [40.89, 89.4],
    "f1": [37.0, 97.7],
    "f2": [50.0, 91.0],
    "f3": [50.0, 98.0],
    "f4": [51.21, 77.89],
    "f5": [200000.0, 940000.0],
    "f6": [0.0, 1.0],
    "f7": [0.0, 1.0],
    "f8": [0.0, 1.0],
    "f9": [0.0, 1.0]
})

groupes_onehot = [["f6", "f7"], ["f8", "f9"]]

boites = generer_boites_init_avec_onehot(boite_init, groupes_onehot)

print(f"Nombre de boîtes générées : {len(boites)}")
for i, b in enumerate(boites, 1):
    print(f"Boîte {i}: {b}")