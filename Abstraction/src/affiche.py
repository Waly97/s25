import xgboost as xgb
import matplotlib.pyplot as plt
import sys

# Charger un modèle sauvegardé au format .json ou .bin
model = xgb.Booster()
model.load_model(sys.argv[1])  # <-- adapte le chemin ici

# Afficher le premier arbre (index 0)
xgb.plot_tree(model, num_trees=6)
plt.rcParams['figure.figsize'] = [15, 10]
plt.show()
