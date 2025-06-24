import lightgbm as lgb
import matplotlib.pyplot as plt
import sys
# Charger le modèle
booster = lgb.Booster(model_file=sys.argv[1])  # ou modèle entraîné
# Choisir l’arbre à afficher (par ex. le 0e arbre)
ax = lgb.plot_tree(booster, tree_index=0, figsize=(20, 10), show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
plt.show()



