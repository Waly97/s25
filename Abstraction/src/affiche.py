import sys
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_tree
from stable import StabilityChecker
from boite import Boite



# model_path = sys.argv[1]
# model = XGBClassifier()
# model.load_model(model_path)
# xgb.plot_tree(model, tree_idx=2)
# plt.show()

def leq(i1,i2):
    return all(i1[f]<=i2[f] for f in i1)

def is_minimal(instance,boxe):
    return not any(leq(other,instance) and other != instance for other in boxe)
    
def is_maximal(instance,boxe):
    return not any(leq(instance,other) and other != instance for other in boxe)
    
def extract_minmax_boxes(boxes):
    fmins = [Boite.f_min(b) for b in boxes]
    fmaxs =[Boite.f_max(b) for b in boxes]

    min_boxes = [b for b in boxes if is_minimal(Boite.f_min(b),fmins)]
    max_boxes =[b for b in boxes if is_maximal(Boite.f_max(b),fmaxs)]
    return min_boxes,max_boxes

def tracer_toutes_zones_2D(boites_par_classe, features=(0, 1)):
    couleurs = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'brown']

    plt.figure(figsize=(8, 6))

    for classe, boites in boites_par_classe.items():
        # Extraire uniquement les boîtes maximales
        _, max_boxes =extract_minmax_boxes(boites)

        fmins_0 = []
        fmaxs_0 = []
        fmins_1 = []
        fmaxs_1 = []

        for b in max_boxes:
            bornes = b.bornes
            if features[0] in bornes and features[1] in bornes:
                fmins_0.append(bornes[features[0]][0])
                fmaxs_0.append(bornes[features[0]][1])
                fmins_1.append(bornes[features[1]][0])
                fmaxs_1.append(bornes[features[1]][1])

        if fmins_0 and fmins_1:
            x_min = min(fmins_0)
            x_max = max(fmaxs_0)
            y_min = min(fmins_1)
            y_max = max(fmaxs_1)

            couleur = couleurs[classe % len(couleurs)]
            plt.gca().add_patch(
                plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              fill=True, alpha=0.3, edgecolor=couleur, facecolor=couleur,
                              label=f"Classe {classe}")
            )

    plt.xlabel(f"Feature {features[0]}")
    plt.ylabel(f"Feature {features[1]}")
    plt.title("Zone englobante des boîtes maximales par classe")
    plt.legend()
    plt.grid(True)
    plt.show()


