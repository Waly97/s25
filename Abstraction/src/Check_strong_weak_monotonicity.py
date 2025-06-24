import numpy as np
from collections import defaultdict
from stable import StabilityChecker
from boite import Boite
from numba import njit,types
import itertools
from utils import is_weak_candidate



    

class MonotonicityChecker:
    def __init__(self, boxes, propagate, model):
        self.boxes = boxes
        self.propagate = propagate
        self.model = model
        self.stable = StabilityChecker(boxes, self.propagate, model)

    def check_monotony_for_order(self, boxes_inter_class):
        for i in range(len(boxes_inter_class) - 2):
            boxes_inter_c1 = boxes_inter_class[i]
            boxes_inter_c2 = boxes_inter_class[i+1]

            if not boxes_inter_c1 or not boxes_inter_c2:
                continue

            features = list(Boite.f_min(boxes_inter_c1[0]).keys())
            fmins = [Boite.f_min(b) for b in boxes_inter_c2]
            fmaxs = [Boite.f_max(b) for b in boxes_inter_c1]

            fmins_array = np.array([Boite.to_array(f, features) for f in fmins])
            fmaxs_array = np.array([Boite.to_array(f, features) for f in fmaxs])

            for fmax in fmaxs_array:
                for fmin in fmins_array:
                    if not leq_numba(fmax, fmin):
                        return False
        return True

    def verif_monotone(self):
        is_stable, boxes_inter_class = self.stable.verif_stable()
        if not is_stable:
            print("Modèle instable — vérification de monotonie impossible.")
            return False

        if self.check_monotony_for_order(boxes_inter_class):
            print("✅ Monotonie forte respectée entre classes ordonnées.")
            return True
        else:
            print("Monotonie forte non respectée.")
            return False


    def is_group_weakly_monotone(self, boxes_inter_class, F):
        if not F:
            print(" Modèle non faiblement monotone — pas de F donné")
            return False, []

        all_features = list(Boite.f_min(next(iter(boxes_inter_class))[0]).keys())
        remaining = F.copy()
        new_feature = []
        nb_boite = 0

        while remaining:
            candidate = remaining.pop(0)
            current_F = new_feature + [candidate]
            monotone_for_candidate = True

            for i in range(len(boxes_inter_class) - 1):
                c1_boxes = boxes_inter_class[i]
                c2_boxes = boxes_inter_class[i + 1]

                for b1 in c1_boxes:
                    fmax = np.array([Boite.f_max(b1)[f] for f in all_features])
                    for b2 in c2_boxes:
                        fmin = np.array([Boite.f_min(b2)[f] for f in all_features])

                        if  is_weak_candidate(current_F, all_features,fmin,fmax):
                            monotone_for_candidate = False
                            nb_boite += 1
                            break
                    if not monotone_for_candidate:
                        break
                if not monotone_for_candidate:
                    break

            if monotone_for_candidate:
                new_feature.append(candidate)
                print(f"Feature ajoutée au groupe faible : {candidate}")
            else:
                print(f"Feature rejetée (pas monotone partout) : {candidate}")

        if not new_feature:
            return False, []

        return True, new_feature


    def detect_largest_weakly_monotone_group(self):
        F, boxes = self.stable.is_weakly_stable_on_F()

        if not F:
            print("Modèle non faiblement stable — arrêt.")
            return []

        class_keys = list(boxes.keys())

        for perm in itertools.permutations(class_keys):
            reordered = [boxes[k] for k in perm]
            valid, features = self.is_group_weakly_monotone(reordered, F)
            if valid:
                print("\n Modèle faiblement monotone pour les features :", features)
                print("Ordre des classes utilisé :", perm)
                return features

        print("\nAucun ordre ne rend le modèle faiblement monotone.")
        return []


            
    


    def compare_strong_vs_weak_monotone_groups(self):
        strong = self.detect_largest_group_monotone_features()
        weak = self.detect_largest_weakly_monotone_group()

        print("\n🔍 Comparaison des groupes :")
        print(f"➡️ Forte  ({len(strong)} features) : {strong}")
        print(f"➡️ Faible ({len(weak)} features) : {weak}")

        extra_in_weak = [f for f in weak if f not in strong]
        if extra_in_weak:
            print(f"Features uniquement dans le groupe faible : {extra_in_weak}")
        else:
            print("Les deux groupes sont identiques.")
