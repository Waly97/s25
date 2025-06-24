import numpy as np
from collections import defaultdict
from stable import StabilityChecker, leq_numba,eq_numba
from boite import Boite
import itertools

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
            print("❌ Modèle instable — vérification de monotonie impossible.")
            return False

        if self.check_monotony_for_order(boxes_inter_class):
            print("✅ Monotonie forte respectée entre classes ordonnées.")
            return True
        else:
            print("❌ Monotonie forte non respectée.")
            return False

    def is_group_monotone(self, boxes_inter_class, features):
        for i in range(len(boxes_inter_class) - 2):
            c1_boxes = boxes_inter_class[i]
            c2_boxes = boxes_inter_class[i + 1]

            for b1 in c1_boxes:
                for b2 in c2_boxes:
                    v1 = np.array([Boite.f_max(b1)[f] for f in features])
                    v2 = np.array([Boite.f_min(b2)[f] for f in features])
                    if not leq_numba(v1, v2):
                        return False
        return True

    def detect_largest_group_monotone_features(self):
        is_stable, boxes_inter_class = self.stable.verif_stable()

        if is_stable:
            all_features = list(Boite.f_min(next(iter(boxes_inter_class.values()))[0]).keys())
            F = []
            best_order = None

            class_keys = list(boxes_inter_class.keys())

            for f in all_features:
                candidate = F + [f]
                valid = False

                for perm in itertools.permutations(class_keys):
                    reordered = [boxes_inter_class[k] for k in perm]
                    if self.is_group_monotone(reordered, candidate):
                        valid = True
                        best_order_candidate = perm
                        break

                if valid:
                    F.append(f)
                    best_order = best_order_candidate  # Mémorise la permutation valide trouvée

            if F:
                # if not is_stable:
                #     self.stable.is_weakly_stable_on_F(boxes_inter_class,F)
                print(f"✅ Groupe de monotonie forte détecté : {F}")
                print(f"🔢 Ordre des classes respectant la monotonie : {best_order}")
                return F, best_order

        print("❌ Aucun groupe de monotonie détecté.")
        return [], None



    def is_group_weakly_monotone(self, boxes_inter_class, features):
        for i in range(len(boxes_inter_class) - 2):
            c1_boxes = boxes_inter_class[i]
            c2_boxes = boxes_inter_class[i + 1]

            for b1 in c1_boxes:
                for b2 in c2_boxes:
                    v1 = np.array([Boite.f_max(b1)[f] for f in features])
                    v2 = np.array([Boite.f_min(b2)[f] for f in features])
                    v3 = np.array([Boite.f_max(b1)[f] for f in features if f not in features])
                    v4 = np.array([Boite.f_min(b2)[f] for f in features if f not in features])

                    if not leq_numba(v1, v2):
                        return False
                    if not eq_numba(v3,v4):
                        return False            
        return True


    def detect_largest_weakly_monotone_group(self):
        is_stable, boxes_inter_class = self.stable.verif_stable()
        # if not is_stable:
        #     print("⚠️ Modèle instable — impossible de détecter une monotonie faible.")
        #     return []

        all_features = list(Boite.f_min(boxes_inter_class[0][0]).keys())
        F = []
        best_order = None

        class_keys = list(boxes_inter_class.keys())

        for f in all_features:
            candidate = F + [f]
            valid = False

            for perm in itertools.permutations(class_keys):
                reordered = [boxes_inter_class[i] for i in perm]
                if self.is_group_weakly_monotone(reordered, candidate):
                    valid = True
                    best_order_candidate = perm
                    break
            if valid:
                F.append(f)
                best_order = best_order_candidate
        # if not is_stable:
        #     self.stable.is_weakly_stable_on_F(boxes_inter_class,F)
        print(f"✅ Groupe de monotonie faible détecté : {F}")
        return F,best_order_candidate 
    


    def compare_strong_vs_weak_monotone_groups(self):
        strong = self.detect_largest_group_monotone_features()
        weak = self.detect_largest_weakly_monotone_group()

        print("\n🔍 Comparaison des groupes :")
        print(f"➡️ Forte  ({len(strong)} features) : {strong}")
        print(f"➡️ Faible ({len(weak)} features) : {weak}")

        extra_in_weak = [f for f in weak if f not in strong]
        if extra_in_weak:
            print(f"🔸 Features uniquement dans le groupe faible : {extra_in_weak}")
        else:
            print("✅ Les deux groupes sont identiques.")
