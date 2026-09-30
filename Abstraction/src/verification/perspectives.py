import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from numba.typed import List
from tqdm import tqdm
from src.verification.boite import Boite 
from src.verification.build_boite import  BoitePropagator
from math import ceil
from collections import defaultdict 
import numpy as np
from src.verification.utils import filter_dominated,is_not_in,filter_non_dominated

def is_weakly_stable_on_F(self):
        boxes_inter_class = defaultdict(list)
        all_features = list(Boite.f_min(next(iter(self.boxes_by_class.values()))[0]).keys())
        F = []
        remaining = all_features.copy()
        broken = None

        print("Features totales :", all_features)
        
        while remaining:
            candidate = remaining.pop(0)
            current_F = F + [candidate]
            
            stable_for_candidate = True
            nb_boite = 0
            nb_tested =0

            for cls, boxes in self.boxes_by_class.items():
                min_boxes, max_boxes = self.extract_minmax_boxes(boxes)
                boites = self.generate_inter_boxe_ameliorer(min_boxes, max_boxes)
                for b in boites:
                    boxes_inter_class[cls].append(b)

                for boite in boites:
                    fmin, fmax = Boite.f_min(boite), Boite.f_max(boite)
                    sous_boites = self.propagate.propagate_boite(boite)

                    for sb in sous_boites:
                        sb_fmin, sb_fmax = Boite.f_min(sb["boite"]), Boite.f_max(sb["boite"])
                        
                        # on vérifie la stabilité faible sur les autres features
                        other_features = [f for f in all_features if f not in current_F]

                        if is_not_in(other_features, fmin, sb_fmin) and is_not_in(other_features, fmin, fmax):
                            nb_tested +=1
                            # Si la prédiction change, ce n'est pas stable
                            if sb["prediction"] != cls:
                                stable_for_candidate = False
                                nb_boite += 1
                                b_broken = sb["boite"]
                                broken=(boite,b_broken)

                                break
                    if not stable_for_candidate:
                        break
                if not stable_for_candidate:
                    break

            if stable_for_candidate:
                # On ajoute le candidat dans le groupe F
                F.append(candidate)
                print(f"✅ Feature ajoutée au groupe faible : {candidate}")
            else:
                print(f"❌ Feature rejetée (pas stable partout) : {candidate}")
                print(broken)

        print("\nRésultat final :")
        print("Features totales :", all_features)
        print("Features validées (stabilité faible) :", F)
        print("Nombre de boîtes instables détectées :", nb_boite)
        print("Nombre de boîtes tester :", nb_tested)

        return F,boxes_inter_class

def point_in_box(point, fmin, fmax):
    return all(fmin[i] <= point[i] <= fmax[i] for i in range(len(point)))

def boite_est_vide(boite, X, features):
    fmin = np.array([Boite.f_min(boite)[f] for f in features])
    fmax = np.array([Boite.f_max(boite)[f] for f in features])
    if X != None:
        return not any(point_in_box(x, fmin, fmax) for x in X)
    return True