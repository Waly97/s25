from numba.typed import List
from tqdm import tqdm
from boite import Boite 
from itertools import combinations
from build_boite import  BoitePropagator
from math import ceil
from collections import defaultdict 
from numba import njit,types
import numpy as np
from check_cexemple import extract_instance_from_boite,predict_from_boites

def leq_reverse(a, b, leq_fn):
    return leq_fn(b, a)

@njit
def is_dominated(candidate, current_boxes, leq_fn):
    for other in current_boxes:
        if leq_fn(other, candidate):
            return True
    return False


@njit
def leq_numba(i1, i2):
    for i in range(len(i1)):
        if i1[i] > i2[i]:
            return False
    return True

@njit
def filter_non_dominated(instances):
    result = List.empty_list(types.float32[:])

    for i in range(len(instances)):
        dominated = False
        for j in range(len(result)):
            if leq_numba(result[j], instances[i]):
                dominated = True
                break

        if not dominated:
            # On va supprimer les dominés de result
            keep = List.empty_list(types.float32[:])
            for r in result:
                if not leq_numba(instances[i], r):
                    keep.append(r)
            keep.append(instances[i])
            result.clear()
            for r in keep:
                result.append(r)

    return result

@njit
def filter_dominated(instances):
    result = List.empty_list(types.float32[:])

    for i in range(len(instances)):
        dominated = False
        for j in range(len(result)):
            if leq_numba( instances[i],result[j]):
                dominated = True
                break

        if not dominated:
            # On va supprimer les dominés de result
            keep = List.empty_list(types.float32[:])
            for r in result:
                if not leq_numba(r,instances[i]):
                    keep.append(r)
            keep.append(instances[i])
            result.clear()
            for r in keep:
                result.append(r)

    return result


class StabilityChecker:

    def __init__(self,boxes_by_class, propagate : BoitePropagator,model):
        self.boxes_by_class = boxes_by_class
        self.propagate=propagate
        self.model = model

    def leq(self,i1,i2):
        return all(i1[f]<=i2[f] for f in i1)
    
    def leq_strict(self,i1,i2):
        return all(float(i1[f])< float(i2[f]) for f in i1)
  
    
    def is_valid(self,boxe1,boxe2):
        b1= Boite.to_interval_instance(boxe1)
        b2=Boite.to_interval_instance(boxe2)
        if (self.leq(b1["fmax"],b2["fmin"])):
            return Boite.from_bounds(b1["fmax"],b2["fmin"])
        if (self.leq(b2["fmax"],b1["fmin"])):
            return Boite.from_bounds(b2["fmax"],b1["fmin"])
        return None
        
    def generate_inter_boxes_class(self,boxes):
        inter_boxes =[]

        for b1,b2 in combinations(boxes,2):
            inter = self.is_valid(b1,b2)
            if inter is not None:
                inter_boxes.append(inter)
        return inter_boxes

    
    def test_validation(self,boxes,inter_boxes):
        fmins = [Boite.f_min(b) for b in boxes]
        fmaxs = [Boite.f_max(b) for b in boxes]
        fminInter = [Boite.f_min(b) for b in inter_boxes]
        fmaxInter = [Boite.f_max(b) for b in inter_boxes]

        for b in fminInter:
            if not self.is_minimal(b,fmins):
                print("min boxes broken", b)
                print("collecting min failled")
                return False
        for b in fmaxInter:
            if not self.is_maximal(b,fmaxs):
                print("collecting max failled")
                return False
        return True
            
    def _is_stable_intra_class(self, class_id, boxes):
        min_boxes, max_boxes = self.extract_minmax_boxes(boxes)
        inter_boxes = self.generate_inter_boxe_ameliorer(min_boxes, max_boxes)

        if not inter_boxes:
            return True
        
        # Test de validation de l'extraction des mins maxs
        if not self.test_validation(boxes,inter_boxes):
            return False
        
        i=1
        for b in inter_boxes:
            tqdm.write(f"🔁 Classe {class_id} — boite {i} / {len(inter_boxes)}")
            result = self.propagate.propagate_boite(b)
            i+=1
            # Regroupe les boîtes par classe prédite
            regroupement = BoitePropagator.regrouper_boites_par_classe(result)

            # Si une seule classe est présente et correspond à class_id → stable
            if len(regroupement) == 1 and class_id in regroupement:
                continue
            c_exemple = 0
            for cls in regroupement:
                if cls != class_id :
                    print (len(regroupement[cls]))
                    c_exemple = cls
                    break
            print(f"⚠️ Stabilité rompue pour la classe {class_id}. Classes rencontrées : {list(regroupement.keys())}")
            print("boxe for broken", regroupement[c_exemple][1])
            print("compare boxe",b)

            boxe_in = regroupement[c_exemple][0]
            print("boxe in",boxe_in)
            result =predict_from_boites(self.model,boxe_in,b)
            print(result)
            return False,[]
        return  True,inter_boxes

    

    def _verif_stable_intra_class(self):
        boxes_inter_by_classe= defaultdict(list)
        for class_id,boxes in self.boxes_by_class.items():
            is_stable,inter_boxes= self._is_stable_intra_class(class_id,boxes)
            if is_stable:
                for b in inter_boxes:
                    boxes_inter_by_classe[class_id].append(b)
                continue
            print("stability has broken")
            return False,None
        print("stability is respected")
        return True,boxes_inter_by_classe
    
    def is_minimal(self, instance, boxe):
        for other in boxe:
            if self.leq_strict(other, instance) and other != instance:
                print("It's this ", other)
                return False
        return True

    
    def is_maximal(self,instance,boxe):
        return not any(self.leq_strict(instance,other) and other != instance for other in boxe)

    def extract_minmax_boxe(self, boxes):
        print(f"📦 Début de l'extraction des boîtes min/max (total : {len(boxes)})")

        fmins = [Boite.f_min(b) for b in boxes]
        fmaxs = [Boite.f_max(b) for b in boxes]

        i_min = fmins.pop()
        i_max = fmaxs.pop()

        min_boxes = [i_min]
        max_boxes = [i_max]

        print("🔍 Calcul des min_boxes")
        for b_min in tqdm(fmins, desc="⏬ min_boxes", ncols=80):
            is_dominated = False
            for j_min in min_boxes[:]:  # copie sécurisée
                if self.leq(j_min, b_min):
                    is_dominated = True
                    break
                elif self.leq(b_min, j_min):
                    min_boxes.remove(j_min)
            if not is_dominated:
                min_boxes.append(b_min)

        print("🔍 Calcul des max_boxes")
        for b_max in tqdm(fmaxs, desc="⏫ max_boxes", ncols=80):
            is_dominated = False
            for j_max in max_boxes[:]:
                if self.leq(b_max, j_max):
                    is_dominated = True
                    break
                elif self.leq(j_max, b_max):
                    max_boxes.remove(j_max)
            if not is_dominated:
                max_boxes.append(b_max)

        print(f"✅ Extraction terminée : {len(min_boxes)} min, {len(max_boxes)} max")
        return min_boxes, max_boxes

    def extract_minmax_boxes(self, boxes):
        print(f"🚀 [Numba] Extraction de {len(boxes)} boîtes")

        # Ordre cohérent des features
        features = list(Boite.f_min(boxes[0]).keys())
        fmins = [Boite.f_min(b) for b in boxes]
        fmaxs = [Boite.f_max(b) for b in boxes]

        # Convertir en tableaux NumPy
        fmins_array = np.array([Boite.to_array(f, features) for f in fmins])
        fmaxs_array = np.array([Boite.to_array(f, features) for f in fmaxs])

        # Appliquer le filtrage
        print("⏬ Calcul des min_boxes...")
        min_boxes_np = filter_non_dominated(fmins_array)

        print("⏫ Calcul des max_boxes...")
        max_boxes_np = filter_dominated(fmaxs_array)

        # (optionnel) convertir de nouveau en dictionnaires
        def array_to_box(arr):
            return {f: float(val) for f, val in zip(features, arr)}

        min_boxes = [array_to_box(b) for b in min_boxes_np]
        max_boxes = [array_to_box(b) for b in max_boxes_np]

        print(f"✅ Terminé : {len(min_boxes)} min | {len(max_boxes)} max")
        return min_boxes, max_boxes



    def build_max_boxes(self, fmin, max_boxes):
        inter_boxes = []
        candidates = [fmax for fmax in max_boxes if self.leq(fmin, fmax)]
        for fmax in candidates:
            inter_box = Boite.from_bounds(fmin, fmax)
            inter_boxes.append(inter_box)
        return inter_boxes

    def build_max_boxes_list(self, min_boxes, max_boxes):
        inter_boxes = self.build_max_boxes(min_boxes[0], max_boxes)
        for fmin in min_boxes[1:]:
            inter_boxes.extend(self.build_max_boxes(fmin, max_boxes))
        return inter_boxes

    def generate_inter_boxe_ameliorer(self, min_boxes, max_boxes, batch_size=100):
        result = []
        nb_boxes = 0
        num_batches = ceil(len(min_boxes) / batch_size)

        batches = [
            min_boxes[i * batch_size:(i + 1) * batch_size]
            for i in range(num_batches)
        ]

        pbar = tqdm(batches, desc="🔄 Génération optimisée", ncols=80)
        for batch in pbar:
            boxes = self.build_max_boxes_list(batch, max_boxes)
            result.extend(boxes)
            nb_boxes += len(boxes)

        print(f"✅ Nombre total de boîtes générées : {nb_boxes}")
        return result


    def is_stable(self,box1, box2):
        fmin1, fmax1 = box1["fmin"],box1["fmax"]
    
        fmin2 ,fmax2=  box2["fmin"],box2["fmax"]

        if self.leq(fmin1,fmin2):
            if self.leq(fmin2,fmax1) or self.leq(fmax2,fmax1):
                return False
        return True

    def is_stable_list(self,boxes1,boxes2):
        total= len(boxes1) * len(boxes2)
        pbar = tqdm(total=total, desc="statability boxes list",ncols=80)
        for b in boxes1:
            i1= Boite.to_interval_instance(b)
            for b1 in boxes2:
                i2 = Boite.to_interval_instance(b1)
                if (not self.is_stable(i1,i2)) or (not self.is_stable(i2,i1)):
                    pbar.update(1)
                    print("instance1",i1)
                    print("instance2", i2)
                    pbar.close()
                    return False
                pbar.update(1)
        pbar.close()
        return True

    def is_stable_boxes_by_class(self,boxes_by_class):
        keys = list(boxes_by_class.keys())  # pour garder l’ordre
        total = len(keys) * (len(keys) - 1) // 2
        pbar = tqdm(total=total, desc="Checking stability", ncols=80)

        # On fait une copie pour pouvoir supprimer
        boxes_copy = boxes_by_class.copy()

        while len(boxes_copy) > 1:
            c, bxs1 = next(iter(boxes_copy.items()))
            boxes_copy.pop(c)

            for c2, bxs2 in boxes_copy.items():
                if not self.is_stable_list(bxs1, bxs2):
                    pbar.update(1)
                    pbar.close()
                    return False
                pbar.update(1)

        pbar.close()
        return True
    

    def verif_stable(self):
        # if self.is_stable_parallele():
        is_stable,boxes = self._verif_stable_intra_class()
        if is_stable :
            print("The model is stable")
            return True,boxes
        else:
            print("The model isn't stable")
            return False,boxes
