from numba import njit,types
import numpy as np
from collections import defaultdict 
from numba.typed import List
from src.verification.boite import Boite
from collections import defaultdict
import pandas as pd
import re

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
def eq_numba(features,i1, i2):
    for f in features:
        if i1[f] != i2[f]:
            return False
    return True

def eq(features,i1, i2):
    for f in features:
        if i1[f] != i2[f]:
            return False
    return True

def is_not_in(features,fmax,fmin):
    if not features:
        return True
    v1 = np.array([fmax[f] for f in features])
    v2 = np.array([fmin[f] for f in features])
    return eq_numba(features,v1,v2)

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

def leq(features,i1, i2):
    for f in features:
        if i1[f] >= i2[f]:
            return False
    return True

def is_weak_candidate(features,all_features,fmax,fmin):
    other = [f for f in all_features if f  not in features ]
    return (leq(features,fmax,fmin) and eq_numba(other,fmin,fmax))

def filter_wweak_dominated(features,all_features,instances):
    other_features = [f for f in all_features if f  not in features ]
    result = List.empty_list(types.float32[:])

    for i in range(len(instances)):
        dominated = False
        for j in range(len(result)):
            if leq(features,instances[i],result[j]) and eq_numba(other_features,instances[i],result[j]):
                dominated = True
                break

        if not dominated:
            # On va supprimer les dominés de result
            keep = List.empty_list(types.float32[:])
            for r in result:
                if not (leq(features,r,instances[i]) and eq_numba(other_features,r,instances[i])):
                    keep.append(r)
            keep.append(instances[i])
            result.clear()
            for r in keep:
                result.append(r)

    return result

def filter_weak_non_dominated(features,all_features,instances):
    other_features = [f for f in all_features if f  not in features ]
    result = List.empty_list(types.float32[:])

    for i in range(len(instances)):
        dominated = False
        for j in range(len(result)):
            if leq(features,result[j], instances[i]) and eq_numba(other_features,instances[i],result[j]):
                dominated = True
                break

        if not dominated:
            # On va supprimer les dominés de result
            keep = List.empty_list(types.float32[:])
            for r in result:
                if not (leq(features,instances[i],r) and eq_numba(other_features,r,instances[i])):
                    keep.append(r)
            keep.append(instances[i])
            result.clear()
            for r in keep:
                result.append(r)

    return result



# --------- UTILS NUMBA ---------

@njit
def leqF_numba(a, b, F_idx, notF_idx):
    for j in F_idx:
        if a[j] > b[j]:
            return False
    for j in notF_idx:
        if a[j] != b[j]:
            return False
    return True

@njit
def filter_non_dominated_F(instances, F_idx, notF_idx):
    result = List.empty_list(types.float32[:])

    for i in range(len(instances)):
        dominated = False
        for j in range(len(result)):
            if leqF_numba(result[j], instances[i], F_idx, notF_idx):
                dominated = True
                break

        if not dominated:
            keep = List.empty_list(types.float32[:])
            for r in result:
                if not leqF_numba(instances[i], r, F_idx, notF_idx):
                    keep.append(r)
            keep.append(instances[i])
            result.clear()
            for r in keep:
                result.append(r)

    return result

@njit
def filter_dominated_F(instances, F_idx, notF_idx):
    result = List.empty_list(types.float32[:])

    for i in range(len(instances)):
        dominated = False
        for j in range(len(result)):
            if leqF_numba(instances[i], result[j], F_idx, notF_idx):
                dominated = True
                break

        if not dominated:
            keep = List.empty_list(types.float32[:])
            for r in result:
                if not leqF_numba(r, instances[i], F_idx, notF_idx):
                    keep.append(r)
            keep.append(instances[i])
            result.clear()
            for r in keep:
                result.append(r)

    return result

# --------- ADJUSTED EXTRACTION ---------


def extract_minmaxF_boxes(boxes, features, F):
    # Convertir les fmin et fmax en tableaux NumPy
    fmin= [Boite.f_min(b) for b in boxes]
    fmax = [Boite.f_max(b) for b in boxes]

    # Convertir en tableaux NumPy
    fmins = np.array([Boite.to_array(f, features) for f in fmin])
    fmaxs = np.array([Boite.to_array(f, features) for f in fmax])

    # Obtenir les indices F et non-F
    F_idx = np.array([features.index(f) for f in F], dtype=np.int32)
    notF_idx = np.array([i for i in range(len(features)) if i not in F_idx], dtype=np.int32)

    adjusted_candidates = []

    for i in range(len(fmins)):
        dominated = False
        for j in range(len(fmins)):
            if i == j:
                continue
            if leqF_numba(fmins[j], fmins[i], F_idx, notF_idx):
                # Vérifier la domination sur les fmax (nonF) maintenant
                if np.all(fmaxs[j][notF_idx] >= fmaxs[i][notF_idx]):
                    # Boîte j couvre plus que boîte i sur notF → on élimine i
                    dominated = True
                    break
                else:
                    # On ajuste le point dominé
                    # point = np.copy(fmins[i])
                    # for idx in notF_idx:
                    #     point[idx] = fmaxs[j][idx]
                    adjusted_candidates.append(fmaxs[j])
        if not dominated:
            adjusted_candidates.append(fmins[i])

    # Convertir en tableau numpy
    adjusted_candidates = np.array(adjusted_candidates, dtype=np.float32)

    # Filtrage final avec le ≤F
    minF_np = filter_non_dominated_F(adjusted_candidates, F_idx, notF_idx)
    maxF_np = filter_dominated_F(fmaxs, F_idx, notF_idx)

    # (optionnel) convertir de nouveau en dictionnaires
    def array_to_box(arr):
        return {f: float(val) for f, val in zip(features, arr)}


    min_boxes = [array_to_box(b) for b in minF_np]
    max_boxes = [array_to_box(b) for b in maxF_np]

    return min_boxes, max_boxes




def detect_onehot_groups_from_dataset(csv_path):
    """
    Détecte automatiquement les groupes one-hot à partir du nom original
    des colonnes, et retourne les colonnes encodées (f0, f1, ...) regroupées.
    """
    df = pd.read_csv(csv_path, nrows=1)
    groups = defaultdict(list)

    # Corrigé : extraire juste le groupe "gr1", "gr2", etc.
    pattern = re.compile(r"^(gr\d)")

    for idx, col in enumerate(df.columns):
        match = pattern.match(col)
        if match:
            group_prefix = match.group(1)   # ex: 'gr1'
            feature_name = f"f{idx}"        # ex: 'f6'
            groups[group_prefix].append(feature_name)

    return list(groups.values())
