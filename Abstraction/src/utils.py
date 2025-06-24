from numba import njit,types
import numpy as np
from collections import defaultdict 
from numba.typed import List

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
        if i1[f] > i2[f]:
            return False
    return True

def is_weak_candidate(features,all_features,fmax,fmin):
    other = [f for f in all_features if f  not in features ]
    return (leq(features,fmax,fmin) and eq_numba(other,fmin,fmax))