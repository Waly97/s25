from numba.typed import List
from tqdm import tqdm
from boite import Boite 
from itertools import combinations
from build_boite import  BoitePropagator
from math import ceil
from collections import defaultdict 
from numba import njit,types
import numpy as np
from utils import leq




def xM_from_fmin_fmax(fmin, fmax, F):
    return {f: fmin[f] if f in F else fmax[f] for f in fmin}

def is_in_boxe(fmin1,fmax1,fmin2,fmax2):
    return leq(fmin1,fmin2) and leq(fmax1,fmax2)

class StabilityChecker:

    def __init__(self,boxes_by_class, propagate : BoitePropagator,model):
        self.boxes_by_class = boxes_by_class
        self.propagate=propagate
        self.model = model
        self.broken = defaultdict(list)
        self.contre_exemple = defaultdict(list)
        self.taux_stability= 0

    
    def sweep_planel(boxes, F):
        def merge_dicts(base, override, keys):
            return {f: override[f] if f in keys else base[f] for f in base}

        events = []
        for fmin, fmax in boxes:
            events.append((tuple(fmin[f] for f in fmin if f not in F), 'start', fmin, fmax))
            events.append((tuple(fmax[f] for f in fmax if f not in F), 'end', fmin, fmax))
        events.sort()

        result = []
        active = []
        last_box = None
        current_fmin = None
        current_fmax = None

        for key, typ, fmin, fmax in events:
            if typ == 'start':
                if not active:
                    current_fmin, current_fmax = fmin, fmax
                else:
                    if last_box is None:
                        new_fmin = merge_dicts(current_fmin, current_fmin, F)
                        new_fmax = merge_dicts(current_fmin, fmin, F)
                        result.append((new_fmin, new_fmax))
                        last_box = (new_fmax, current_fmax)
                        current_fmin, current_fmax = fmin, fmax
                    else:
                        new_fmax = merge_dicts(current_fmin, last_box[1], F)
                        result.append((current_fmin, new_fmax))
                        current_fmin = merge_dicts(current_fmin, fmin, F)
                        fmin_last = merge_dicts(last_box[0], current_fmin, F)
                        last_box = (fmin_last, last_box[1])
                active.append((fmin, fmax))

            elif typ == 'end':
                if last_box:
                    if fmax == last_box[1]:
                        current_fmin = merge_dicts(current_fmin, fmax, F)
                        
                        if leq(F, current_fmax, last_box[1]):
                            active.remove((fmin, fmax))
                            result.append((current_fmin, fmax))
                            last_box = None
                        else:
                            new_fmax = merge_dicts(current_fmax, fmax, F)
                            result.append((last_box[0], new_fmax))
                            last_box = None
                    elif fmax == current_fmax:
                        current_fmin = merge_dicts(fmin, current_fmax, F)
                        current_fmax = fmax
                        if leq(F, current_fmax, last_box[0]):
                            new_fmax = merge_dicts(fmax, current_fmax, F)
                            result.append((current_fmin, new_fmax))  
                            last_box = None 
                        else:
                            result.append((last_box[0], current_fmax))
                            last_box = None
                    else:
                        current_fmin = merge_dicts(current_fmin, fmax, F)
                        last_box = (merge_dicts(last_box[0], fmax, F), last_box[1])
                        if leq(F, current_fmax, fmin) and leq(F, last_box[1], fmin):
                            chosen_fmin = current_fmin if leq(F, current_fmax, last_box[0]) else last_box[0]
                            result.append((chosen_fmin, fmax))
                        elif leq(F, fmax, current_fmin) and leq(F, current_fmax, last_box[0]):
                            new_fmax = merge_dicts(last_box[1], fmax, F)
                            result.append((fmin, new_fmax))
                        elif leq(F, fmax, current_fmin) and leq(F, last_box[1], current_fmin):
                            new_fmax = merge_dicts(current_fmax, fmax, F)
                            result.append((fmin, new_fmax))
                else:
                    result.append((fmin, fmax))
                    active = []

        if last_box:
            result.append((last_box[0], last_box[1]))
        return result




