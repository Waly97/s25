from boite import Boite
from stable import StabilityChecker, leq_numba
import numpy as np

import itertools

class MonotonicityChecker:
    def __init__(self, boxes, model, order_classes):
        self.boxes = boxes
        self.model = model
        self.order_classes = order_classes
        self.stable = StabilityChecker(boxes, model)

    def check_monotony_for_order(self, ordered_classes, boxes_inter_class):
        """
        Vérifie la monotonie pour un ordre donné de classes.
        """

        if  boxes_inter_class is not None:
            for i in range(len(ordered_classes)):
                for j in range(i + 1, len(ordered_classes)):
                    c1 = ordered_classes[i]
                    c2 = ordered_classes[j]

                    boxes_inter_c1 = boxes_inter_class[i]
                    boxes_inter_c2 = boxes_inter_class[j]

                    if not boxes_inter_c1 or not boxes_inter_c2:
                        continue
                    f1 = list(Boite.f_min(boxes_inter_c1[0]).keys())

                    fmins = [Boite.f_min(b) for b in boxes_inter_c2]
                    fmaxs = [Boite.f_max(b) for b in boxes_inter_c1]

                    fmins_array = np.array([Boite.to_array(f, f1) for f in fmins])
                    fmaxs_array = np.array([Boite.to_array(f, f1) for f in fmaxs])

                    for fmax in fmaxs_array:
                        for fmin in fmins_array:
                            if not leq_numba(fmax, fmin):
                                return False
        return True

    def verif_monotone(self):
        is_stable, boxes_inter_class = self.stable.verif_stable()

        if not is_stable:
            print("Model is not stable, cannot check monotony.")
            return False

        classes = list(self.order_classes.keys())
        classes.sort(key=lambda c: self.order_classes[c])

        if self.check_monotony_for_order(classes, boxes_inter_class):
            print("Monotony is respected for the given order.")
            return True
        else:
            print("Given order does not respect monotony. Searching for alternative orders...")
            
            # Testing all permutations
            for perm in itertools.permutations(self.order_classes.keys()):
                if self.check_monotony_for_order(perm, boxes_inter_class):
                    print(f"Monotony is respected with new order: {perm}")
                    return True

            print("No order satisfies monotony. Model is totally non-monotonic.")
            return False



