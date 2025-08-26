from boite import Boite
from src.verification.stable_improve import StabilityChecker
import numpy as np
from collections import defaultdict
from src.verification.utils import leq_numba
import networkx as nx

class MonotonicityChecker:
    def __init__(self, boxes, propagate, order_classes,model):
        self.boxes = boxes
        self.propagate= propagate
        self.model = model
        self.c_exemple=[]
        self.order_classes = order_classes
        self.stable = StabilityChecker(boxes, self.propagate,model)

    def check_monotony_for_order(self,boxes_inter_class):
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

   


    def check_monotony_pairwise(self,c1_boxes, c2_boxes,cls,f1):
        """Vérifie si c1 ≤ c2 pour toutes les boîtes."""
        fmins = [Boite.f_min(b) for b in c2_boxes]
        fmaxs = [Boite.f_max(b) for b in c1_boxes]

        fmins_array = np.array([Boite.to_array(f, f1) for f in fmins])
        fmaxs_array = np.array([Boite.to_array(f, f1) for f in fmaxs])

        for fmax in fmaxs_array:
            for fmin in fmins_array:
                if not leq_numba(fmax, fmin):
                    minval= [float(val) for val in fmin]
                    maxval=[float(val) for val in fmax]
                    return False,(minval,maxval)
        return True,()
    
    def build_monotony_matrix(self,classes, boxes_inter_class):
        n = len(classes)
        matrix = np.zeros((n, n), dtype=int)
        contradiction_found = False

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                boxes_i = boxes_inter_class[i]
                boxes_j = boxes_inter_class[j]

                if not boxes_i or not boxes_j:
                    continue

                f1 = list(Boite.f_min(boxes_i[0]).keys())

                c1_leq_c2,c1 = self.check_monotony_pairwise(boxes_i, boxes_j,i,f1)
                c2_leq_c1,c2 = self.check_monotony_pairwise(boxes_j, boxes_i,j, f1)

                if c1_leq_c2 and not c2_leq_c1:
                    matrix[i][j] = 1  # c1 < c2
                elif c2_leq_c1 and not c1_leq_c2:
                    matrix[j][i] = 1  # c2 < c1
                elif c2_leq_c1 and  c1_leq_c2:
                    contradiction_found = True
                    self.c_exemple.append(c1)
                    self.c_exemple.append(c2)
                    print(f"Contradiction between {classes[i]} and {classes[j]}")

        return matrix, contradiction_found

    def generate_orders_from_matrix(self,classes, matrix):
        G = nx.DiGraph()
        G.add_nodes_from(classes)

        for i, c1 in enumerate(classes):
            for j, c2 in enumerate(classes):
                if matrix[i][j]:
                    G.add_edge(c1, c2)

        if not nx.is_directed_acyclic_graph(G):
            print("Graph contains cycles. No total order possible.")
            return []

        all_orders = list(nx.all_topological_sorts(G))
        return all_orders

    def verif_monotone(self):
        is_stable,boxes_inter_class,f = self.stable.verif_stable()

        if not is_stable:
            print("Model is not stable, cannot check monotony.")
            return False

        classes = list(self.order_classes.keys())
        classes.sort(key=lambda c: self.order_classes[c])

        if self.check_monotony_for_order(boxes_inter_class):
            print("Monotony is respected for the given order.")
            return True

        print("Given order does not respect monotony. Building monotony matrix...")

        matrix, contradiction = self.build_monotony_matrix(classes, boxes_inter_class)

        if contradiction:
            print("Contradictions detected. Model is non-monotonic.")
            return False

        all_orders = self.generate_orders_from_matrix(classes, matrix)

        if not all_orders:
            print("No total order satisfies monotony.")
            return False

        print(f"Found {len(all_orders)} valid total order(s) satisfying monotony.")
        for order in all_orders:
            print("Valid order:", order)

        return True
