import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
)

from tqdm import tqdm
from math import ceil
from collections import defaultdict

import numpy as np

from src.verification.boite import Boite
from src.verification.build_boite import BoitePropagator
from src.verification.utils import (
    filter_dominated,
    filter_non_dominated,
)


class StabilityChecker:

    def __init__(
        self,
        boxes_by_class,
        propagate: BoitePropagator,
        model
    ):
        self.boxes_by_class = boxes_by_class
        self.propagate = propagate
        self.model = model

        self.broken = defaultdict(list)
        self.contre_exemple = defaultdict(list)

        # Taux moyen de stabilité sur les classes
        self.taux_stability = 0.0

        # Taux de stabilité classe par classe
        self.class_stability_rates = {}

        # True seulement s'il n'existe aucune violation
        self.formal_stable = False

        # True si chaque classe possède un taux >= 90 %
        self.stable_90 = False

    # =========================================================
    # ORDRE
    # =========================================================

    def leq(self, i1, i2):
        """
        Product order:
        i1 <= i2 iff every coordinate of i1 is <=
        the corresponding coordinate of i2.
        """
        return all(
            float(i1[f]) <= float(i2[f])
            for f in i1
        )

    def leq_strict(self, i1, i2):
        """
        Strict product order:

            i1 <= i2

        and at least one coordinate is strictly smaller.
        """
        return (
            all(
                float(i1[f]) <= float(i2[f])
                for f in i1
            )
            and
            any(
                float(i1[f]) < float(i2[f])
                for f in i1
            )
        )

    # =========================================================
    # UTILITAIRES
    # =========================================================

    def _box_key(self, box):
        """
        Canonical representation of a box.
        Used to remove duplicated boxes.
        """
        return tuple(
            (
                feature,
                float(bounds[0]),
                float(bounds[1]),
            )
            for feature, bounds in sorted(box.bornes.items())
        )

    def _remove_duplicate_boxes(self, boxes):
        """
        Remove duplicated boxes while preserving the first
        occurrence.
        """
        unique = {}

        for box in boxes:
            key = self._box_key(box)

            if key not in unique:
                unique[key] = box

        return list(unique.values())

    # =========================================================
    # VALIDATION MIN / MAX
    # =========================================================

    def is_minimal(self, instance, instances):
        """
        True iff there is no strictly smaller instance.
        """
        return not any(
            other != instance
            and self.leq_strict(other, instance)
            for other in instances
        )

    def is_maximal(self, instance, instances):
        """
        True iff there is no strictly greater instance.
        """
        return not any(
            other != instance
            and self.leq_strict(instance, other)
            for other in instances
        )

    def test_validation(self, boxes, inter_boxes):

        fmins = [
            Boite.f_min(b)
            for b in boxes
        ]

        fmaxs = [
            Boite.f_max(b)
            for b in boxes
        ]

        fmin_inter = [
            Boite.f_min(b)
            for b in inter_boxes
        ]

        fmax_inter = [
            Boite.f_max(b)
            for b in inter_boxes
        ]

        for b in fmin_inter:

            if not self.is_minimal(b, fmins):

                print(
                    "Min boxes broken:",
                    b
                )

                print(
                    "Collecting min failed"
                )

                return False

        for b in fmax_inter:

            if not self.is_maximal(b, fmaxs):

                print(
                    "Max boxes broken:",
                    b
                )

                print(
                    "Collecting max failed"
                )

                return False

        return True

    # =========================================================
    # EXTRACTION DES MINIMAUX / MAXIMAUX
    # =========================================================

    def extract_minmax_boxes(self, boxes):

        print(
            f"🚀 [Numba] Extraction de {len(boxes)} boîtes"
        )

        features = list(
            Boite.f_min(boxes[0]).keys()
        )

        fmins = [
            Boite.f_min(b)
            for b in boxes
        ]

        fmaxs = [
            Boite.f_max(b)
            for b in boxes
        ]

        # Conversion NumPy
        fmins_array = np.array(
            [
                Boite.to_array(f, features)
                for f in fmins
            ]
        )

        fmaxs_array = np.array(
            [
                Boite.to_array(f, features)
                for f in fmaxs
            ]
        )

        # Points minimaux
        print(
            "⏬ Calcul des min_boxes..."
        )

        min_boxes_np = filter_non_dominated(
            fmins_array
        )

        # Points maximaux
        print(
            "⏫ Calcul des max_boxes..."
        )

        max_boxes_np = filter_dominated(
            fmaxs_array
        )

        def array_to_instance(arr):

            return {
                feature: float(value)
                for feature, value
                in zip(features, arr)
            }

        min_boxes = [
            array_to_instance(b)
            for b in min_boxes_np
        ]

        max_boxes = [
            array_to_instance(b)
            for b in max_boxes_np
        ]

        print(
            f"✅ Terminé : "
            f"{len(min_boxes)} min | "
            f"{len(max_boxes)} max"
        )

        return min_boxes, max_boxes

    # =========================================================
    # GENERATION DES BOITES [MIN, MAX]
    # =========================================================

    def build_max_boxes(
        self,
        fmin,
        max_boxes
    ):

        inter_boxes = []

        # On ne considère que les maxima comparables
        candidates = [
            fmax
            for fmax in max_boxes
            if self.leq(fmin, fmax)
        ]

        for fmax in candidates:

            inter_box = Boite.from_bounds(
                fmin,
                fmax
            )

            inter_boxes.append(
                inter_box
            )

        return inter_boxes

    def build_max_boxes_list(
        self,
        min_boxes,
        max_boxes
    ):

        inter_boxes = []

        for fmin in min_boxes:

            inter_boxes.extend(
                self.build_max_boxes(
                    fmin,
                    max_boxes
                )
            )

        return inter_boxes

    def generate_inter_boxe_ameliorer(
        self,
        min_boxes,
        max_boxes,
        batch_size=100
    ):

        result = []
        nb_boxes = 0

        if len(min_boxes) == 0:
            return result

        num_batches = ceil(
            len(min_boxes) / batch_size
        )

        batches = [
            min_boxes[
                i * batch_size:
                (i + 1) * batch_size
            ]
            for i in range(num_batches)
        ]

        pbar = tqdm(
            batches,
            desc="🔄 Génération optimisée",
            ncols=80
        )

        for batch in pbar:

            boxes = self.build_max_boxes_list(
                batch,
                max_boxes
            )

            result.extend(boxes)

            nb_boxes += len(boxes)

        # Supprimer d'éventuels doublons
        result = self._remove_duplicate_boxes(
            result
        )

        print(
            "✅ Nombre total de boîtes générées :",
            len(result)
        )

        return result

    # =========================================================
    # VERIFICATION POUR UNE CLASSE
    # =========================================================

    def _is_stable_intra_class(
        self,
        class_id,
        boxes
    ):

        # ---------------------------------------------
        # Extraction des points minimaux / maximaux
        # ---------------------------------------------

        min_boxes, max_boxes = (
            self.extract_minmax_boxes(boxes)
        )

        # ---------------------------------------------
        # Construction des boîtes [min, max]
        # ---------------------------------------------

        inter_boxes = (
            self.generate_inter_boxe_ameliorer(
                min_boxes,
                max_boxes
            )
        )

        print(
            f"\nClasse {class_id}: "
            f"{len(boxes)} final boxes | "
            f"{len(min_boxes)} min | "
            f"{len(max_boxes)} max | "
            f"{len(inter_boxes)} intermediate boxes"
        )

        # Aucun couple min/max comparable à vérifier
        if not inter_boxes:

            return (
                True,
                inter_boxes,
                defaultdict(list)
            )

        broken = defaultdict(list)

        i = 1

        # ---------------------------------------------
        # Propagation de chaque boîte intermédiaire
        # ---------------------------------------------

        for b in inter_boxes:

            tqdm.write(
                f"🔁 Classe {class_id} "
                f"— boite {i} / {len(inter_boxes)}"
            )

            result = (
                self.propagate.propagate_boite(b)
            )

            i += 1

            predictions = {
                r["prediction"]
                for r in result
            }

            print(
                f"Classe attendue = {class_id} | "
                f"prédictions trouvées = {predictions}"
            )

            for r in result:

                if r["prediction"] != class_id:

                    broken[
                        class_id
                    ].append(
                        r["boite"]
                    )

                    contre_exemple = (
                        b,
                        r
                    )

                    self.contre_exemple[
                        class_id
                    ].append(
                        contre_exemple
                    )

        # ---------------------------------------------
        # Suppression des violations dupliquées
        # ---------------------------------------------

        broken[class_id] = (
            self._remove_duplicate_boxes(
                broken[class_id]
            )
        )

        print(
            f"Nombre de boîtes violant "
            f"la stabilité pour la classe "
            f"{class_id}: "
            f"{len(broken[class_id])}"
        )

        if len(
            broken[class_id]
        ) != 0:

            return (
                False,
                inter_boxes,
                broken
            )

        return (
            True,
            inter_boxes,
            broken
        )

    # =========================================================
    # VERIFICATION SUR TOUTES LES CLASSES
    # =========================================================

    def _verif_stable_intra_class(self):

        boxes_inter_by_classe = defaultdict(
            list
        )

        broken = defaultdict(
            list
        )

        features = None

        # Reset au cas où la méthode est appelée plusieurs fois
        self.taux_stability = 0.0
        self.class_stability_rates = {}
        self.contre_exemple = defaultdict(list)

        formal_stable = True

        # Critère expérimental :
        # toutes les classes doivent avoir >= 90 %
        stable_90 = True

        for (
            class_id,
            boxes
        ) in self.boxes_by_class.items():

            if len(boxes) == 0:
                continue

            features = list(
                Boite.f_min(
                    boxes[0]
                ).keys()
            )

            (
                is_stable,
                inter_boxes,
                bk
            ) = (
                self._is_stable_intra_class(
                    class_id,
                    boxes
                )
            )

            # =========================================
            # CAS 1 : aucune violation
            # =========================================

            if is_stable:

                taux_stability = 1.0

                for b in inter_boxes:

                    boxes_inter_by_classe[
                        class_id
                    ].append(b)

            # =========================================
            # CAS 2 : violations
            # =========================================

            else:

                formal_stable = False

                broken_classes = (
                    bk[class_id]
                )

                broken[
                    class_id
                ] = broken_classes

                # Volume total de la classe
                volume_total = sum(
                    Boite.volume(box)
                    for box in boxes
                )

                # Volume des violations
                volume_violation = sum(
                    Boite.volume(box)
                    for box
                    in broken_classes
                )

                if volume_total <= 0:

                    taux_stability = 0.0

                else:

                    taux_stability = (
                        1.0
                        -
                        (
                            volume_violation
                            /
                            volume_total
                        )
                    )

                # Évite éventuellement un taux négatif
                taux_stability = max(
                    0.0,
                    min(
                        1.0,
                        taux_stability
                    )
                )

                # Critère >= 90 % pour cette classe
                if taux_stability >= 0.90:

                    for b in inter_boxes:

                        boxes_inter_by_classe[
                            class_id
                        ].append(b)

                else:

                    stable_90 = False

            # =========================================
            # Enregistrer le taux de la classe
            # =========================================

            self.class_stability_rates[
                class_id
            ] = taux_stability

            print(
                "\n--------------------------------"
            )

            print(
                f"Classe {class_id}"
            )

            print(
                "Formal stability :",
                is_stable
            )

            print(
                "Stability rate : "
                f"{taux_stability * 100:.2f}%"
            )

            print(
                "--------------------------------\n"
            )

        # =====================================================
        # TAUX MOYEN DU MODELE
        # =====================================================

        if len(
            self.class_stability_rates
        ) > 0:

            self.taux_stability = (
                sum(
                    self.class_stability_rates.values()
                )
                /
                len(
                    self.class_stability_rates
                )
            )

        else:

            self.taux_stability = 0.0

        self.formal_stable = formal_stable
        self.stable_90 = stable_90
        self.broken = broken

        # =====================================================
        # AFFICHAGE FINAL
        # =====================================================

        print(
            "\n========================================"
        )

        print(
            "Formal stability :",
            self.formal_stable
        )

        print(
            "Each class >= 90% :",
            self.stable_90
        )

        print(
            "Average stability rate : "
            f"{self.taux_stability * 100:.2f}%"
        )

        print(
            "Rates by class:"
        )

        for (
            class_id,
            rate
        ) in self.class_stability_rates.items():

            print(
                f"  Class {class_id}: "
                f"{rate * 100:.2f}%"
            )

        print(
            "========================================\n"
        )

        # Ici on utilise le critère expérimental des 90 %
        if self.stable_90:

            return (
                True,
                boxes_inter_by_classe,
                features
            )

        return (
            False,
            None,
            features
        )

    # =========================================================
    # API PRINCIPALE
    # =========================================================

    def verif_stable(self):

        (
            stable,
            boxes,
            features
        ) = (
            self._verif_stable_intra_class()
        )

        if stable:

            print(
                "The model satisfies the "
                "90% stability criterion."
            )

        else:

            print(
                "The model does not satisfy the "
                "90% stability criterion."
            )

        print(
            "Average stability rate:",
            self.taux_stability
        )

        print(
            "Formal stability:",
            self.formal_stable
        )

        return (
            stable,
            boxes,
            features
        )