from src.verification.boite import Boite
import numpy as np 
import json
from collections import defaultdict


class CorrectedBoxClassifier:
    def __init__(self, boites_intermediaires, features):
        self.boites = boites_intermediaires
        self.features = features

    def predict(self, point):
        for cls,boxes in self.boites.items():
            for box in boxes:
                fmin = np.array([Boite.f_min(box)[f] for f in self.features])
                fmax = np.array([Boite.f_max(box)[f] for f in self.features])
                if self._point_in_box(point, fmin,fmax):
                    return cls
        return 1  # ou classe par défaut

    def _point_in_box(self, point, fmin,fmax):
         return all(fmin[i] <= point[i] <= fmax[i] for i in range(len(point)))

    
    def predicts(self, dataset):
        results = [(x,self.predict(x)) for x in dataset]
        return results
    
    def save_to_json(self, path):
        serializable = {
            "features": self.features,
            "boites": {
                str(cls): [b.to_dict() for b in boxes]
                for cls, boxes in self.boites.items()
            }
        }
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    @staticmethod
    def load_from_json(path):
        import json
        from collections import defaultdict

        with open(path, "r") as f:
            data = json.load(f)

        # Forcer les noms de features à être des chaînes
        features = [str(f) for f in data["features"]]

        boites = defaultdict(list)
        for cls_str, list_dicts in data["boites"].items():
            cls = int(cls_str)
            for d in list_dicts:
                # Forcer les clés à être des str (au cas où elles seraient des entiers)
                d_str_keys = {str(k): v for k, v in d.items()}
                boites[cls].append(Boite.from_dict(d_str_keys))

        return CorrectedBoxClassifier(boites, features)
