import pandas as pd


class Boite:
    def __init__(self, bornes):
        """
        bornes : dict[int, list[float]]
        Exemple : {0: [0.0, 5.0], 1: [1.0, 3.0]}
        """
        self.bornes = {k: list(v) for k, v in bornes.items()}

    @classmethod
    def from_bounds(cls,fmin,fmax):
        bornes = {f:(fmin[f],fmax[f]) for f in fmin}
        return cls(bornes)

    def copy(self):
        return Boite({k: list(v) for k, v in self.bornes.items()})

    def split(self, feature, threshold):
        feature = int(feature) 
        a, b = self.bornes[feature]
        threshold = float(threshold)

        # Cas spécial : l'intervalle est déjà figé à une seule valeur
        if a == b:
            if a < threshold:
                left = self
                return left, None
            elif a >= threshold:
                right = self
                return None, right
        if b < threshold :
            return self,None
        if a >= threshold :
            return None,self
        
        if float(b) == threshold:
            left=self.copy()
            right=self.copy()
            left.bornes[feature]= [a,(b-1)]
            right.bornes[feature]=[b,b]
            return left,right

        # Cas général
        left_min, left_max = a, min(b, threshold)
        right_min, right_max = max(a, threshold), b

        left = self.copy() if left_min < left_max else None
        right = self.copy() if right_min < right_max else None

        if left:
            left.bornes[feature] = [left_min, left_max]
        if right:
            right.bornes[feature] = [right_min, right_max]

        return left, right

    def is_valid(self):
        """Autorise les boîtes plates (a == b)"""
        return all(a <= b for a, b in self.bornes.values())
    

    def intersection(self, other):
        result = {}
        for f in self.bornes:
            a1, b1 = self.bornes[f]
            a2, b2 = other.bornes[f]
            a, b = max(a1, a2), min(b1, b2)
            if a > b:
                return None
            result[f] = [a, b]
        return Boite(result)

    def f_min(self):
        return  {f:a for f,(a,b) in self.bornes.items()}

    def f_max(self):
        return  {f: b for f,(a,b) in self.bornes.items()}

    def __repr__(self):
        return f"Boite({{ {', '.join(f'f{feat}: [{a}, {b}]' for feat, (a, b) in self.bornes.items())} }})"
    

    def to_interval_instance(self):
        fmin= {f:a for f,(a,b) in self.bornes.items()}
        fmax = {f: b for f,(a,b) in self.bornes.items()}
        return {"fmin":fmin, "fmax":fmax}
    

    @staticmethod
    def creer_boite_initiale_depuis_dataset(df):
        df = pd.read_csv(df)
        boxes = []
        if "label" in df.columns:
            df = df.drop(columns=["label"])
        if "output" in df.columns:
            df = df.drop(columns=["output"])
        bornes = {i: [df.iloc[:, i].min(), df.iloc[:, i].max()] for i in range(df.shape[1])}
        boite = Boite(bornes)
        # boxes.append(boite)
        return boite
