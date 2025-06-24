import pandas as pd
import numpy as np



class Boite:
    __slots__ = ('bornes',)  # Évite la création du __dict__ par instance

    def __init__(self, bornes):
        """
        bornes : dict[int, list[float]]
        Exemple : {0: [0.0, 5.0], 1: [1.0, 3.0]}
        """
        self.bornes = {k: list(v) for k, v in bornes.items()}

    @classmethod
    def from_bounds(cls, fmin, fmax):
        bornes = {f: (fmin[f], fmax[f]) for f in fmin}
        return cls(bornes)

    def copy(self):
        return Boite(self.bornes.copy())

    def split(self, feature, threshold):
        a, b = self.bornes[feature]
        # Tout à gauche
        if b <= threshold:
            return self, None

        # Tout à droite
        if a > threshold:
            return None, self

        # Cas général : split réel
        left_bornes = dict(self.bornes)
        right_bornes = dict(self.bornes)

        if (a == int(a)):
            left_bornes[feature] = [a, threshold]  # prendre l'entier oou le réel qui vient avant le seuil
            right_bornes[feature] = [threshold + 1, b]
        else:
            left_bornes[feature] = [a, threshold]  # prendre l'entier oou le réel qui vient avant le seuil
            right_bornes[feature] = [np.nextafter(np.float32(threshold), +np.float32(np.inf)), b] 

        return Boite(left_bornes), Boite(right_bornes)

    def is_valid(self):
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
        return {f: a for f, (a, b) in self.bornes.items()}

    def f_max(self):
        return {f: b for f, (a, b) in self.bornes.items()}

    def __repr__(self):
        return f"Boite({{ {', '.join(f'f{feat}: [{a}, {b}]' for feat, (a, b) in self.bornes.items())} }})"

    def to_interval_instance(self):
        fmin = {f: a for f, (a, b) in self.bornes.items()}
        fmax = {f: b for f, (a, b) in self.bornes.items()}
        return {"fmin": fmin, "fmax": fmax}

    @staticmethod
    def creer_boite_initiale_depuis_dataset(df):
        df = pd.read_csv(df)
        df = df.iloc[:, :-1]
        bornes = {i: [df.iloc[:, i].min(), df.iloc[:, i].max()] for i in range(df.shape[1])}
        return Boite(bornes)

    @staticmethod
    def to_array(b, feature_order):
        return np.array([b[f] for f in feature_order], dtype=np.float32)
    

    @staticmethod
    def volume(boite):
        return np.prod([b-a if b-a != 0 else 1 for (a,b) in boite.bornes.values()])
