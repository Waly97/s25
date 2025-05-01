import xgboost as xgb
import numpy as np
import pandas as pd
from boite import Boite

def extract_instance_from_boite(boite:Boite, mode='min'):
    """
    Extrait une instance (vecteur) à partir d'une boîte.
    mode: 'min' ou 'max'
    """
    if mode == 'min':
        return [bounds[0] for bounds in boite.bornes.values()]
    elif mode == 'max':
        return [bounds[1] for bounds in boite.bornes.values()]
    else:
        raise ValueError("mode must be 'min' or 'max'")

def predict_from_boites(m, boite1, boite2, feature_names=None):
    # Auto-détection des noms de features

    model = xgb.Booster()
    model.load_model(m)
    if feature_names is None:
        if getattr(model, 'feature_names', None) is not None:
            feature_names = model.feature_names
        elif hasattr(boite1, 'bornes'):
            feature_names = list(boite1.bornes.keys())
        else:
            raise ValueError("Impossible de déduire les noms des features.")

    # Nettoyage
    import re
    feature_names = [re.sub(r'[^A-Za-z0-9_]', '_', str(f)) for f in feature_names]

    # Création des 3 instances
    x1 = extract_instance_from_boite(boite1, mode='min')
    x2 = extract_instance_from_boite(boite2, mode='min')
    x3 = extract_instance_from_boite(boite2, mode='max')

    # Prédictions une par une
    pred1 = model.predict(xgb.DMatrix([x1], feature_names=feature_names))[0]
    pred2 = model.predict(xgb.DMatrix([x2], feature_names=feature_names))[0]
    pred3 = model.predict(xgb.DMatrix([x3], feature_names=feature_names))[0]

    return {
        'boite1_min': (x1, pred1),
        'boite2_min': (x2, pred2),
        'boite2_max': (x3, pred3)
    }