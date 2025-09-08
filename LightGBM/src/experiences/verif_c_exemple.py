import os
import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    import joblib  # pour charger .pkl/.joblib
except ImportError:
    joblib = None


def _load_lgb_model(model_path):
    """
    Retourne (booster, estimator) où:
      - booster: lightgbm.Booster
      - estimator: éventuellement un LGBMClassifier/Regressor (ou None)
    """
    ext = os.path.splitext(model_path)[1].lower()

    # Cas modèle texte/Booster sauvegardé par LightGBM (save_model)
    if ext in {".txt", ".lgb", ".lightgbm"} or ext == "":
        booster = lgb.Booster(model_file=model_path)
        return booster, None

    # Cas pickle/joblib d'un estimator sklearn LightGBM
    if ext in {".pkl", ".pickle", ".joblib"}:
        if joblib is None:
            raise RuntimeError("Installez joblib pour charger des fichiers pickle/joblib.")
        estimator = joblib.load(model_path)
        if isinstance(estimator, lgb.Booster):
            return estimator, None
        if hasattr(estimator, "booster_"):  # LGBMClassifier/Regressor entraîné
            return estimator.booster_, estimator
        raise ValueError("Fichier chargé, mais type non supporté pour LightGBM.")
    
    # Tentative par défaut (fichiers texte non standard)
    return lgb.Booster(model_file=model_path), None


def _get_feature_names(booster, estimator):
    """
    Essaye d'obtenir les noms de features depuis le booster / estimator.
    Sinon, fallback en f0...f{n-1}.
    """
    names = None
    # Booster (format natif LightGBM)
    try:
        names = booster.feature_name()
    except Exception:
        names = None

    # Estimator sklearn (LGBMClassifier/Regressor)
    if (not names) and estimator is not None:
        names = getattr(estimator, "feature_name_", None)

    # Fallback si vraiment rien
    if not names:
        n = int(booster.num_feature())
        names = [f"f{i}" for i in range(n)]
    return list(names)


def predict(model_path, instance):
    # Charger le modèle LightGBM
    booster, estimator = _load_lgb_model(model_path)

    # Récupérer les noms de colonnes attendus
    feature_names = _get_feature_names(booster, estimator)

    # Construire le DataFrame dans le bon ordre
    df_instance = pd.DataFrame([instance], columns=feature_names)

    # Prédire (pour un binaire: proba de la classe positive ; multiclass: vecteur de probas)
    pred = booster.predict(df_instance)

    # Aplatir et retourner la première valeur par compatibilité avec votre signature
    pred = np.ravel(pred)[0]
    return (instance, float(pred))


# # Exemple :
# model = sys.argv[1]
# instance = [125, 256, 6000, 256, 16, 128]
# print(predict(model, instance))
