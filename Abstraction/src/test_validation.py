import xgboost as xgb
import numpy as np
from boite import Boite
from build_boite import BoitePropagator

def get_leaf_path(model, point_array):
    """
    Retourne les indices des feuilles atteintes par un point dans tous les arbres.
    """
    dpoint = xgb.DMatrix(np.array([point_array]))
    return model.predict(dpoint, pred_leaf=True)[0]

def validate_box_against_model(model_path, box, expected_class):
    """
    Prédit fmin, fmax et le centre de la boîte avec XGBoost, et vérifie s'ils donnent tous la bonne classe.
    """
    model = xgb.Booster()
    model.load_model(model_path)

    features = list(box.bornes.keys())
    fmin = Boite.f_min(box)
    fmax = Boite.f_max(box)
    center = {f: (a + b) / 2 for f, (a, b) in box.bornes.items()}

    fmin_array = Boite.to_array(fmin, features)
    fmax_array = Boite.to_array(fmax, features)
    center_array = Boite.to_array(center, features)

    preds = []
    for arr in [fmin_array, fmax_array, center_array]:
        dmat = xgb.DMatrix(np.array([arr]))
        pred = int(model.predict(dmat)[0])
        preds.append(pred)

    return all(p == expected_class for p in preds), preds

def test_all_boxes_with_leaf_check(model_path, boxes_by_class):
    """
    Pour chaque boîte : fmin, fmax, center doivent tous prédire la classe attendue.
    """
    model = xgb.Booster()
    model.load_model(model_path)
    erreurs = []

    testeur = boxes_by_class[0]
    boxe_zero = testeur[0]
    ok, preds = validate_box_against_model(model_path, boxe_zero, 0)
    if not ok:
        erreurs.append((boxe_zero,0, preds))

    if erreurs:
        print(f"\n❌ {len(erreurs)} incohérences détectées :")
        for b, c, p in erreurs[:100]:  # afficher les 5 premières
            print(f" - Classe attendue: {c}, prédictions: {p}, boîte: {b}")
    else:
        print("✅ Toutes les boîtes sont cohérentes avec le modèle XGBoost.")

    return len(erreurs) == 0


