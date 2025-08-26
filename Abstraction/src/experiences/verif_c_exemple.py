import json
import sys
import xgboost as xgb
import numpy as np
import pandas as pd 

def predict(model_path, instance):
    # Charger le modèle XGBoost
    booster = xgb.Booster()
    booster.load_model(model_path)

    # Extraire les noms de features depuis le JSON
    with open(model_path, "r") as f:
        model_json = json.load(f)
    feature_names = model_json["learner"]["feature_names"]

    # Créer une DataFrame avec les bons noms
    df_instance = pd.DataFrame([instance], columns=feature_names)

    # Créer le DMatrix avec noms de colonnes
    dtest = xgb.DMatrix(df_instance, feature_names=feature_names)

    # Prédire
    prediction = booster.predict(dtest)
    return (instance,float(prediction[0]))


# # Exemple :
# model= sys.argv[1]
# instance = [125,256,6000,256,16,128]
# print(predict(model, instance))
