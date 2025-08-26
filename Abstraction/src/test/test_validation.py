from sklearn.metrics import accuracy_score
import sys
import pandas as pd
from src.verification.boite_model import CorrectedBoxClassifier
import numpy as np 
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb




"""
Pour le test :

python3 run_verif_stable.py model/car_evaluation.json datasets_encoded/car_evaluation.csv
"""

def main():
    df = sys.argv[3]
    model = sys.argv[1]
    begin_model= sys.argv[2]

    # Chargement du dataset
    dataset = pd.read_csv(df)
    X = dataset.iloc[:, :-1]
    y_true = dataset.iloc[:, -1]

    # Encodage et nettoyage
    X = X.astype(float).values.tolist()
    y_true = y_true.values

    unique_y = np.sort(np.unique(y_true))
    if not np.array_equal(unique_y, np.arange(len(unique_y))):
        print(f"⚠️ Labels non consécutifs détectés, correction...")
        le = LabelEncoder()
        y_true = le.fit_transform(y_true)

    # Chargement du modèle à base de boîtes
    model_boite = CorrectedBoxClassifier.load_from_json(model)

    # Prédictions du modèle à base de boîtes
    y_pred_boite = [model_boite.predict(x) for x in X]
    acc_boite = accuracy_score(y_true, y_pred_boite)
    print(f"✅ Accuracy du modèle basé sur les boîtes : {acc_boite:.4f}")

    # Entraînement et prédiction avec XGBoost
    m=xgb.Booster()
    m.load_model(begin_model)
    
    dX = xgb.DMatrix(X,feature_names=m.feature_names)
    y_pred_xgb = m.predict(dX)
   
    
    acc_xgb = accuracy_score(y_true, y_pred_xgb)
    print(f"✅ Accuracy du modèle XGBoost : {acc_xgb:.4f}")

    # Comparaison
    diff = acc_xgb - acc_boite
    if diff > 0:
        print(f"📈 XGBoost surpasse les boîtes de {diff:.4f} en accuracy")
    elif diff < 0:
        print(f"📉 Les boîtes surpassent XGBoost de {-diff:.4f} en accuracy")
    else:
        print("⚖️ Les deux modèles ont exactement la même accuracy")

if __name__ == "__main__":
    main()
