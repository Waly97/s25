import os
import sys
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np 
from sklearn.preprocessing import LabelEncoder

def train_and_save_model(csv_path, model_dir="models/", target_column="output"):
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        print(f"❌ La colonne cible '{target_column}' est absente dans {csv_path}.")
        return

    # Séparer features et target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Vérifier que les labels sont bien dans [0, num_class-1]
    unique_y = np.sort(y.unique())
    expected = np.arange(len(unique_y))

    if not np.array_equal(unique_y, expected):
        print(f"⚠️ Labels non consécutifs détectés : {unique_y}, correction automatique...")
        le = LabelEncoder()
        y = le.fit_transform(y)
        unique_y = np.sort(np.unique(y))

    # 🔧 Nettoyer les noms de colonnes pour XGBoost
    X.columns = X.columns.astype(str).str.replace(r"[\[\]<>]", "_", regex=True)

    nb_classes = len(set(y))
    feature_monotones = [1] * X.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'objective': 'multi:softmax',
        'num_class': nb_classes,
        'monotone_constraints': '(' + ','.join([str(m) for m in feature_monotones]) + ')',
        'verbosity': 0
    }

    cv = xgb.cv(params, dtrain, num_boost_round=500, nfold=3, early_stopping_rounds=10)
    best_round = cv.shape[0]

    model = xgb.train(params, dtrain, num_boost_round=best_round)
    preds = model.predict(dval)
    acc = accuracy_score(y_val, preds)

    os.makedirs(model_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(csv_path))[0]
    model.save_model(os.path.join(model_dir, f"{name}.json"))

    print(f"{name}: accuracy = {acc:.2%}, modèle sauvegardé dans {model_dir}{name}.json")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python3 build_mono.py <csv_folder>")
        sys.exit(1)

    csv_folder = sys.argv[1]
    if not os.path.isdir(csv_folder):
        print(f"Erreur : {csv_folder} n’est pas un dossier.")
        sys.exit(1)

    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            train_and_save_model(os.path.join(csv_folder, filename))
