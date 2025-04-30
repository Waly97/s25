import pandas as pd
import os
import xgboost as xgb
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
import urllib.request

DATASETS_INFO = {
    "iris": {
        "loader": load_iris,
        "target_col": "target",
    },
    "wine": {
        "loader": load_wine,
        "target_col": "target",
    },
    "glass": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
        "columns": [
            "Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"
        ],
        "target_col": "Type",
    },
    # D'autres datasets peuvent être ajoutés ici
}

def download_glass():
    """Télécharge le dataset glass"""
    path = "datasets/glass.csv"
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    urllib.request.urlretrieve(
        DATASETS_INFO["glass"]["url"], path
    )
    print(f"✅ glass téléchargé dans {path}")
    return path

def prepare_dataset(name):
    if name == "glass":
        path = download_glass()
        df = pd.read_csv(path, names=DATASETS_INFO[name]["columns"])
    else:
        loader = DATASETS_INFO[name]["loader"]
        data = loader()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target

    # Vérification cible
    target_col = DATASETS_INFO[name]["target_col"]
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible {target_col} introuvable dans {name}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Corrige les labels pour être 0, ..., num_classes - 1
    y = pd.Series(pd.Categorical(y).codes)

    # Encodage one-hot pour X
    X_encoded = pd.get_dummies(X)

    final_df = pd.concat([X_encoded, y.rename("target")], axis=1)

    output_csv = f"datasets/encoded/{name}_encoded.csv"
    if not os.path.exists("datasets/encoded"):
        os.makedirs("datasets/encoded")
    final_df.to_csv(output_csv, index=False)
    print(f"✅ Encodé et enregistré dans {output_csv}")

    return final_df

def train_xgboost(df, name):
    X = df.drop(columns=["target"])
    y = df["target"]

    dtrain = xgb.DMatrix(data=X.values, label=y.values)

    params = {
        "objective": "multi:softmax",
        "num_class": len(set(y)),
        "verbosity": 0,
        "monotone_constraints": "(" + ",".join(["1"] * X.shape[1]) + ")",
    }

    model = xgb.train(params, dtrain, num_boost_round=50)

    if not os.path.exists("models"):
        os.makedirs("models")

    model.save_model(f"models/{name}.json")
    print(f"✅ Modèle XGBoost sauvegardé dans models/{name}.json")

def main():
    datasets = ["iris", "wine", "glass"]  # tu peux ajouter ici d'autres noms
    for name in datasets:
        try:
            print(f"\n⬇️ Traitement de {name}...")
            df = prepare_dataset(name)
            train_xgboost(df, name)
        except Exception as e:
            print(f"❌ Erreur pour {name}: {e}")

if __name__ == "__main__":
    main()
