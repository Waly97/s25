import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
import joblib

def detect_monotone_constraints(X, y, threshold=0.3):
    constraints = []
    for col in X.columns:
        coef, _ = spearmanr(X[col], y)
        if coef >= threshold:
            constraints.append(1)
        elif coef <= -threshold:
            constraints.append(-1)
        else:
            constraints.append(0)
    return constraints

def train_and_save_model(csv_file):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Encode target if needed
    label_encoder = None
    unique_y = np.sort(np.unique(y))
    if not np.array_equal(unique_y, np.arange(len(unique_y))):
        print("⚠️ Labels non consécutifs détectés. Application de LabelEncoder.")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    num_class = len(np.unique(y))
    is_binary = num_class == 2

    # Calcul automatique de scale_pos_weight (pour binaire seulement)
    scale_pos_weight = 1.0
    if is_binary:
        class_counts = pd.Series(y).value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1]
        print(f"📈 Calcul automatique de scale_pos_weight: {scale_pos_weight:.3f}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

    # Détection contraintes de monotonie
    monotone_constraints = detect_monotone_constraints(X_train, y_train)
    print("📈 Contraintes de monotonie détectées :", monotone_constraints)

    # Dataset LGB
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Paramètres
    params = {
        'objective': 'binary' if is_binary else 'multiclass',
        'metric': 'binary_logloss' if is_binary else 'multi_logloss',
        'learning_rate': 0.1,
        'max_depth': 6,
        'verbosity': -1,
        'monotone_constraints': monotone_constraints,
        'scale_pos_weight': scale_pos_weight
    }

    if not is_binary:
        params['num_class'] = num_class

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        valid_names=['valid'],
        callbacks=[lgb.early_stopping(10)]
    )

    # Prédiction
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    if is_binary:
        y_pred_classes = (y_pred > 0.5).astype(int)
    else:
        y_pred_classes = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test, y_pred_classes)
    print(f"✅ Accuracy: {acc:.4f}")

    # Sauvegarde du modèle
    name = os.path.splitext(os.path.basename(csv_file))[0]
    os.makedirs("models", exist_ok=True)

    # 🔹 JSON pour analyse / parsing
    with open(f"models/{name}.json", "w") as f:
        json.dump(model.dump_model(), f, indent=2)

    # 🔹 Sauvegarde natif LightGBM
    model.save_model(f"models/{name}.txt")

    # 🔹 Sauvegarde du label encoder si utilisé
    if label_encoder is not None:
        joblib.dump(label_encoder, f"models/{name}_label_encoder.pkl")
        print("💾 LabelEncoder sauvegardé.")

def test_model_coherence(csv_file, model_txt, encoder_path=None):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1]
    y_true = df.iloc[:, -1]

    # Encoder si besoin
    if encoder_path is not None:
        label_encoder = joblib.load(encoder_path)
        y_true = label_encoder.transform(y_true)

    # Charger le modèle natif
    booster = lgb.Booster(model_file=model_txt)

    # Prédiction
    y_pred = booster.predict(X)
    if y_pred.ndim == 1:
        y_pred_classes = (y_pred > 0.5).astype(int)
    else:
        y_pred_classes = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_true, y_pred_classes)
    print(f"✅ Accuracy sur le dataset complet : {acc:.4f}")

    print("\n🔍 Exemple de 20 prédictions :")
    print(pd.DataFrame({"True": y_true[:20], "Pred": y_pred_classes[:20]}))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage : python build_model_lightgbm.py <mode: train|test> <csv_folder_or_file>")
        sys.exit(1)

    mode = sys.argv[1]
    path = sys.argv[2]

    if mode == "train":
        if not os.path.isdir(path):
            print(f"❌ Erreur : {path} n’est pas un dossier.")
            sys.exit(1)

        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                print(f"\n🚀 Entraînement sur : {filename}")
                train_and_save_model(os.path.join(path, filename))

    elif mode == "test":
        # On suppose que c'est un seul fichier CSV à tester
        if not os.path.isfile(path):
            print(f"❌ Erreur : {path} n’est pas un fichier CSV.")
            sys.exit(1)

        name = os.path.splitext(os.path.basename(path))[0]
        model_txt = f"models/{name}.txt"
        encoder_pkl = f"models/{name}_label_encoder.pkl" if os.path.exists(f"models/{name}_label_encoder.pkl") else None

        print(f"🔍 Test de cohérence pour : {name}")
        test_model_coherence(path, model_txt, encoder_pkl)

    else:
        print("❌ Mode inconnu. Utilise 'train' ou 'test'.")
