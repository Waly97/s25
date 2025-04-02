import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
logging.disable(sys.maxsize)

def leq(f1, f2):
    return all(a <= b for a, b in zip(f1, f2))

def f_min(m, instance):
    return np.minimum(m, instance)

def f_max(m, instance):
    return np.maximum(m, instance)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def classify_and_box(model_path, dataset_path, split=False, test_size=0.3):
    print("📦 Chargement du modèle et des données...")

    model = XGBClassifier()
    model.load_model(model_path)
    booster = model.get_booster()

    df = pd.read_csv(dataset_path)

    # Supprimer la colonne cible si elle existe
    possible_targets = ["label", "target", "output", "class", "y"]
    target_col = next((col for col in df.columns if col.lower() in possible_targets), None)

    if target_col:
        print(f"🧹 Colonne cible détectée : '{target_col}' → elle est ignorée.")
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()

    X.columns = [f"f{i}" for i in range(X.shape[1])]

    if split:
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
        print(f"✂️ Split activé : {len(X_train)} train / {len(X_test)} test")
        X_to_predict = X_test
    else:
        X_to_predict = X

    dtest = xgb.DMatrix(X_to_predict, feature_names=X.columns.tolist())

    print("🔎 Prédiction des logits...")
    logits = booster.predict(dtest, output_margin=True)

    if logits.ndim == 1:
        # Binaire
        probs = 1 / (1 + np.exp(-logits))
        pred_classes = (probs >= 0.5).astype(int)
    else:
        # Multi-classe
        probs = np.apply_along_axis(softmax, 1, logits)
        pred_classes = np.argmax(probs, axis=1)

    boxes_by_class = {}

    print("\n🚀 Classification et placement dans les boîtes :")
    for i, instance in tqdm(enumerate(X_to_predict.values), total=len(X_to_predict), ncols=80):
        cls = int(pred_classes[i])
        logit = logits[i]
        proba = probs[i] if probs.ndim > 1 else probs[i]

        print(f"\n📌 Instance {i}")
        print(f"  → Logits     : {logit}")
        print(f"  → Probas     : {proba}")
        print(f"  → Classe prédite : {cls}")

        if cls not in boxes_by_class:
            boxes_by_class[cls] = []

        compatible_found = False
        for box in boxes_by_class[cls]:
            if all(leq(instance, f) or leq(f, instance) for f in box["instances"]):
                box["instances"].append(instance)
                box["f_min"] = f_min(box["f_min"], instance)
                box["f_max"] = f_max(box["f_max"], instance)
                compatible_found = True
                # pas de break → multi-placement

        if not compatible_found:
            boxes_by_class[cls].append({
                "instances": [instance],
                "f_min": instance,
                "f_max": instance
            })

    print("\n✅ Boîtes créées avec succès !")
    return boxes_by_class
