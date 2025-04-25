import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import os
import sys
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp   

from build_boite import BoitePropagator
from stable import StabilityChecker
from boite import Boite


def experimenter_max_leaves(csv_path, target_col, output_dir="models_experiments_leaves", max_leaves_list=[2,4]):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    unique_y = np.sort(y.unique())
    expected = np.arange(len(unique_y))

    if not np.array_equal(unique_y, expected):
        print(f"⚠️ Labels non consécutifs détectés : {unique_y}, correction automatique...")
        le = LabelEncoder()
        y = le.fit_transform(y)
        unique_y = np.sort(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for leaves in max_leaves_list:
        print(f"\n--- Entraînement avec max_leaves = {leaves} ---")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'objective': 'multi:softmax',
            'num_class': len(set(y)),
            'max_depth': 0,
            'max_leaves': leaves,
            'verbosity': 0,
            'grow_policy': 'lossguide',
            'eval_metric': 'merror'
        }

        evals = [(dtrain, 'train'), (dtest, 'test')]
        evals_result = {}

        start = time.time()
        model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, evals_result=evals_result, verbose_eval=False)
        duration = time.time() - start

        preds = model.predict(dtest)
        acc = accuracy_score(y_test, preds)

        model_name = f"{os.path.splitext(os.path.basename(csv_path))[0]}_leaves{leaves}.json"
        model_path = os.path.join(output_dir, model_name)
        model.save_model(model_path)

        # Courbe d’apprentissage
        train_error = evals_result['train']['merror']
        test_error = evals_result['test']['merror']
        plt.figure()
        plt.plot(train_error, label='Train error')
        plt.plot(test_error, label='Test error')
        plt.title(f"Courbe d'apprentissage (max_leaves = {leaves})")
        plt.xlabel("Boosting round")
        plt.ylabel("Erreur de classification")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"{os.path.basename(csv_path)}leaves{leaves}.png")
        plt.savefig(plot_path)
        plt.close()

        # Vérification de la stabilité avec BoitePropagator et StabilityChecker
        try:
            b_init = Boite.creer_boite_initiale_depuis_dataset(csv_path)
            bp = BoitePropagator(model_path, b_init)
            final_boite = bp.run()
            fusion_boxes = bp.regrouper_boites_par_classe(final_boite)
            sbt = StabilityChecker(fusion_boxes,model_path)
            is_stable = sbt.verif_stable()
            stability = "Stable ✅" if is_stable else "Non stable ❌"
        except Exception as e:
            print(f"Erreur lors de la vérification de stabilité : {e}")
            stability = "Erreur ⚠️"

        results.append({
            "max_leaves": leaves,
            "accuracy": round(acc * 100, 2),
            "duration_sec": round(duration, 2),
            "model_path": model_path,
            "learning_curve": plot_path,
            "stability": stability
        })

    print("\n=== Résumé des expériences ===")
    for r in results:
        print(f"max_leaves = {r['max_leaves']} | Acc: {r['accuracy']}% | Durée: {r['duration_sec']}s | Stabilité: {r['stability']} | Modèle: {r['model_path']} | Courbe: {r['learning_curve']}")

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        experimenter_max_leaves(sys.argv[1], "output")
