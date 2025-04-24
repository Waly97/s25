import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def tune_xgb(X, y, name="dataset"):
    param_grid = {
        'max_depth': [2,3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [10,25],
        'min_child_weight': [1, 2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.7, 0.9],
        'gamma': [0, 0.1, 0.2],
    }

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=0)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    grid = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=20,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    grid.fit(X, y)

    # Sauvegarde des résultats
    results_df = pd.DataFrame(grid.cv_results_)
    results_df.sort_values(by='mean_test_score', ascending=False).to_csv(f"gridsearch_results_{name}.csv", index=False)
    print(f"📁 Résultats sauvegardés : gridsearch_results_{name}.csv")

    return grid.best_estimator_

def train_and_save_model(csv_path, model_dir="models/", target_column="output"):
    df = pd.read_csv(csv_path)
    name = os.path.splitext(os.path.basename(csv_path))[0]

    if target_column not in df.columns:
        print(f"❌ Colonne cible '{target_column}' absente dans {csv_path}.")
        return

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encodage cible
    if y.dtype == 'object' or not np.array_equal(np.sort(np.unique(y)), np.arange(len(np.unique(y)))):
        print(f"⚠️ Labels non consécutifs détectés : correction automatique...")
        y = LabelEncoder().fit_transform(y)

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col])

    X.columns = X.columns.astype(str).str.replace(r"[\[\]<>]", "_", regex=True)

    # Split train / val / test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Tuning
    best_model = tune_xgb(X_train, y_train, name)
    best_params = best_model.get_params()

    print(f"🔧 Meilleur modèle pour {name} :")
    for k, v in best_params.items():
        print(f"  - {k}: {v}")

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    evals = [(dtrain, 'train'), (dval, 'eval')]
    eval_result = {}

    # xgb.train n'utilise pas tous les paramètres, on filtre
    train_params = {k: v for k, v in best_params.items()
                    if k in ['max_depth', 'learning_rate', 'min_child_weight', 'subsample',
                             'colsample_bytree', 'colsample_bylevel', 'gamma', 'reg_alpha',
                             'reg_lambda', 'max_delta_step', 'scale_pos_weight']}
    train_params.update({
        'objective': 'multi:softmax',
        'num_class': len(np.unique(y)),
        'eval_metric': 'mlogloss',
        'verbosity': 0
    })

    final_model = xgb.train(
        params=train_params,
        dtrain=dtrain,
        evals=evals,
        num_boost_round=best_params.get('n_estimators', 100),
        evals_result=eval_result,
        verbose_eval=False
    )

    # Courbe d'apprentissage
    eval_df = pd.DataFrame(eval_result['eval'])
    plt.figure(figsize=(8, 5))
    for metric in eval_df.columns:
        plt.plot(eval_df[metric], label=metric)
    plt.xlabel("Boosting Round")
    plt.ylabel("Metric")
    plt.title(f"📈 Courbe d'apprentissage - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"learning_curve_{name}.png")
    print(f"📊 Courbe d'apprentissage sauvegardée : learning_curve_{name}.png")

    # Évaluation
    val_acc = accuracy_score(y_val, final_model.predict(dval))
    test_acc = accuracy_score(y_test, final_model.predict(dtest))

    # Sauvegarde modèle
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{name}.json")
    final_model.save_model(model_path)

    print(f"✅ {name} : val_acc = {val_acc:.2%} | test_acc = {test_acc:.2%}")
    print(f"📦 Modèle sauvegardé dans : {model_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python build_model_with_search_and_curve.py <csv_folder>")
        sys.exit(1)

    csv_folder = sys.argv[1]
    if not os.path.isdir(csv_folder):
        print(f"❌ Erreur : {csv_folder} n’est pas un dossier.")
        sys.exit(1)

    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            print(f"\n🚀 Traitement de : {filename}")
            train_and_save_model(os.path.join(csv_folder, filename))
