import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def infer_monotonicity(X):
    """
    Détecte automatiquement les contraintes de monotonie
    selon les noms de colonnes.
    
    Retourne une liste [+1, -1, 0, ...]
    """
    increasing_keywords = ['age', 'taille', 'revenu', 'income', 'experience', 'years']
    decreasing_keywords = ['prix', 'cost', 'charge', 'expense', 'loss']

    constraints = []
    for col in X.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in increasing_keywords):
            constraints.append(1)
        elif any(word in col_lower for word in decreasing_keywords):
            constraints.append(-1)
        else:
            constraints.append(0)
    print(f"🔎 Contraintes détectées automatiquement : {constraints}")
    return constraints


def train_best_xgboost_models(
    folder_path,
    max_depths=[2, 4, 6],
    num_boost_rounds=[10,25, 50, 100],
    auto_monotonicity=True,
    force_all_monotonicity=None 
):
    """
    Entraine XGBoost sur tous les fichiers CSV d'un dossier,
    avec détection automatique des contraintes de monotonie.
    
    Arguments:
        folder_path: chemin du dossier contenant les CSV
        max_depths: liste des profondeurs d'arbres à tester
        num_boost_rounds: liste des nombres de boosting rounds à tester
        auto_monotonicity: booléen pour activer la détection automatique
    """

    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not files:
        print("Aucun fichier CSV trouvé dans le dossier.")
        return

    for file in files:
        print(f"\n✨ Traitement du fichier : {file}")
        dataset_path = os.path.join(folder_path, file)
        data = pd.read_csv(dataset_path)

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Correction des labels si besoin
        unique_y = np.sort(np.unique(y))
        if not np.array_equal(unique_y, np.arange(len(unique_y))):
            print(f"⚠️ Labels non consécutifs détectés, correction...")
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model_name_base = os.path.splitext(file)[0]

        best_model = None
        best_accuracy = 0
        best_params = None
        best_result = None

        # Préparation des contraintes de monotonie automatiques
        monotone_constraints = None
        if force_all_monotonicity is not None:
            # Forcer toutes les colonnes à la même contrainte (+1 ou -1)
            monotone_constraints = [force_all_monotonicity] * X_train.shape[1]
            print(f"🔗 Contraintes forcées sur toutes les features : {monotone_constraints}")
        elif auto_monotonicity:
            # Sinon détection automatique
            monotone_constraints = infer_monotonicity(X_train)

        for depth in max_depths:
            for n_rounds in num_boost_rounds:
                print(f"\n🔍 Entrainement avec max_depth={depth}, num_boost_round={n_rounds}...")

                params = {
                    'objective': 'multi:softmax',
                    'num_class': len(np.unique(y)),
                    'max_depth': depth,
                    'eta': 0.3,
                    'verbosity': 1,
                    'grow_policy': 'lossguide',
                    'booster': 'gbtree',
                    'eval_metric': 'merror'
                }

                if monotone_constraints is not None:
                    if isinstance(monotone_constraints, list):
                        monotone_constraints = "(" + ",".join(map(str, monotone_constraints)) + ")"
                    params['monotone_constraints'] = monotone_constraints
                evals_result = {}
                bst_cv = xgb.cv(params, dtrain,n_rounds, nfold = 2, early_stopping_rounds=10)
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=bst_cv.shape[0],
                    evals=[(dtrain, 'train'), (dtest, 'eval')],
                    evals_result=evals_result,
                    verbose_eval=False
                )

                y_pred = model.predict(dtest)
                acc = accuracy_score(y_test, y_pred)
                print(f"Accuracy de validation: {acc:.4f}")

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = model
                    best_params = (depth, n_rounds)
                    best_result = evals_result

        print(f"\n✅ Meilleur modèle pour {file} : max_depth={best_params[0]}, num_boost_round={best_params[1]} avec accuracy={best_accuracy:.4f}")

        # Création des dossiers si besoin
        os.makedirs("models", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

        # Sauvegarder le meilleur modèle
        model_save_path = f"models/{model_name_base}_best_depth{best_params[0]}_rounds{best_params[1]}.json"
        best_model.save_model(model_save_path)
        print(f"📦 Modèle sauvegardé sous {model_save_path}")

        # Tracer la courbe du meilleur modèle
        plt.figure(figsize=(12, 8))
        plt.plot(
            range(len(best_result['eval']['merror'])),
            best_result['eval']['merror'],
            label=f'depth={best_params[0]}, rounds={best_params[1]}'
        )
        plt.xlabel("Boosting Rounds")
        plt.ylabel("Classification Error")
        plt.title(f"Courbe d'apprentissage pour {file}")
        plt.legend()
        plt.grid(True)

        plot_path = f"plots/learning_curve_{model_name_base}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"📈 Courbe sauvegardée sous {plot_path}")


# ====== Exemple d'utilisation super simple ======

train_best_xgboost_models(
    folder_path="datasets/encoded",
    auto_monotonicity=False,
    force_all_monotonicity=1
)

