import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import sys

### Lire les données depuis un fichier CSV
csv_file = sys.argv[1]
dataset = pd.read_csv(csv_file)  # Le dataset doit avoir un en-tête

### Séparation des caractéristiques et de la cible
feature_names = list(dataset.columns)
nb_feature = len(feature_names)  # Nombre total de colonnes
X = dataset.iloc[:, 0:(nb_feature - 1)]  # Toutes les colonnes sauf la dernière
Y = dataset.iloc[:, (nb_feature - 1)]  # Dernière colonne = cible

# Diviser les données en ensemble d'entraînement (80%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

# Créer et entraîner un modèle XGBoost
model = xgb.XGBClassifier(n_estimators=10, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Exactitude du modèle : {accuracy:.4f}")
print("\n🔍 Rapport de classification :\n", classification_report(y_test, y_pred))

# Afficher tous les arbres de décision
num_trees = model.get_booster().num_boosted_rounds()

for i in range(num_trees):
    plt.figure(figsize=(12, 6))
    plot_tree(model, num_trees=i)  # Afficher chaque arbre
    plt.title(f"Arbre de décision {i+1}")
    plt.show()
