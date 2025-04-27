import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

if len(sys.argv) < 2:
    print("Usage: python build_model.py <path_to_dataset.csv>")
    sys.exit(1)

# Charger les données
dataset_path = sys.argv[1]
data = pd.read_csv(dataset_path)

# Séparation features/target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

 # Vérifier que les labels sont bien dans [0, num_class-1]
unique_y = np.sort(y.unique())
expected = np.arange(len(unique_y))

if not np.array_equal(unique_y, expected):
    print(f"⚠️ Labels non consécutifs détectés : {unique_y}, correction automatique...")
    le = LabelEncoder()
    y = le.fit_transform(y)
    unique_y = np.sort(np.unique(y))

# Découpage en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création des DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Paramètres du modèle
nb_classes = len(set(y))
params = {
    'objective': 'multi:softmax',
    'num_class': nb_classes,
    'max_depth': 4,             # Limite la profondeur de chaque arbre
    'eta': 0.1,                 # Taux d’apprentissage
    'verbosity': 1,
    'grow_policy': 'lossguide',
    'booster': 'gbtree',
    'eval_metric': 'merror'
}

# Définir le nombre d'arbres (boost rounds)
num_boost_round = 300


evals_result = {}
eval = [(dtrain,'train'),(dtest,'eval')]
# Entraînement
model = xgb.train(params, dtrain, num_boost_round=num_boost_round,evals=eval,evals_result=evals_result,verbose_eval=False)

# Prédictions
y_pred = model.predict(dtest)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

# Sauvegarde du modèle
model_name = os.path.splitext(os.path.basename(dataset_path))[0]
model.save_model(f"models/{model_name}.json")
print(f"Modèle sauvegardé sous models/{model_name}.json")

# Courbe d'apprentissage 
plt.figure(figsize=(10,6))
epochs = range(len(evals_result['train']['merror']))
plt.plot(epochs,evals_result['train']['merror'], label = 'Train Error')
plt.plot(epochs,evals_result['eval']['merror'], label = 'Test Error')
plt.xlabel("Boosting Rounds")
plt.ylabel("Classification Error")
plt.title("Courbe d apprentissage XGBOOST")
plt.legend()
plt.grid(True)
os.makedirs("Courbe_App",exist_ok=True)
plot_path = f"Courbe_App/{os.path.basename(dataset_path)}.png"
plt.savefig(plot_path)
print(f"Courbe d'apprentissage sauvegarder sous {plot_path}")
