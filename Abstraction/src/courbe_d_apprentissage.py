import matplotlib.pyplot as plt
import sys
from xgboost import XGBClassifier


model_path = sys.argv[1]
model = XGBClassifier()
model.load_model(model_path)
results = model.evals_result()
plt.plot(results['validation_0']['logloss'], label='Train')
plt.plot(results['validation_1']['logloss'], label='Test')
plt.legend()
plt.title("Courbe d'apprentissage")
plt.show()