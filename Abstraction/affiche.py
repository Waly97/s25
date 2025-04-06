import sys
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_tree



model_path = sys.argv[1]
model = XGBClassifier()
model.load_model(model_path)
xgb.plot_tree(model, tree_idx=0)
plt.show()

