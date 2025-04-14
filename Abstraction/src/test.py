from xgboost import XGBClassifier
from boite import Boite
import sys
import xgboost as xgb
import pandas as pd
import build_boite as bb
import json
from multiprocessing import freeze_support

def main():
    df = pd.read_csv(sys.argv[2])
    bt = Boite.creer_boite_initiale_depuis_dataset(df)
    final_boite = bb.regrouper_boites_par_classe(bb.propagate_boites_cascade(sys.argv[1], bt))
    print(final_boite)

if __name__ == "__main__":
    freeze_support()  # ← Nécessaire si tu fais un exécutable, sinon safe à garder
    main()
