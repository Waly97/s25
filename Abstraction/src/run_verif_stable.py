from xgboost import XGBClassifier
from boite import Boite
from build_boite import  BoitePropagator
from stable import StabilityChecker
import sys
import xgboost as xgb
import pandas as pd
import build_boite as bb
import json
import cProfile
import pstats 

"""
Pour le test :

python3 run_verif_stable.py model/car_evaluation.py datasets_encoded/car_evaluation.py
"""

def main():
    df = sys.argv[2]
    model =sys.argv[1]
    bt = Boite.creer_boite_initiale_depuis_dataset(df)
    buil_prop = BoitePropagator(model,bt)

    result = buil_prop.run()

    final_boite = buil_prop.regrouper_boites_par_classe(result)

    # tracer_toutes_zones_2D(final_boite)
    verif_stablity= StabilityChecker(final_boite,model)

    verif_stablity.verif_stable()
    

if __name__ == "__main__":
    """
    profiling pour observer les fonction qui prend plus de temps pour l'optimisation du code 
    """
    cProfile.run('main()','profiling_stats')
    p=pstats.Stats('profiling_stats')
    p.sort_stats('cumtime').print_stats(30)
