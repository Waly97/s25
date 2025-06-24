from xgboost import XGBClassifier
from boite import Boite
from build_boite import  BoitePropagator
# from monotonicity_checker import MonotonicityChecker
from Check_strong_weak_monotonicity import MonotonicityChecker
import sys
from test_validation import test_all_boxes_with_leaf_check
from stable import StabilityChecker



"""
Pour le test :

python3 run_verif_stable.py model/car_evaluation.py datasets_encoded/car_evaluation.py
"""

def main():
    df = sys.argv[2]
    model =sys.argv[1]
    bt = Boite.creer_boite_initiale_depuis_dataset(df)
    
    print("boite initiale ", bt)
    buil_prop = BoitePropagator(model,bt)
    result_final= buil_prop.run()


   
    for k in range(len(result_final)) :
        final_boites = buil_prop.regrouper_boites_par_classe(result_final[k])
        monotone = MonotonicityChecker(final_boites,buil_prop,model)
        monotone.detect_largest_weakly_monotone_group()

if __name__ == "__main__":
    """
    profiling pour observer les fonction qui prend plus de temps pour l'optimisation du code 
    """
    main()
    #cProfile.run('main()','profiling_stats')
    #p=pstats.Stats('profiling_stats')
    # p.sort_stats('cumtime').print_stats(30)
