from xgboost import XGBClassifier
from boite import Boite
from build_boite import  BoitePropagator
from monotonicity_checker import MonotonicityChecker
import sys
from test_validation import test_all_boxes_with_leaf_check



"""
Pour le test :

python3 run_verif_stable.py model/car_evaluation.py datasets_encoded/car_evaluation.py
"""

def main():
    df = sys.argv[2]
    model =sys.argv[1]
    bt = Boite.creer_boite_initiale_depuis_dataset(df)
    buil_prop = BoitePropagator(model,bt)

    result= buil_prop.run()

    final_boites = buil_prop.regrouper_boites_par_classe(result)
    print("liste des boites finqles",len(final_boites))

    # test_all_boxes_with_leaf_check(model,final_boites)
    order_classes_iris = {
        "setosa": 0 ,
        "versicolor":1,
        "verginica":2
    }

    order_classes_glass={
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    
    order_classes_wine = {0: 0, 1: 1, 2: 2}

    order_classes_titanic = {0: 0, 1: 1}

    order_classes_car_evaluation = {
    "unacc": 0,
    "acc": 1,
    "good": 2,
    "vgood": 3
    }



    
    monotony_checker = MonotonicityChecker(final_boites,buil_prop,order_classes_car_evaluation,model)

    monotony_checker.verif_monotone()
    

if __name__ == "__main__":
    """
    profiling pour observer les fonction qui prend plus de temps pour l'optimisation du code 
    """
    main()
    #cProfile.run('main()','profiling_stats')
    #p=pstats.Stats('profiling_stats')
    # p.sort_stats('cumtime').print_stats(30)
