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
    groupes_onehot = [
    ["f6","f7"],
    ["f8","f9"],
    ["f10","f11"],
    ["f12","f13"],
    ["f14","f15"]
]
 
#     groupes_onehot_mamo = [
#     ["f3","f4","f5","f6"],
#     ["f7","f8","f9","f10","f11"]
# ]
    bt = Boite.creer_boite_initiale_depuis_dataset(df)
    
    print("boite initiale ", bt)
    buil_prop = BoitePropagator(model,bt,group_one_hot=groupes_onehot)

    result_final= buil_prop.run()


    i=1
    final_boite1=  buil_prop.regrouper_boites_par_classe(result_final[0])
    monotony_checker = MonotonicityChecker(final_boite1,buil_prop,model)
    F,_= monotony_checker.detect_largest_group_monotone_features()
    valid=False
    for k in range(1,len(result_final)) :
        final_boites = buil_prop.regrouper_boites_par_classe(result_final[k])
        print("boite init ",i)
        monotony_checker = MonotonicityChecker(final_boites,buil_prop,model)
        f,_=monotony_checker.detect_largest_group_monotone_features()
        if f == F:
            valid=True
        else:
            valid= False
        i+=1
    if valid:
        print(f"✅ Groupe de monotonie forte détecté : {F}")
    else :
        print("❌ Groupe de monotonie non detecté")

if __name__ == "__main__":
    """
    profiling pour observer les fonction qui prend plus de temps pour l'optimisation du code 
    """
    main()
    #cProfile.run('main()','profiling_stats')
    #p=pstats.Stats('profiling_stats')
    # p.sort_stats('cumtime').print_stats(30)
