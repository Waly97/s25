# s25
This project has for goal the verification of the stability and the monptonicity of Gradian boosting tree models such that XGBOOST and LightGBM..

If you want to test other models different of XGBOOST or LightGBM you must adapted to the adequate implementation of split (located to the file Abstraction/src/verification/boite) of the models used and to applique it at the line 19 of the code located in Abstraction/src/verification/arbre. 

# Depandence: 
To compile and run this code, you will need to install a few libraries:

XGBoost: https://xgboost.readthedocs.io/en/stable/install.html 

LightGBM: https://lightgbm.readthedocs.io/en/stable/ 

panda: pip innstall panda 
...

# Implementation : 
The implementation of this project is in Python. 
All the source code is located in the src folders.

To realize  the tests over my models, use the files located in the folder : Abstraction/src/expériences

If you wish to test your owner models, you must encode your datasets in ordinary or in one-hot and train your model with this dataset encoded using XGBoost (note that you must return a model using tree). After your place your datasets encoded in the folder "Abtraction/guest_datasets" and your model in the "folder guest_models". For command test follow this commandes:

- For monotonicity :  python3 src/experiences/experience_monotonie.py 'folder of datasets' 'folder of models'

-For stability whithout one hot encoded : python3 src/experiences/experience_monotonie.py 'folder of datasets encoded in one hot' 'folder of models'

-For stability with one hot encoded : python3 src/experiences/experience_one_hot.py 'folder of datasets encoded in one hot' 'folder of models'

# 



