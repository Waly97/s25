import pandas as pd # for read csv files
import sys # for reading the line aguments
from sklearn.calibration import LabelEncoder
from sklearn.isotonic import spearmanr
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split # for spliting the data
from sklearn.metrics import accuracy_score # for calculte the accuracy
import os
import numpy as np 


def detect_monotone_constraints(X, y, threshold=0.3):
    constraints = []
    for col in X.columns:
        coef, _ = spearmanr(X[col], y)
        if coef >= threshold:
            constraints.append(1)
        elif coef <= -threshold:
            constraints.append(-1)
        else:
            constraints.append(0)
    return constraints


def train_and_save_model(csv_file):
    dataset = pd.read_csv(csv_file) # Datasets needs header
    ### splitting the Data
    # Number of features
    feature_names = list(dataset.columns)
    nb_feature = len(feature_names) # including the output
    # Firsts columns are the features 
    X = dataset.iloc[:,0:(nb_feature-1)]
    # Last column is the target
    Y = dataset.iloc[:,(nb_feature-1)]
    # Find the number of classes
   
    # split the data into a 67:33 train:test ratio

     # Correction des labels si besoin
    unique_y = np.sort(np.unique(Y))
    if not np.array_equal(unique_y, np.arange(len(unique_y))):
        print(f"⚠️ Labels non consécutifs détectés, correction...")
        le = LabelEncoder()
        Y = le.fit_transform(Y)

    nb_classes = len(set(Y))
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=7)

    ### Training the XGBoost Model

    ### Xgboost Learning API
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # parameters of learning
    # create the monotony constraint : all increasing
    monotone_constraints = detect_monotone_constraints(X, Y)
    # print("Monotone constraints:", monotone_constraints)

    params = {
        'objective' : "multi:softmax",
        'learning_rate' : 0.1,
        'num_class' : nb_classes,
        'max_depth':6,
        'verbosity': 0,  # 0 is silent, 3 is debug
        'monotone_constraints' : '(' + ','.join([str(m) for m in monotone_constraints]) + ')',# Monotony constraints
        'booster' : 'gbtree'
    }

    # Use CV to find the best number of trees
    bst_cv = xgb.cv(params, dtrain,500, nfold = 2, early_stopping_rounds=10)

    model = xgb.train(params=params,
                    dtrain=dtrain,
                    num_boost_round = bst_cv.shape[0],
                    verbose_eval=None)

    ### Making predictions on the Test Data
    y_pred = model.predict(dtest)
    predictions = [round(value) for value in y_pred]

    ### Testing the XGBoost Model Performance
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


    ### Saving the model
    # Get the name of the dataset
    dirs = sys.argv[1].split('/')
    name = os.path.splitext(os.path.basename(csv_file))[0]
    model.save_model("model/"+name+".json")
    # model.save_model("models/"+name+".txt") # Not needed both format

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python build_mono.py <csv_folder>")
        sys.exit(1)

    csv_folder = sys.argv[1]
    if not os.path.isdir(csv_folder):
        print(f"❌ Erreur : {csv_folder} n’est pas un dossier.")
        sys.exit(1)

    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            print(f"\n🚀 Traitement de : {filename}")
            train_and_save_model(os.path.join(csv_folder, filename))