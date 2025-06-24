import pandas as pd
import numpy as np
import time
import sys
sys.path.append("../../")
import lightgbm as lgb


#Defining the results array which ´will contain execution time and non-monotonicity score
resultArr = np.zeros((2, ))
df = pd.read_csv('Datasets/Mammographic.csv') 
#Applying monotonicity constraints
fileMon = open('monFeature.txt', 'w')
fileMon.write('BI-RADS \nAge \nDensity')
fileMon.close()
noOfFe = 3

data = df.values

X = data[:, :-1]
Y = data[:, -1]

dfT = pd.read_csv('Datasets/Mammographic.csv') 
dfT.drop('Class', axis=1, inplace=True)

feature_names = dfT.columns.values


feature_monotones = [0] * (len(feature_names))
with open('monFeature.txt') as f:
	feArr = f.readlines()
feArr = [x.strip() for x in feArr]

constr=1
#Adding monotonicity constraints
if(constr == 1):
	for i in range(noOfFe):
		for j in range(dfT.shape[1]):
			if(feArr[i] == dfT.columns.values[j]):
				feature_monotones[j] = 1  

monotone_model = lgb.LGBMClassifier(min_child_samples=5, monotone_constraints=feature_monotones)
model = monotone_model.fit(data[:, :-1].reshape(len(X), len(feature_names)), Y)
# Save in txt format
model.booster_.save_model("models/Mammo.txt")

# Save in json format (manually)
import json
model_dict = model.booster_.dump_model()
with open("models/Mammo.json", "w") as f:
    json.dump(model_dict, f, indent=2)
