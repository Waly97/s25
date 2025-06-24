from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff("Abstraction/datasets/notebook/mammo.arff")
df = pd.DataFrame(data)
df.to_csv("Abstraction/datasets/encoded/mammo.csv", index=False)
