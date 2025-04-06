import numpy as np
import pandas as pd
import copy
import logging
import sys
import csv
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split # for spliting the data
logging.disable(sys.maxsize)
import boxes as bx
import test as t
import stable as st


if (sys.argv[1] == "-m") :
    boxes= bx.classify_and_box(sys.argv[2],sys.argv[3])
    st.verif_stable(boxes)
