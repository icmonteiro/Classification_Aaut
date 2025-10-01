# ===============================================================
# Project  Part 1 - Regression 
# Instituto Superior Técnico - MEEC
#
# Students:
#   Inês Monteiro (ist1113307)
#   Tiago Anastácio (ist1116348)
#
# Date: September 27, 2025
# ===============================================================

import pandas as pd
import numpy as np


#load data from first classification problem 
#load 1

data = pd.read_pickle('Xtrain1.pkl')
Y_train = np.load('Ytrain1.pkl')

#print first lines
data.head()

#check shapes
print(data.shape)
print(Y_train.shape)
