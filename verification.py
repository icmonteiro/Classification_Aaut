import pandas as pd
import numpy as np
from mymodel import predict
from sklearn.metrics import f1_score, accuracy_score

# Load training/test data
X_test_raw = pd.read_pickle("Xtrain1.pkl")
X_test = np.stack(X_test_raw['Skeleton_Features'].values)
Y_test = np.load("Ytrain1.npy")

# Make predictions
Y_pred = predict(X_test)

# Print first 20 predictions
print("First 20 predictions:", Y_pred[:20])

# Quick metrics
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("F1 macro:", f1_score(Y_test, Y_pred, average='macro'))
