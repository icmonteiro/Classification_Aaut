# ===============================================================
# verification.py
# Verification script to check if mymodel.py runs with correct shapes
# ===============================================================

import numpy as np
from mymodel import predict

# Simulate a test dataset as the teacher would provide
# 200 samples, 132 features (raw Skeleton_Features)
X_test = np.random.rand(200, 132)

# Make predictions using your submitted predict function
y_pred = predict(X_test)

# Verify output shape
if y_pred.shape != (200,):
    raise ValueError(f"Shape mismatch: {y_pred.shape} vs (200,)")

# Optional: check if predictions are valid class indices (0,1,2)
if not np.all(np.isin(y_pred, [0, 1, 2])):
    print("Warning: some predictions are outside the expected class range {0,1,2}")

print("Prediction format is valid.")
print(f"First 20 predictions: {y_pred[:20]}")