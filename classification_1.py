# ===============================================================
# Project: Classification of Rehabilitation Exercises
# Instituto Superior Técnico - MEEC
#
# Students:
#   Inês Monteiro (ist1113307)
#   Tiago Anastácio (ist1116348)
#
# Date: 11th of October
# ===============================================================

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, balanced_accuracy_score, confusion_matrix, classification_report)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from utils import patient_features

# -------------------------
# Load Data
# -------------------------
data = pd.read_pickle("Xtrain1.pkl")
Y_train = np.load("Ytrain1.npy")
groups = data['Patient_Id'].values
X_raw = np.stack(data['Skeleton_Features'].values)

print(f"Dataset: {X_raw.shape[0]} samples, {X_raw.shape[1]} features, {len(np.unique(Y_train))} classes")
print(f"Unique patients: {len(np.unique(groups))}")


print("Transforming features...")
X_features = patient_features(X_raw)
print(f"Original shape: {X_raw.shape}, Engineered shape: {X_features.shape}")

# -------------------------
# Scale Features
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# -------------------------
# Define Models
# -------------------------
models = {
    'kNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM RBF': SVC(kernel='rbf', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'MLP': MLPClassifier(max_iter=800, random_state=42)
}

# -------------------------
# Baseline with Stratified K-Fold
# -------------------------
print("\nBaseline evaluation with 5-fold Stratified K-Fold:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_scores = {}

for name, clf in models.items():
    scores = cross_val_score(clf, X_scaled, Y_train, cv=skf, scoring='f1_macro')
    baseline_scores[name] = scores.mean()
    print(f"{name:15s} - 5-fold F1: {scores.mean():.4f} (+/- {scores.std():.4f})")

# -------------------------
# LOPO Evaluation
# -------------------------
print("\nLOPO CV evaluation (Leave-One-Patient-Out):")
logo = LeaveOneGroupOut()
results_lopo = {}

for name, clf in models.items():
    scores = cross_val_score(clf, X_scaled, Y_train, groups=groups, cv=logo, scoring='f1_macro')
    results_lopo[name] = scores.mean()
    print(f"{name:15s} - LOPO F1: {scores.mean():.4f} (+/- {scores.std():.4f})")

# -------------------------
# Select Best Model (based on LOPO)
# -------------------------
best_name = max(results_lopo, key=results_lopo.get)
best_model = models[best_name]
print(f"\nSelected model: {best_name} (LOPO F1: {results_lopo[best_name]:.4f})")

# -------------------------
# Train Final Model on Full Dataset
# -------------------------
best_model.fit(X_scaled, Y_train)
Y_pred_train = best_model.predict(X_scaled)

# -------------------------
# Evaluate Training Performance
# -------------------------
train_acc = accuracy_score(Y_train, Y_pred_train)
train_f1 = f1_score(Y_train, Y_pred_train, average='macro')
train_prec = precision_score(Y_train, Y_pred_train, average='macro')
train_rec = recall_score(Y_train, Y_pred_train, average='macro')
train_bal = balanced_accuracy_score(Y_train, Y_pred_train)

print("\nTraining performance:")
print(f"Accuracy: {train_acc:.4f}, F1 macro: {train_f1:.4f}")
print(f"Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, Balanced Acc: {train_bal:.4f}")

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(Y_train, Y_pred_train)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix (Training)')
plt.colorbar()
ticks = ['E1 (hair)', 'E2 (teeth)', 'E5 (hip)']
plt.xticks(np.arange(3), ticks, rotation=45)
plt.yticks(np.arange(3), ticks)
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                 color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

print("\nClassification Report (Training Data):")
print(classification_report(Y_train, Y_pred_train, target_names=ticks, digits=4))

# -------------------------
# Save Model for Submission
# -------------------------
model_data = {
    'model': best_model,
    'scaler': scaler,
    'model_name': best_name,
    'f1_score': results_lopo[best_name]
}

joblib.dump(model_data, "classification_model.pkl")
print(f"\nModel saved as 'classification_model.pkl' ({best_name})")
