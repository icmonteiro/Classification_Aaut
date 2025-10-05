# ===============================================================
# Project: Classification of Rehabilitation Exercises
# Instituto Superior Técnico - MEEC
#
# Students:
#   Inês Monteiro (ist1113307)
#   Tiago Anastácio (ist1116348)
#
# Date: 5th October
# ===============================================================

import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)
from sklearn.ensemble import VotingClassifier

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
# Define Models and Parameter Grids
# -------------------------
logo = LeaveOneGroupOut()

model_grids = {
    'kNN': {
        'pipeline': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())]),
        'params': {'clf__n_neighbors':[3,5,7], 'clf__weights':['uniform','distance']}
    },
    'Naive Bayes': {
        'pipeline': Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())]),
        'params': {}
    },
    'SVM RBF': {
        'pipeline': Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', class_weight='balanced', random_state=42))]),
        'params': {'clf__C':[0.1,1,10], 'clf__gamma':['scale','auto']}
    },
    'Decision Tree': {
        'pipeline': Pipeline([('scaler', StandardScaler()), ('clf', DecisionTreeClassifier(class_weight='balanced', random_state=42))]),
        'params': {'clf__max_depth':[None,5,10,20], 'clf__min_samples_split':[2,5,10]}
    },
    'Random Forest': {
        'pipeline': Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))]),
        'params': {'clf__n_estimators':[50,100,200],'clf__max_depth':[None,10,20],'clf__min_samples_split':[2,5]}
    },
    'MLP': {
        'pipeline': Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(max_iter=1000, random_state=42, early_stopping=True))]),
        'params': {'clf__hidden_layer_sizes':[(100,),(100,50),(150,50)], 'clf__alpha':[0.0001,0.001,0.01]}
    }
}

# -------------------------
# Grid Search LOPO per model
# -------------------------
results_lopo = {}
best_estimators = {}

for name, mg in model_grids.items():
    print(f"\nRunning GridSearchCV for {name}...")
    grid = GridSearchCV(
        estimator=mg['pipeline'],
        param_grid=mg['params'],
        scoring='f1_macro',
        cv=logo,
        n_jobs=-1
    )
    grid.fit(X_features, Y_train, groups=groups)
    results_lopo[name] = grid.best_score_
    best_estimators[name] = grid.best_estimator_
    print(f"{name} - Best LOPO F1: {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")

# -------------------------
# Select Best Model
# -------------------------
best_name = max(results_lopo, key=results_lopo.get)
best_model = best_estimators[best_name]
print(f"\nSelected model: {best_name} (LOPO F1: {results_lopo[best_name]:.4f})")

# -------------------------
# Train Final Model on Full Dataset
# -------------------------
best_model.fit(X_features, Y_train)

# -------------------------
# Evaluate Training Performance
# -------------------------
Y_pred_train = best_model.predict(X_features)
train_acc = accuracy_score(Y_train, Y_pred_train)
train_f1 = f1_score(Y_train, Y_pred_train, average='macro')
train_prec = precision_score(Y_train, Y_pred_train, average='macro')
train_rec = recall_score(Y_train, Y_pred_train, average='macro')
train_bal = balanced_accuracy_score(Y_train, Y_pred_train)

print("\nTraining performance:")
print(f"Accuracy: {train_acc:.4f}, F1 macro: {train_f1:.4f}")
print(f"Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, Balanced Acc: {train_bal:.4f}")

# -------------------------
# LOPO Confusion Matrix (out-of-fold)
# -------------------------
print("\nGenerating LOPO out-of-fold predictions for confusion matrix...")
y_pred_lopo = cross_val_predict(best_model, X_features, Y_train, groups=groups, cv=logo, method='predict')
cm_lopo = confusion_matrix(Y_train, y_pred_lopo)

plt.figure(figsize=(6, 5))
plt.imshow(cm_lopo, cmap='Blues')
plt.title(f'Confusion Matrix (LOPO out-of-fold) - {best_name}')
plt.colorbar()
ticks = ['E1 (hair)', 'E2 (teeth)', 'E5 (hip)']
plt.xticks(np.arange(3), ticks, rotation=45)
plt.yticks(np.arange(3), ticks)
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm_lopo[i, j], ha='center', va='center',
                 color='white' if cm_lopo[i, j] > cm_lopo.max()/2 else 'black')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

print("\nClassification Report (LOPO out-of-fold):")
print(classification_report(Y_train, y_pred_lopo, target_names=ticks, digits=4))

# -------------------------
# Save Best Model for Submission
# -------------------------
model_data = {
    'model': best_model,
    'model_name': best_name,
    'f1_score': results_lopo[best_name]
}

joblib.dump(model_data, "classification_model.pkl")
print(f"\nBest model saved as 'classification_model.pkl' ({best_name})")
