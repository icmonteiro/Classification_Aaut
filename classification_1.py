# ===============================================================
# Project  Classification (Exercise Recognition) 
# Instituto Superior Técnico - MEEC
#
# Students:
#   Inês Monteiro (ist1113307)
#   Tiago Anastácio (ist1116348)
#
# Date: 11th of October
# ===============================================================

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np



# --- LOAD DATA ---
data = pd.read_pickle("Xtrain1.pkl") 
Y_train = np.load("Ytrain1.npy")

# --- SEE DATA ---
print("First rows of the DataFrame:")
print(data.head(), "\n")

print("Columns in the DataFrame:")
print(list(data.columns), "\n")

print(f"DataFrame shape: {data.shape}")
print(f"Labels shape (Y_train):   {Y_train.shape}")
# Show the first 20 labels
print(f"First 20 Labels: {Y_train[:20]}")

# --- EXTRACT FEATURES ---

# Each row in 'Skeleton_Features' is an array of 132 features
X_train = np.stack(data['Skeleton_Features'].values)
print(f"X_train shape: {X_train.shape}\n")

# We will use Patient_Id for proper group-based validation
groups = data['Patient_Id'].values


print(f"groups : {groups.shape} (unique patients = {len(np.unique(groups))})")

print(f"Class distribution:")
print(f"  - Class 0 (Hair brushing): {(Y_train == 0).sum()} samples") # E1 (brushing hair) -> Label 0
print(f"  - Class 1 (Teeth brushing): {(Y_train == 1).sum()} samples") #  E2 (brushing teeth) -> Label 1
print(f"  - Class 2 (Hip flexion): {(Y_train == 2).sum()} samples") # E5 (hip flexion) -> Label 2


# --- STATS PER CLASS ---
for class_id in range(3):
    mask = Y_train == class_id
    X_class = X_train[mask]
        
    print(f"\nClass {class_id} statistics:")
    print(f"  Mean feature values: {X_class.mean(axis=0)[:5]}... (showing first 5)")
    print(f"  Standard Deviation feature values: {X_class.std(axis=0)[:5]}...")
    print(f"  Min: {X_class.min():.4f}, Max: {X_class.max():.4f}\n\n")
    

def patient_features(X):
    """
    Extract features that are invariant to patient characteristics, with debug prints.
    """
    n_samples = X.shape[0]
    print(f"Number of samples: {n_samples}")
    
    # Split into averages and standard deviations
    avg_features = X[:, :66]  # x,y averages for 33 keypoints
    std_features = X[:, 66:]  # x,y std for 33 keypoints
    print(f"avg_features shape: {avg_features.shape}, std_features shape: {std_features.shape}")
    print(f"First row of avg_features (first 10 values): {avg_features[0, :10]}")
    print(f"First row of std_features (first 10 values): {std_features[0, :10]}")
    
    # Reshape to (n_samples, 33, 2)
    avg_positions = avg_features.reshape(n_samples, 33, 2)
    std_positions = std_features.reshape(n_samples, 33, 2)
    
    feature_list = []
    
    # --- FEATURE GROUP 1: NORMALIZED POSITIONS ---
    hip_center = (avg_positions[:, 23, :] + avg_positions[:, 24, :]) / 2
    normalized_pos = avg_positions - hip_center[:, np.newaxis, :]
    shoulder_width = np.linalg.norm(avg_positions[:, 11, :] - avg_positions[:, 12, :], axis=1)
    shoulder_width = np.maximum(shoulder_width, 1e-6)
    normalized_pos = normalized_pos / shoulder_width[:, np.newaxis, np.newaxis]
    feature_list.append(normalized_pos.reshape(n_samples, -1))
    print(f"First row normalized positions (first 5 values): {normalized_pos[0, :5, :]}")

    # --- FEATURE GROUP 2: KEY DISTANCES ---
    key_distances = []
    key_distances.append(np.linalg.norm(avg_positions[:, 15, :] - avg_positions[:, 0, :], axis=1))  # Right wrist to nose
    key_distances.append(np.linalg.norm(avg_positions[:, 16, :] - avg_positions[:, 0, :], axis=1))  # Left wrist to nose
    key_distances.append(np.linalg.norm(avg_positions[:, 15, :] - avg_positions[:, 11, :], axis=1))  # Right wrist to shoulder
    key_distances.append(np.linalg.norm(avg_positions[:, 16, :] - avg_positions[:, 12, :], axis=1))  # Left wrist to shoulder
    key_distances.append(np.linalg.norm(avg_positions[:, 27, :] - avg_positions[:, 23, :], axis=1))  # Right knee to hip
    key_distances.append(np.linalg.norm(avg_positions[:, 28, :] - avg_positions[:, 24, :], axis=1))  # Left knee to hip
    key_distances = np.column_stack(key_distances) / shoulder_width[:, np.newaxis]
    feature_list.append(key_distances)
    print(f"First row key distances: {key_distances[0]}")

    # --- FEATURE GROUP 3: VERTICAL POSITIONS ---
    key_points = [0, 15, 16, 23, 24, 27, 28]
    vertical_features = normalized_pos[:, key_points, 1]
    feature_list.append(vertical_features)
    print(f"First row vertical features: {vertical_features[0]}")

    # --- FEATURE GROUP 4: MOVEMENT PATTERNS ---
    std_magnitude = np.linalg.norm(std_positions, axis=2)
    feature_list.append(std_magnitude)
    upper_body_kps = [0, 11, 12, 13, 14, 15, 16]
    lower_body_kps = [23, 24, 25, 26, 27, 28]
    upper_movement = np.mean(std_magnitude[:, upper_body_kps], axis=1)
    lower_movement = np.mean(std_magnitude[:, lower_body_kps], axis=1)
    movement_ratio = upper_movement / (lower_movement + 1e-6)
    feature_list.append(movement_ratio.reshape(-1, 1))
    left_wrist_movement = std_magnitude[:, 16]
    right_wrist_movement = std_magnitude[:, 15]
    wrist_asymmetry = np.abs(left_wrist_movement - right_wrist_movement)
    feature_list.append(wrist_asymmetry.reshape(-1, 1))
    print(f"First row movement ratio: {movement_ratio[0]}, wrist asymmetry: {wrist_asymmetry[0]}")

    # --- FEATURE GROUP 5: EXERCISE-SPECIFIC ---
    max_hand_height = np.maximum(normalized_pos[:, 15, 1], normalized_pos[:, 16, 1])
    head_height = normalized_pos[:, 0, 1]
    hand_elevation = max_hand_height - head_height
    feature_list.append(hand_elevation.reshape(-1, 1))
    print(f"First row hand elevation: {hand_elevation[0]}")

    # Combine all features
    combined_features = np.hstack(feature_list)
    print(f"Shape of combined features: {combined_features.shape}")
    print(f"First row of combined features (first 10 values): {combined_features[0, :10]}")

    return combined_features


X_train_features = patient_features(X_train)



