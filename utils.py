import numpy as np

def patient_features(X):
    n = X.shape[0]
    avg_pos = X[:, :66].reshape(n, 33, 2)
    std_pos = X[:, 66:].reshape(n, 33, 2)

    # Center on hip center
    hip_center = (avg_pos[:, 23] + avg_pos[:, 24]) / 2
    avg_centered = avg_pos - hip_center[:, None, :]

    # Scale by shoulder width
    shoulder_width = np.linalg.norm(avg_pos[:, 11] - avg_pos[:, 12], axis=1, keepdims=True)
    shoulder_width = np.maximum(shoulder_width, 1e-6)
    avg_norm = avg_centered / shoulder_width[:, None, :]

    # Movement per keypoint
    movement = np.linalg.norm(std_pos, axis=2)  # shape (n, 33)

    # Extra features (6)
    # Distances and hand/hip features
    d_rh_head = np.linalg.norm(avg_pos[:, 15] - avg_pos[:, 0], axis=1) / shoulder_width.squeeze()
    d_lh_head = np.linalg.norm(avg_pos[:, 16] - avg_pos[:, 0], axis=1) / shoulder_width.squeeze()
    d_rk_hip = np.linalg.norm(avg_pos[:, 27] - avg_pos[:, 23], axis=1) / shoulder_width.squeeze()
    d_lk_hip = np.linalg.norm(avg_pos[:, 28] - avg_pos[:, 24], axis=1) / shoulder_width.squeeze()
    max_hand_y = np.maximum(avg_norm[:, 15, 1], avg_norm[:, 16, 1])
    head_y = avg_norm[:, 0, 1]
    hand_elevation = max_hand_y - head_y
    upper_mov = movement[:, [0, 11, 12, 15, 16]].mean(axis=1)
    lower_mov = movement[:, [23, 24, 27, 28]].mean(axis=1)
    mov_ratio = upper_mov / (lower_mov + 1e-6)

    extra_features = np.column_stack([d_rh_head, d_lh_head, d_rk_hip, d_lk_hip, hand_elevation, mov_ratio])

    # Concatenate all: 66 + 33 + 6 = 105
    X_new = np.hstack([avg_norm.reshape(n, -1), movement, extra_features])
    return X_new
