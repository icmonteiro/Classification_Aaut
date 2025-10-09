import numpy as np

def patient_features(X):
    n = X.shape[0]
    avg_pos = X[:, :66].reshape(n, 33, 2)
    std_pos = X[:, 66:].reshape(n, 33, 2)

    # --- normalize ---
    hip_center = (avg_pos[:, 23] + avg_pos[:, 24]) / 2
    avg_centered = avg_pos - hip_center[:, None, :]
    shoulder_width = np.linalg.norm(avg_pos[:, 11] - avg_pos[:, 12], axis=1, keepdims=True)
    shoulder_width = np.maximum(shoulder_width, 1e-6)
    avg_norm = avg_centered / shoulder_width[:, None, :]

    # --- movement per keypoint ---
    movement = np.linalg.norm(std_pos, axis=2)             # (n,33)
    mov_x = std_pos[:, :, 0]                               # std of x
    mov_y = std_pos[:, :, 1]                               # std of y

    # --- base relations ---
    head = avg_norm[:, 0]
    lh, rh = avg_norm[:, 15], avg_norm[:, 16]
    mouth_L, mouth_R = avg_norm[:, 9], avg_norm[:, 10]

    # distances
    d_rh_head = np.linalg.norm(rh - head, axis=1)
    d_lh_head = np.linalg.norm(lh - head, axis=1)
    d_rh_mouth = np.linalg.norm(rh - mouth_R, axis=1)
    d_lh_mouth = np.linalg.norm(lh - mouth_L, axis=1)

    # hand elevation
    hand_elev = np.maximum(lh[:,1], rh[:,1]) - head[:,1]

    # upper/lower movement ratio
    upper_mov = movement[:, [0,11,12,15,16]].mean(axis=1)
    lower_mov = movement[:, [23,24,27,28]].mean(axis=1)
    mov_ratio = upper_mov / (lower_mov + 1e-6)

    # --- NEW: motion direction ratio (horizontal vs vertical) ---
    #  >1 → vertical motion, <1 → horizontal motion
    hand_dir_ratio = (mov_y[:,15] + mov_y[:,16]) / (mov_x[:,15] + mov_x[:,16] + 1e-6)

    # --- NEW: horizontal alignment with mouth (brushing teeth cue) ---
    mouth_x = (mouth_L[:,0] + mouth_R[:,0]) / 2
    hand_mouth_xdiff = np.abs(((lh[:,0] + rh[:,0]) / 2) - mouth_x)
    

    # --- combine ---
    extra_features = np.column_stack([
        d_rh_head, d_lh_head, d_rh_mouth, d_lh_mouth,
        hand_elev, mov_ratio, hand_dir_ratio, hand_mouth_xdiff
    ])  # 8 extras

    

    X_new = np.hstack([
        avg_norm.reshape(n, -1),   # 66
        movement,                  # 33
        extra_features             # 8 → total 107
    ])
    return X_new
