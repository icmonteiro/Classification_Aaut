import numpy as np

def patient_features(X):
    n = X.shape[0]
    avg_pos = X[:, :66].reshape(n, 33, 2) 
    std_pos = X[:, 66:].reshape(n, 33, 2)

    # ----------------------------------------------------
    # 1. Normalização
    # ----------------------------------------------------
    hip_center = (avg_pos[:, 23] + avg_pos[:, 24]) / 2 
    avg_centered = avg_pos - hip_center[:, None, :] 
    shoulder_width = np.linalg.norm(avg_pos[:, 11] - avg_pos[:, 12], axis=1, keepdims=True) 
    shoulder_width = np.maximum(shoulder_width, 1e-6)
    avg_norm = avg_centered / shoulder_width[:, None, :] 

    movement = np.linalg.norm(std_pos, axis=2) 

    # ----------------------------------------------------
    # 2. Métricas de Alcance e Elevação (21 Features Simples)
    # Estas ajudam o MLP a identificar as ações E1, E2, E5
    # ----------------------------------------------------
    Wrist_R, Wrist_L = avg_norm[:, 16], avg_norm[:, 15]
    Elbow_R, Elbow_L = avg_norm[:, 14], avg_norm[:, 13]
    Shoulder_R, Shoulder_L = avg_norm[:, 12], avg_norm[:, 11]
    Ear_R, Ear_L = avg_norm[:, 8], avg_norm[:, 7] 
    Hip_R, Hip_L = avg_norm[:, 24], avg_norm[:, 23] 
    Nose = avg_norm[:, 0]
    
    # 2.1 Distância (Alcance)
    d_rwrist_ear = np.linalg.norm(Wrist_R - Ear_R, axis=1) # E1
    d_lwrist_ear = np.linalg.norm(Wrist_L - Ear_L, axis=1)
    d_rwrist_nose = np.linalg.norm(Wrist_R - Nose, axis=1) # E2
    d_lwrist_nose = np.linalg.norm(Wrist_L - Nose, axis=1)

    # 2.2 Elevação Vertical (Y)
    # Pulso acima do ombro (Reach)
    wrist_height_r = Wrist_R[:, 1] - Shoulder_R[:, 1]
    wrist_height_l = Wrist_L[:, 1] - Shoulder_L[:, 1]
    # Cotovelo acima do ombro
    elbow_height_r = Elbow_R[:, 1] - Shoulder_R[:, 1]
    elbow_height_l = Elbow_L[:, 1] - Shoulder_L[:, 1]
    # Joelho acima da anca (E5)
    knee_height_r = avg_norm[:, 26, 1] - Hip_R[:, 1]
    knee_height_l = avg_norm[:, 25, 1] - Hip_L[:, 1]

    # 2.3 Lateralidade (X)
    # Pulso relativo ao centro (nariz)
    dist_x_center_r = Wrist_R[:, 0] - Nose[:, 0]
    dist_x_center_l = Wrist_L[:, 0] - Nose[:, 0]
    
    # 2.4 Relação de Movimento
    upper_mov = movement[:, [0, 11, 12, 15, 16]].mean(axis=1)
    lower_mov = movement[:, [23, 24, 27, 28]].mean(axis=1)
    mov_ratio = upper_mov / (lower_mov + 1e-6)
    
    # Relação de Coordenadas (E5)
    d_rk_hip = np.linalg.norm(avg_pos[:, 27] - avg_pos[:, 23], axis=1) / shoulder_width.squeeze()
    d_lk_hip = np.linalg.norm(avg_pos[:, 28] - avg_pos[:, 24], axis=1) / shoulder_width.squeeze()

    extra_metrics = np.column_stack([
        d_rwrist_ear, d_lwrist_ear, d_rwrist_nose, d_lwrist_nose,
        wrist_height_r, wrist_height_l, elbow_height_r, elbow_height_l,
        knee_height_r, knee_height_l, 
        dist_x_center_r, dist_x_center_l, 
        mov_ratio, d_rk_hip, d_lk_hip
    ])
    

    
    pos_norm_features = avg_norm.reshape(n, -1) 
    X_new = np.hstack([pos_norm_features, movement, extra_metrics]) 
    
    return X_new