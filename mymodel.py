import numpy as np
import joblib
#from utils import patient_features

# Carregar o modelo
model_data = joblib.load("classification_model.pkl")
model = model_data['model']
scaler = model_data['scaler']

def predict(Xtest):
    """
    Xtest: np.array de shape (200,132)
    retorna: np.array de shape (200,) com as previsões
    """
    X_features = patient_features(Xtest)
    X_scaled = scaler.transform(X_features)
    return model.predict(X_scaled)