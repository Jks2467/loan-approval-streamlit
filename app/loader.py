import os
import joblib

def load_model(clf_path, reg_path):
    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"Classifier model file not found at {clf_path}")
    if not os.path.exists(reg_path):
        raise FileNotFoundError(f"Regressor model file not found at {reg_path}")

    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)

    return clf, reg