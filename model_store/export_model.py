"""
Exporting the model with joblib which persists an any arbitary python object.

Alternatives are using ONNX models, however this is not necessary.
"""
import joblib
import numpy as np
import xgboost as xgb
from models import split_train_test, xgboost


def get_trained_model() -> tuple[list[str], xgb.XGBRegressor]:
    x = np.load("./data/CGMacros/feature_label/x.npy", allow_pickle=True)
    y = np.load("./data/CGMacros/feature_label/y.npy", allow_pickle=True)
    feature_names = np.load("./data/CGMacros/feature_label/feature_names.npy", allow_pickle=True)
    x_train, y_train, x_test, y_test = split_train_test(x, y)
    model: xgb.XGBRegressor = xgboost(x_train, y_train)

    return feature_names, model


def main():
    filename = "model_v1.pkl"
    feature_names, model = get_trained_model()
    joblib.dump(value=[feature_names, model], filename=f"./models/{filename}")


if __name__ == "__main__":
    main()
