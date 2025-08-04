from pathlib import Path
import joblib
import xgboost as xgb
import numpy as np
from scipy.stats import pearsonr

MODEL_DIR: Path = Path("../models/")


def load_model(filename) -> tuple[list[str], xgb.XGBRegressor]:
    """Return: (feature_names, model)"""
    return joblib.load(MODEL_DIR / filename)


def load_test_inputs(rows: int) -> tuple[np.array, np.array]:
    """Pick {rows} random rows"""
    x = np.load("../data/CGMacros/feature_label/x.npy", allow_pickle=True)
    y = np.load("../data/CGMacros/feature_label/y.npy", allow_pickle=True)

    n = len(x)
    permutation = np.random.permutation(n)

    return x[permutation][:rows], y[permutation][:rows]


def predict(model: xgb.XGBRegressor, inputs):
    return model.predict(inputs)

def compare(preds, y_test):
    r, p = pearsonr(preds, y_test)
    print(f"R: {r}, p: {p}")


if __name__ == "__main__":
    feature_names, model = load_model("model_v1.pkl")
    test_x, test_y = load_test_inputs(10)

    predictions = predict(model, test_x)

    # ---
    compare(predictions, test_y)
