from enum import Enum
from typing import Union
import numpy as np
import shap
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

TRAIN_TEST_PROPORTION = 0.8


class ModelType(Enum):
    XGBOOST = 0,
    GRADIENT_BOOSTING = 1


def split_train_test(x_values, y_values, proportion=TRAIN_TEST_PROPORTION) -> tuple[list, list, list, list]:
    """
    :return: x_train, y_train, x_test, y_test
    """

    valid_mask = np.array([
        not any(
            (not isinstance(col, (bool, np.bool_))) and np.isnan(col)
            for col in row
        )
        for row in x_values
    ])

    x_values = x_values[valid_mask]
    y_values = y_values[valid_mask]

    n = len(y_values)
    permutation = np.random.permutation(n)
    x_values = x_values[permutation]
    y_values = y_values[permutation]
    i = int(n*proportion)
    return x_values[:i], y_values[:i], x_values[i:], y_values[i:]


def gradient_boosting(x_train: list,
                      y_train: list,
                      n_estimators=1_000,
                      learning_rate=0.05,
                      max_depth=8
                      ) -> object:
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )
    return model.fit(x_train, y_train)


def xgboost(x_train: list,
            y_train: list,
            n_estimators=1_000,
            learning_rate=0.05,
            max_depth=8
            ) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )
    return model.fit(x_train, y_train)


def SHAP_analysis(x_test, model, feature_names, shap_out: Union[str, None] = None,):
    explainer = shap.Explainer(model)
    shap_values = explainer(x_test)

    plt.title("SHAP Summary Plot")
    shap.summary_plot(shap_values, features=x_test, feature_names=feature_names, show=False)
    if shap_out is not None:
        plt.savefig(shap_out, bbox_inches='tight')
    plt.show()
    plt.close()
    # shap.dependence_plot("FeatureA", shap_values, X, interaction_index="auto")


def PDP_analysis(model, x_test, feature_names, pdp_out):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title("Partial Dependence Plots")
    display = PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=x_test,
        features=range(len(feature_names)),
        feature_names=feature_names,
        ax=ax,
    )
    display.figure_.subplots_adjust(hspace=0.5)

    if pdp_out is not None:
        plt.savefig(pdp_out, bbox_inches='tight')
    plt.show()
    plt.close()


def actual_expected_plt(preds, y_test, out):
    plt.plot(preds, y_test, 'o')

    if out is not None:
        plt.savefig(out, bbox_inches='tight')
    plt.show()


def get_new_preds(x: list, y: list, model_type: ModelType = ModelType.XGBOOST):
    """
    Split data and re-build model and find predictions
    :return: Tuple of (pred, y_test)
        pred: 1D NumPy array, prediction from model
        y_test: 1D NumPy array, actual test output
    """
    x_train, y_train, x_test, y_test = split_train_test(x, y)

    if model_type == ModelType.XGBOOST:
        model = xgboost(x_train, y_train)
    elif model_type == ModelType.GRADIENT_BOOSTING:
        model = gradient_boosting(x_train, y_train)
    else:
        raise ValueError("Model Type not supported")

    return model.predict(x_test), y_test



if __name__ == "__main__":
    pass
