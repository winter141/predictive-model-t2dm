"""
Index(['Sex', 'UserID', 'Date', 'Time', 'Timestamp', 'FoodItem', 'Energy',
       'Carbohydrate', 'Protein', 'Fat', 'Tag', 'Weight', 'cgm_window', 'auc'],
"""
import numpy as np
import pandas
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


from process_data import load_dataframe

X_LABELS = ["Energy", "Carbohydrate", "Protein", "Fat"]
Y_LABEL = "auc"
TRAIN_TEST_PROPORTION = 0.8
MIN_CGM_READINGS = 20  # Minimum cgm readings to qualify food for model


class FeatureLabelReducer:

    def __init__(self, df: pandas.DataFrame):
        """
        :param df: pickled object dataframe
            includes field: cgm_window which contains a dataframe of cgm value readings
        """
        self.df = df

    @staticmethod
    def reduce_cgm_window_to_area(cgm_df):
        """
        Compute trapezoidal area between each pair.

        :param cgm_df:
        :return:
        """
        if len(cgm_df) < MIN_CGM_READINGS:
            return np.nan
        cgm_df['NZT'] = pd.to_datetime(cgm_df['NZT'])
        cgm_df = cgm_df.sort_values('NZT')

        # TODO Update this
        baseline = cgm_df.iloc[0]["value"]

        auc = ((cgm_df['value'].rolling(2).mean() - baseline) * cgm_df['NZT'].diff().dt.total_seconds() / 60).sum()

        return auc

    def reduce(self):
        self.df["auc"] = self.df["cgm_window"].apply(self.reduce_cgm_window_to_area)

        return self.df

    def get_x_y_data(self, x_labels=None, y_label=Y_LABEL):
        if x_labels is None:
            x_labels = X_LABELS

        reduced = self.reduce()

        # Remove points with iAUC of nan
        reduced = reduced[reduced["auc"].notna()]

        x_values = reduced[x_labels].values
        y_values = reduced[y_label].values

        return x_values, y_values


def split_train_test(x_values, y_values, proportion=TRAIN_TEST_PROPORTION) -> tuple[list, list, list, list]:
    """
    :return: x_train, y_train, x_test, y_test
    """
    # TEMPORARY
    x_values = np.where(np.isinf(x_values), np.nan, x_values)
    y_values = np.where(np.isinf(y_values), np.nan, y_values)

    valid_mask = ~np.isnan(x_values).any(axis=1)

    x_values = x_values[valid_mask]
    y_values = y_values[valid_mask]

    n = len(y_values)
    np.random.shuffle(x_values)
    np.random.shuffle(y_values)
    i = int(n*proportion)
    return x_values[:i], y_values[:i], x_values[i:], y_values[i:]


def gradient_boosting(x_train, y_train):
    model = GradientBoostingRegressor(
        n_estimators=1_000,
        learning_rate=0.05,
        max_depth=8
    )
    return model.fit(x_train, y_train)


def xgboost(x_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=1_000,
        max_depth=8,
        learning_rate=0.05,
    )
    return model.fit(x_train, y_train)


def print_model_results(model: xgb.XGBRegressor, x_test, y_test):
    predictions = model.predict(x_test)

    # Print regression metrics
    print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
    print("R² Score:", r2_score(y_test, predictions))
    # print([f"{max(p):.2f}" for p in predictions])
    # print(accuracy(model.predict(x_test), y_test))


def plt_model_results(model, x_test, y_test):
    preds = model.predict(x_test)

    min_val = min(min(preds), min(y_test))
    max_val = max(max(preds), max(y_test))

    r_value, p_value = pearsonr(preds, y_test)
    print(f"R_value {r_value}, p_value: {p_value}")

    plt.plot(preds, y_test, 'o', ms=3)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # red dashed y=x line
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel("Predicted PPGR")
    plt.ylabel("Measured PPGR")

    plt.show()

def accuracy(preds, actual):
    """ TODO predict regression accuracy, maybe MAE, R^2"""
    pass
    # correct = sum(1 for pred, a in zip(preds, actual) if pred == a)
    # return correct / len(preds)



if __name__ == "__main__":
    filepath = "./data/log_with_cgm.pkl"
    df = load_dataframe(filepath)

    # ----------------------------------- #

    reducer = FeatureLabelReducer(df)
    x, y = reducer.get_x_y_data()
    x_train, y_train, x_test, y_test = split_train_test(x, y)

    # MODEL
    XGBoost_model = xgboost(x_train, y_train)
    GradientBoost_model = gradient_boosting(x_train, y_train)

    plt_model_results(GradientBoost_model, x_test, y_test)
    # R_value 0.03724978155568783, p_value: 0.3070168799836446

    # plt_model_results(XGBoost_model, x_test, y_test)
    # R_value 0.03276457510200377, p_value: 0.3686363759122124


    # CREATE LOTS OF GRAPHS AND PDT





