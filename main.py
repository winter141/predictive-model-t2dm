"""
Index(['Sex', 'UserID', 'Date', 'Time', 'Timestamp', 'FoodItem', 'Energy',
       'Carbohydrate', 'Protein', 'Fat', 'Tag', 'Weight', 'cgm_window', 'auc'],
"""
import numpy as np
import pandas
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

from process_data import load_dataframe

X_LABELS = ["Energy", "Carbohydrate", "Protein", "Fat"]
Y_LABEL = "auc"
TRAIN_TEST_PROPORTION = 0.8


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
        cgm_df['NZT'] = pd.to_datetime(cgm_df['NZT'])
        cgm_df = cgm_df.sort_values('NZT')

        auc = (cgm_df['value'].rolling(2).mean() * cgm_df['NZT'].diff().dt.total_seconds()).sum()

        return auc

    def reduce(self):
        self.df["auc"] = self.df["cgm_window"].apply(self.reduce_cgm_window_to_area)

        return self.df

    def get_x_y_data(self, x_labels=None, y_label=Y_LABEL):
        if x_labels is None:
            x_labels = X_LABELS

        reduced = self.reduce()

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

    n = len(y_values)
    np.random.shuffle(x_values)
    np.random.shuffle(y_values)

    i = int(n*proportion)
    return x_values[:i], y_values[:i], x_values[i:], y_values[i:]


def gradient_boosting(x_train, y_train):
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5)
    return model.fit(x_train, y_train)


def xgboost(x_train, y_train):
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)

    return model.fit(x_train, y_train)


def print_model_results(model: xgb.XGBRegressor, x_test, y_test):
    predictions = model.predict(x_test)

    # Print regression metrics
    print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
    print("R² Score:", r2_score(y_test, predictions))
    # print([f"{max(p):.2f}" for p in predictions])
    # print(accuracy(model.predict(x_test), y_test))


def accuracy(preds, actual):
    """ TODO predict regression accuracy, maybe MAE, R^2"""
    pass
    # correct = sum(1 for pred, a in zip(preds, actual) if pred == a)
    # return correct / len(preds)


if __name__ == "__main__":
    filepath = "./data/log_with_cgm.pkl"
    df = load_dataframe(filepath)
    cgm_window_ex = df.loc[0, 'cgm_window']
    reducer = FeatureLabelReducer(df)
    x, y = reducer.get_x_y_data()
    x_train, y_train, x_test, y_test = split_train_test(x, y)


    # MODEL
    XGBoost_model = xgboost(x_train, y_train)
    print_model_results(XGBoost_model, x_test, y_test)





