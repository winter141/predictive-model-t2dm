"""
Lets see if the accuracy improves when doing a weighted ensemble

Lets consider 8 days of data
Step One: compute the actual/expected for each user, using the base model
 - Look into how this varies per user, per subgroup

Step Two: Compute the actual/expected for each user using their unique logging history model
 - Look into how this varies per user

Step Three: Compute the actual/expected for each user with the weighted ensemble model.
"""
from abc import ABC, abstractmethod
import pandas as pd
from matplotlib import pyplot as plt

from model_evaluation.utils import _get_start_dates, _get_df_dict_in_range, _get_user_filtered_df
from models import xgboost, actual_expected_plt
from process_data.main import FeatureLabelReducer, load_dataframe

CGMacro_USER_GROUPS = {
    "healthy": [1,  2, 4, 6, 15, 17, 18, 19, 21, 27, 31, 32, 33, 34, 48],
    "prediabetes": [7, 8,  9, 10, 11, 13, 16, 20, 22, 23, 26, 29, 41, 43, 44, 45],
    "t2dm": [3, 5, 12, 14, 28, 30, 35, 36, 38, 39, 42, 46, 47, 49]
}
DAYS_OF_DATA = 9

def get_train_test_df_dicts(df_dict: dict[str, pd.DataFrame]):
    user_date_mapping: dict = _get_start_dates(df_dict["cgm"])
    train_df_dict: dict[str, pd.DataFrame] = _get_df_dict_in_range(df_dict, user_date_mapping, DAYS_OF_DATA)
    test_df_dict: dict[str, pd.DataFrame] = _get_df_dict_in_range(df_dict, user_date_mapping, day_upper_delta=100, day_lower_delta=DAYS_OF_DATA)
    return train_df_dict, test_df_dict

class BaseAnalysis(ABC):

    @abstractmethod
    def actual_expected_plot(self, title: str, user_ids: list = None):
        pass

class GlobalModel(BaseAnalysis):
    def __init__(self, train_df_dict: dict[str, pd.DataFrame], test_df_dict: dict[str, pd.DataFrame]):
        self.train_df_dict = train_df_dict
        self.test_df_dict = test_df_dict

    def actual_expected_plot(self, title: str, user_ids: list = None):
        if user_ids is not None and len(user_ids) == 1:
            raise ValueError("Global model, must require at least 2 users. Please use the LocalModel.")

        f_train_dict = self.train_df_dict
        f_test_dict = self.test_df_dict
        if user_ids is not None:
            f_train_dict = _get_user_filtered_df(self.train_df_dict, user_ids)
            f_test_dict = _get_user_filtered_df(self.test_df_dict, user_ids)

        # Train Data
        reducer = FeatureLabelReducer(f_train_dict)
        feature_names, x_train, y_train = reducer.get_x_y_data()

        # Test Data
        reducer = FeatureLabelReducer(f_test_dict)
        _, x_test, y_test = reducer.get_x_y_data()

        model = xgboost(x_train, y_train)
        predictions = model.predict(x_test)

        actual_expected_plt(predictions, y_test, f"Global Model | {DAYS_OF_DATA} Days | {title}")


class LocalModel(BaseAnalysis):
    """
    Separate Model for each user.
    """
    def __init__(self, train_df_dict: dict[str, pd.DataFrame], test_df_dict: dict[str, pd.DataFrame]):
        if "static_user" in train_df_dict or "static_user" in test_df_dict:
            raise KeyError("Local Model does not allow static user dataframe")
        self.train_df_dict = train_df_dict
        self.test_df_dict = test_df_dict


    def get_prediction_test(self, user_id):
        train_df_dict = _get_user_filtered_df(self.train_df_dict, [user_id])
        test_df_dict = _get_user_filtered_df(self.test_df_dict, [user_id])
        if len(train_df_dict["log"]) == 0 or len(test_df_dict["log"]) == 0:
            return [], []

        # Train Data
        reducer = FeatureLabelReducer(train_df_dict)
        feature_names, x_train, y_train = reducer.get_x_y_data()

        # Test Data
        reducer = FeatureLabelReducer(test_df_dict)
        _, x_test, y_test = reducer.get_x_y_data()

        model = xgboost(x_train, y_train)
        predictions = model.predict(x_test)
        return predictions, y_test

    def actual_expected_plot(self, title: str = "", user_ids: list = None):
        """
        Exclude any static user information.
        """
        if user_ids is None:
            user_ids = self.train_df_dict["cgm"]["UserID"].unique()

        predictions = []
        y_test = []
        for user_id in user_ids:
            new_predictions, new_y_test = self.get_prediction_test(user_id)
            predictions.extend(new_predictions)
            y_test.extend(new_y_test)

        if len(predictions) == 0:
            print("User(s) has insufficient data")
            return

        actual_expected_plt(predictions, y_test, f"Local Model | {DAYS_OF_DATA} Days | {title}")

class WeightedEnsembleModel(BaseAnalysis):
    def __init__(self):
        pass
    def actual_expected_plot(self, title: str, user_ids: list = None):
        pass

if __name__ == "__main__":
    base_file_path = "../data/CGMacros/pickle/"
    df_dict = dict()
    for pkl in ["cgm", "dynamic_user", "log", "static_user"]:
        df_dict[pkl] = load_dataframe(base_file_path + pkl + ".pkl")

    train, test = get_train_test_df_dicts(df_dict)
    global_model = GlobalModel(train, test)
    # global_model.actual_expected_plot("Healthy", user_ids=CGMacro_USER_GROUPS["healthy"])
    # global_model.actual_expected_plot("Prediabetes", user_ids=CGMacro_USER_GROUPS["prediabetes"])
    # global_model.actual_expected_plot("T2DM", user_ids=CGMacro_USER_GROUPS["t2dm"])
    # global_model.actual_expected_plot("Single User", user_ids=[1])
    #
    global_model.actual_expected_plot("ALL")

    # LOCAL MODEL
    local_df_dict = df_dict.copy()
    del local_df_dict["static_user"]
    local_train, local_test = get_train_test_df_dicts(local_df_dict)
    local_model = LocalModel(local_train, local_test)
    # local_model.actual_expected_plot("All Users")
