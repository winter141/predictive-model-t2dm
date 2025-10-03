import numpy as np
from scipy.stats import pearsonr

from model_user_evaluation.utils import split_train_test_dicts_per_user
from model_user_evaluation.weighted_ensemble_model2 import get_global_SuFmTgTf, UserRsGenerator
from models import xgboost
from process_data.main import load_dataframe, FeatureLabelReducer, get_x_y_from_features


def split_train_test_ordered(x_values, y_values, proportion=0.8):
    """
    Split into train/test by cutoff index, preserving the order of the data.
    Assumes x_values and y_values are already sorted (e.g. by time).

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
    cutoff_idx = int(n * proportion)

    x_train = x_values[:cutoff_idx]
    y_train = y_values[:cutoff_idx]
    x_test = x_values[cutoff_idx:]
    y_test = y_values[cutoff_idx:]

    return x_train, y_train, x_test, y_test


base_file_path = "../data/CGMacros/pickle/"
df_dict = dict()
for pkl in ["cgm", "dynamic_user", "log", "static_user"]:
    df_dict[pkl] = load_dataframe(base_file_path + pkl + ".pkl")

# print(get_global_SuFmTgTf(df_dict))
for train_proportion in [0.8]:
    feature_groups = {
            "static_user": ["Sex", "Body weight", "Height", "Self-identity"],
            "log_macros": ["Energy", "Carbohydrate", "Protein", "Fat"],
            "temporal_cgm": ["cgm_p30", "cgm_p60", "cgm_p120"],
            "temporal_food": ["meal_hour", "time_since_last_meal"]
        }
    global_feature_groups = {
            "static_user": ["Sex", "Body weight", "Height", "Self-identity"],
            "log_macros": ["Energy", "Carbohydrate", "Protein", "Fat"],
            "temporal_cgm": ["cgm_p30", "cgm_p60", "cgm_p120"],
            "temporal_food": ["meal_hour", "time_since_last_meal"]
        }

    # Train a single global model
    for i in range(30):
        user_train_test, all_train_dict, all_test_dict = split_train_test_dicts_per_user(df_dict, train_ratio=train_proportion)
        reducer = FeatureLabelReducer(all_train_dict, global_feature_groups)
        feature_names, xs, ys = reducer.get_x_y_data()
        global_model = xgboost(xs, ys)
        x_test, y_test = get_x_y_from_features(reducer, FeatureLabelReducer(all_test_dict, global_feature_groups))

        print("Train Proportion", train_proportion, "R", pearsonr(global_model.predict(x_test), y_test))





