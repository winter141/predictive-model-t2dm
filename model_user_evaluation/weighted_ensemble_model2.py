"""
Going to look at a 80/20 split as opposed to the date range (simulation) method.

User Categorisation of Global Model: R=0.739

Global Model: SuFmTgTf: R=0.707

Local Model: FmTgTf:
    - In a globally trained/tested model: R=0.537
    - When trained and tested locally:
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from create_model_summary import run_r_iterations
from model_user_evaluation.utils import _get_user_filtered_df, split_train_test_dicts_per_user, \
    one_hot_encode_self_identity
from models import xgboost, split_train_test
from process_data.main import load_dataframe, FeatureLabelReducer, get_x_y_from_features

CGMacro_USER_GROUPS = {
    "healthy": [1,  2, 4, 6, 15, 17, 18, 19, 21, 27, 31, 32, 33, 34, 48],
    "prediabetes": [7, 8,  9, 10, 11, 13, 16, 20, 22, 23, 26, 29, 41, 43, 44, 45],
    "t2dm": [3, 5, 12, 14, 28, 30, 35, 36, 38, 39, 42, 46, 47, 49]
}


class UserRsGenerator():
    """
    Generate different user models, based on a set of features, return list of rs
    """
    def __init__(self, all_df_dict: dict, include_groups: dict):
        self.df_dict = all_df_dict
        self.include_groups = include_groups

    def get_rs(self, user_ids: list = None):
        f_df_dict = self.df_dict
        if user_ids is not None:
            f_df_dict = _get_user_filtered_df(self.df_dict, user_ids)

        reducer = FeatureLabelReducer(f_df_dict, self.include_groups)
        feature_names, xs, ys = reducer.get_x_y_data()

        rs = run_r_iterations(xs, ys, r_iterations=10)
        # print(np.mean(rs))
        # x_train, y_train, x_test, y_test = split_train_test(xs, ys)
        #
        # model = xgboost(x_train, y_train)
        return rs, len(xs)

    def get_model_details(self, user_ids: list = None):
        f_df_dict = self.df_dict
        if user_ids is not None:
            f_df_dict = _get_user_filtered_df(self.df_dict, user_ids)

        reducer = FeatureLabelReducer(f_df_dict, self.include_groups)
        feature_names, xs, ys = reducer.get_x_y_data()
        x_train, y_train, x_test, y_test = split_train_test(xs, ys)

        model = xgboost(x_train, y_train)
        return model, x_test, y_test



def get_global_SuFmTgTf(all_df_dict: dict, user_ids: list = None):
    feature_groups = {
        "static_user": ["Sex", "Body weight", "Height", "Self-identity"],
        "log_macros": ["Energy", "Carbohydrate", "Protein", "Fat"],
        "temporal_cgm": ["cgm_p30", "cgm_p60", "cgm_p120"],
        "temporal_food": ["meal_hour", "time_since_last_meal"]
    }

    user_model_generator = UserRsGenerator(all_df_dict, feature_groups)
    rs, data_set_size = user_model_generator.get_rs(user_ids)
    return rs, data_set_size


def get_local_FmTgTf(all_df_dict: dict, user_id):
    feature_groups = {
        "log_macros": ["Energy", "Carbohydrate", "Protein", "Fat"],
        "temporal_cgm": ["cgm_p30", "cgm_p60", "cgm_p120"],
        "temporal_food": ["meal_hour", "time_since_last_meal"]
    }
    # Do train/test on each user
    user_model_generator = UserRsGenerator(all_df_dict, feature_groups)
    rs, data_set_size = user_model_generator.get_rs([user_id])
    return rs, data_set_size


def get_weighted_ensemble(all_df_dict: dict, local_weight: float):
    global_feature_groups = {
        "static_user": ["Sex", "Body weight", "Height", "Self-identity"],
        "log_macros": ["Energy", "Carbohydrate", "Protein", "Fat"],
        "temporal_cgm": ["cgm_p30", "cgm_p60", "cgm_p120"],
        "temporal_food": ["meal_hour", "time_since_last_meal"]
    }
    local_feature_groups = {
        "log_macros": ["Energy", "Carbohydrate", "Protein", "Fat"],
        "temporal_cgm": ["cgm_p30", "cgm_p60", "cgm_p120"],
        "temporal_food": ["meal_hour", "time_since_last_meal"]
    }
    # Split dataset in 80/20 by each user - FOR TRAIN/TEST
    user_train_test, all_train_dict, all_test_dict = split_train_test_dicts_per_user(all_df_dict)

    # Train a single global model
    global_reducer = FeatureLabelReducer(all_train_dict, global_feature_groups)
    feature_names, xs, ys = global_reducer.get_x_y_data()
    global_model = xgboost(xs, ys)

    results = []

    for user_id, (user_train_dict, user_test_dict) in user_train_test.items():
        local_reducer = FeatureLabelReducer(user_train_dict, local_feature_groups)
        feature_names, xs, ys = local_reducer.get_x_y_data()
        local_model = xgboost(xs, ys)
        x_test_local, y_test = get_x_y_from_features(local_reducer, FeatureLabelReducer(user_test_dict, local_feature_groups))


        # We need a slightly different global x_test set to account for static user.
        x_test_global, _ = get_x_y_from_features(global_reducer, FeatureLabelReducer(user_test_dict, global_feature_groups))

        # Get predictions
        local_predictions = local_model.predict(x_test_local)
        global_predictions = global_model.predict(x_test_global)

        print(f"User ID {user_id}, {len(local_predictions)} {len(global_predictions)}")


        if len(local_predictions) <= 2:
            print(f"User {user_id} has insufficient data, skipping.")
            continue

        r, p = pearsonr(local_predictions, y_test)
        r1, p1 = pearsonr(global_predictions, y_test)

        print(f"Local R: {r}, Global R: {r1}")

        # Ensemble predictions:
        # ensemble_predictions = local_weight * np.array(local_predictions) + (1 - local_weight) * np.array(
        #     global_predictions)

        # r_e, p_e = pearsonr(ensemble_predictions, y_test)

        results.append({
            "user_id": int(user_id),
            "training_size": len(xs),
            "testing_size": len(x_test_local),
            "y_test": list([float(y) for y in y_test]),
            "local_predictions": list([float(l) for l in local_predictions]),
            "global_predictions": list([float(g) for g in global_predictions])
        })
    return results


if __name__ == "__main__":
    all_users = []
    for users in CGMacro_USER_GROUPS.values():
        for u in users:
            all_users.append(u)
    base_file_path = "../data/CGMacros/pickle/"
    df_dict = dict()
    for pkl in ["cgm", "dynamic_user", "log", "static_user"]:
        df_dict[pkl] = load_dataframe(base_file_path + pkl + ".pkl")

    results = get_weighted_ensemble(df_dict, 0)
    print(results)
    with open("data/local_global_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Global mean R all users for SuFmTgTf
    rs, data_set_size = get_global_SuFmTgTf(df_dict)
    print(f"Global SuFmTgTf | All Users, R: {np.mean(rs)}, Dataset Size: {data_set_size}")

    for key, val in CGMacro_USER_GROUPS.items():
        rs, data_set_size = get_global_SuFmTgTf(df_dict, val)
        print(f"Global SuFmTgTf | User Group: {key}, R: {np.mean(rs)}, Dataset Size: {data_set_size}")

    local_r_means: list[dict] = []
    for user in all_users:
        rs, data_set_size = get_local_FmTgTf(df_dict, user)
        local_r_means.append({
            "UserID": user,
            "R_mean": np.mean(rs)
        })
        print(f"Local FmTgTf, UserID: {user}, R: {np.mean(rs)}, Dataset Size: {data_set_size}")

    # Get locals by groups
    local_group_r_means: dict = {}
    for entry in local_r_means:
        selected_key = None
        for key, val in CGMacro_USER_GROUPS:
            if entry["UserID"] in val:
                selected_key = key
                break
        if selected_key in local_group_r_means:
            local_group_r_means[selected_key].append(entry["R_mean"])
        else:
            local_group_r_means[selected_key] = []

    print("\n" + "-" * 20 + "\n")

    print(local_group_r_means)

    for key, val in local_group_r_means:
        print(f"{key}, R Mean: {np.mean(val)}")















