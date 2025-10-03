from datetime import timedelta

import numpy as np
import pandas as pd

from models import split_train_test
from process_data.main import FeatureLabelReducer, load_dataframe


def _get_start_dates(df: pd.DataFrame):
    users = df["UserID"].unique()
    user_date_mapping = dict()
    for user in users:
        user_date_mapping[user] = min(pd.to_datetime(df[df["UserID"] == user]["Timestamp"]))
    return user_date_mapping

def _get_df_dict_in_range(df_dict, user_date_mapping, day_upper_delta, day_lower_delta=0):
    """
    range [lower, upper)
    @params
        - day_delta: numbers of days PLUS start date.
    """
    new_df_dict: dict[str, pd.DataFrame] = dict()
    if "static_user" in df_dict:
        new_df_dict["static_user"] = df_dict["static_user"]
    for user, start_date in user_date_mapping.items():
        start_collection_date = start_date + timedelta(days=day_lower_delta)
        end_collection_date = start_date + timedelta(days=day_upper_delta)

        for key, df in df_dict.items():
            if "Timestamp" not in df.columns:
                continue

            f_df = df[(df["UserID"] == user) & (df["Timestamp"] >= start_collection_date) & (df["Timestamp"] < end_collection_date)]

            if key in new_df_dict:
                new_df_dict[key] = pd.concat([new_df_dict[key], f_df], ignore_index=True)
            else:
                new_df_dict[key] = f_df.copy()
    return new_df_dict

def split_train_test_dicts_per_user(df_dict: dict[str, pd.DataFrame], train_ratio: float = 0.8):
    """
    Splits each user's data into train and test sets (per-user, time-based split).

    The cutoff is chosen based on each user's earliest and latest timestamp.

    Returns:
        (
          user_train_test,  # dict[user_id -> (train_dict, test_dict)]
          all_train_dict,   # dict[str, pd.DataFrame] combined across users
          all_test_dict     # dict[str, pd.DataFrame] combined across users
        )
    """

    user_train_test = {}
    all_train_dict = {"static_user": df_dict["static_user"], "cgm": df_dict["cgm"], "dynamic_user": df_dict["dynamic_user"]}
    all_test_dict = {"static_user": df_dict["static_user"], "cgm": df_dict["cgm"], "dynamic_user": df_dict["dynamic_user"]}

    log_df = df_dict["log"]
    users = log_df["UserID"].unique()

    for user_id in users:
        # Only do cutoff time based on logs
        # Maintain the other full dataframes.
        user_logs: pd.DataFrame = log_df[log_df["UserID"] == user_id].copy()
        n = len(user_logs)

        # Shuffle indices reproducibly
        permutation = np.random.permutation(n)
        user_logs = user_logs.iloc[permutation].reset_index(drop=True)

        # Cutoff
        i = int(n * train_ratio)
        train_df = user_logs.iloc[:i].copy()
        test_df = user_logs.iloc[i:].copy()

        train_dict = {"static_user": df_dict["static_user"], "cgm": df_dict["cgm"][df_dict["cgm"]["UserID"] == user_id],
                          "dynamic_user": df_dict["dynamic_user"][df_dict["dynamic_user"]["UserID"] == user_id]}
        test_dict = {"static_user": df_dict["static_user"], "cgm": df_dict["cgm"][df_dict["cgm"]["UserID"] == user_id],
                         "dynamic_user": df_dict["dynamic_user"][df_dict["dynamic_user"]["UserID"] == user_id]}

        train_dict["log"] = train_df
        test_dict["log"] = test_df

        # add into global dicts
        if "log" in all_train_dict:
            all_train_dict["log"] = pd.concat([all_train_dict["log"], train_df], ignore_index=True)
            all_test_dict["log"] = pd.concat([all_test_dict["log"], test_df], ignore_index=True)
        else:
            all_train_dict["log"] = train_df.copy()
            all_test_dict["log"] = test_df.copy()

        user_train_test[user_id] = (train_dict, test_dict)

    return user_train_test, all_train_dict, all_test_dict



def _get_user_filtered_df(df_dict: dict[str, pd.DataFrame], user_ids: list):
    return_dict = dict()
    for key, df in df_dict.items():
        return_dict[key] = df[df["UserID"].isin(user_ids)]
    return return_dict

def one_hot_encode_self_identity(static_user_df, user_id):
    self_identity_one_hot_encoding = pd.get_dummies(static_user_df[["UserID", "Self-identity"]])
    row = self_identity_one_hot_encoding[self_identity_one_hot_encoding["UserID"] == user_id]
    return row.drop(columns=["UserID"]).iloc[0].tolist()


if __name__ == "__main__":
    base_file_path = "../data/CGMacros/pickle/"
    df_dict = dict()
    for pkl in ["cgm", "dynamic_user", "log", "static_user"]:
        df_dict[pkl] = load_dataframe(base_file_path + pkl + ".pkl")

    user_train_test, all_train_dict, all_test_dict = split_train_test_dicts_per_user(df_dict)
    for key, val in user_train_test.items():
        train, test = val
        print(key, len(train["log"]), len(test["log"]))
