from datetime import timedelta

import pandas as pd


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


def _get_user_filtered_df(df_dict: dict[str, pd.DataFrame], user_ids: list):
    return_dict = dict()
    for key, df in df_dict.items():
        return_dict[key] = df[df["UserID"].isin(user_ids)]
    return return_dict