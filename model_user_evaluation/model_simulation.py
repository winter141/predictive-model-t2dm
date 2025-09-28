

# Train CGMacros Model with data for approx 3 days
# Compute R Score (with 1000 iterations)

# Retrain the model for another 3 days
# Then another 3 days...

# Compare the performance of the updated model

# Test set will consist of new readings from same data source.

# NEED TO FIND A WAY TO PERSONALISE - priorise the user with XGBoost - THIS IS THE PRESSING CHALLENGE
# Lets just try this naive simulation first, and have a look at the results.

# Options for finetuning

# Fit the model, using sample weights - apply higher weighting for that users data. The only problem is time, we would have to train a model for EVERY user.

# Consider the users glucose readings in past responses to foods or similar foods and combine this with the model output...
# This could build on some of the research by using food groups to categorise foods...

#


import numpy as np
import sys, os
import pandas as pd
from matplotlib import pyplot as plt

from model_user_evaluation.utils import _get_start_dates, _get_df_dict_in_range

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from create_model_summary import run_r_iterations
from process_data.main import FeatureLabelReducer, load_dataframe
from datetime import datetime, timedelta



def simulate_model(
        df_dict: dict[str, pd.DataFrame],
        total_days: int,
        day_step: int,
        r_iterations: int = 100
    ):
    user_date_mapping: dict = _get_start_dates(df_dict["cgm"])
    day_lower = 0
    for day_upper in range(day_step, total_days, day_step):
        new_df_dict: dict[str, pd.DataFrame] = _get_df_dict_in_range(df_dict, user_date_mapping, day_upper)
        reducer = FeatureLabelReducer(new_df_dict)
        feature_names, x, y = reducer.get_x_y_data()
        rs = run_r_iterations(x, y, r_iterations)
        print(day_upper, np.mean(rs))
        
    pass


if __name__ == "__main__":
    x = np.load("../data/CGMacros/feature_label/x.npy", allow_pickle=True)
    y = np.load("../data/CGMacros/feature_label/y.npy", allow_pickle=True)
    feature_names = np.load("../data/CGMacros/feature_label/feature_names.npy", allow_pickle=True)

    base_file_path = "../data/CGMacros/pickle/"
    df_dict = dict()
    for pkl in ["cgm", "dynamic_user", "log", "static_user"]:
        df_dict[pkl] = load_dataframe(base_file_path + pkl + ".pkl")

    # for key, val in df_dict.items():
    #     print(key, val.columns)
    # ----------------------------------- #
    # reducer = FeatureLabelReducer(df_dict)
    # feature_names, x, y = reducer.get_x_y_data()

    # simulate_model(df_dict, 12, 1)

    # xs = range(1, 12)
    # ys = [0.1486298997946426, 0.43273415190440034, 0.5173040330362675, 0.5673804336726491, 0.5570900076872001, 0.5523155769318593,
    #       0.6109467254563611, 0.6135222823065054, 0.6445637737527982, 0.635261622485375, 0.6372489209632133]
    # plt.plot(xs, ys, 'o--')
    # plt.show()
