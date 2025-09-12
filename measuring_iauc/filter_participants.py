"""
Filter participants based on Fasting GLU - PDL

The distribution should be:
- 15 healthy
- 16 pre-diabetes
- 14 T2DM
This was obtained by the thresholds for HbA1C consistent in https://diabetes.org/about-diabetes/diagnosis
"""
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from process_data.main import FeatureLabelReducer


def load_dataframe(filepath) -> pd.DataFrame:
    return pd.read_pickle(filepath)


def filter_users(static_user: pd.DataFrame) -> dict[str, pd.DataFrame]:
    healthy_df = static_user[static_user['A1c PDL (Lab)'] < 5.7]
    prediabetes_df = static_user[(static_user['A1c PDL (Lab)'] >= 5.7) & 
                                 (static_user['A1c PDL (Lab)'] <= 6.4)]
    t2dm_df = static_user[static_user['A1c PDL (Lab)'] > 6.4]

    return {'healthy': healthy_df, 'prediabetes': prediabetes_df, 't2dm': t2dm_df}


if __name__ == "__main__":
    base_file_path = "../data/CGMacros/pickle/"
    df_dict = dict()
    for pkl in ["cgm", "dynamic_user", "log", "static_user"]:
        df_dict[pkl] = load_dataframe(base_file_path + pkl + ".pkl")

    filtered_dfs: dict[str, pd.DataFrame] = filter_users(df_dict['static_user'])

    for key, df in filtered_dfs.items():
        print(key)
        print(df["UserID"].to_numpy())
        print()

    # base_cgmacros_path = "../data/CGMacros/feature_label/"
    # for name, df in filtered_dfs.items():
    #     df_dict['static_user'] = df

    #     reducer = FeatureLabelReducer(df_dict)
    #     feature_names, x, y = reducer.get_x_y_data()

    #     np.save(f"{base_cgmacros_path}{name}/feature_names.npy", feature_names)
    #     np.save(f"{base_cgmacros_path}{name}/x.npy", x)
    #     np.save(f"{base_cgmacros_path}{name}/y.npy", y)

