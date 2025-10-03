"""
UserID=2 Train/Test Split
"""
import json

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from model_user_evaluation.utils import split_train_test_dicts_per_user, one_hot_encode_self_identity
from process_data.main import load_dataframe, FeatureLabelReducer

base_file_path = "../data/CGMacros/pickle/"
df_dict = dict()
for pkl in ["cgm", "dynamic_user", "log", "static_user"]:
    df_dict[pkl] = load_dataframe(base_file_path + pkl + ".pkl")

o = one_hot_encode_self_identity(df_dict["static_user"], 1)
print(o)
# user_train_test, all_train_dict, all_test_dict = split_train_test_dicts_per_user(df_dict)
#
# train, test = user_train_test[2]
#
# reducer = FeatureLabelReducer(test, )
# _, x_test, y_test = reducer.get_x_y_data()

with open('data/local_global_results.json', 'r') as file:
    data = json.load(file)

local_errors, global_errors = [], []
y_test, local_predictions, global_predictions = [], [], []
for user in data:
    local_errors.extend(np.array(user["y_test"]) - np.array(user["local_predictions"]))
    global_errors.extend(np.array(user["y_test"]) - np.array(user["global_predictions"]))
    y_test.extend(user["y_test"])
    local_predictions.extend(user["local_predictions"])
    global_predictions.extend(user["global_predictions"])

r_l, p_l = pearsonr(local_predictions, y_test)
r_g, p_g = pearsonr(global_predictions, y_test)



x = np.arange(len(y_test))  # just indices for each data point

plt.figure(figsize=(10,6))
plt.plot(x, y_test, 'ko-', label="Actual")  # actual values in black
plt.plot(x, local_predictions, 'ro--', label="Local Prediction")  # local in red
plt.plot(x, global_predictions, 'bo--', label="Global Prediction")  # global in blue

plt.xlabel("Data Point Index")
plt.ylabel("Value")
plt.title("Actual vs Local/Global Predictions per Data Point")
plt.legend()
plt.grid(True)
plt.show()

