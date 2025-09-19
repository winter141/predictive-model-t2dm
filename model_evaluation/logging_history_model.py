"""
This creates a model unique to each user, that takes into account their previous logs.

It can be combined with the XGBoost model for a more personalised prediction
"""
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from models import split_train_test, xgboost, get_new_preds
from process_data.main import load_dataframe, FeatureLabelReducer


class LoggingHistoryModel:
    def __init__(self, user_id, input_df_dict):
        user_df_dict = dict()
        for key, df in input_df_dict.items():
            user_df_dict[key] = df[df["UserID"] == user_id]
        self.df_dict = user_df_dict

    def train_mini_model(self):

        # Find logs
        # Categorise into food groups...
        # Find iAUC of logs
        # Step 1. Find logs just for the user

        feature_names, xs, ys = FeatureLabelReducer(self.df_dict).get_x_y_data()
        xs = xs[:, 3:]
        print(len(xs))

        print(xs[0])

        print(feature_names)

        r_iterations = 100

        rs = []
        for i in range(r_iterations):
            preds, y_test = get_new_preds(xs, ys)
            r, p = pearsonr(preds, y_test)
            rs.append(r)
        print(f"\n\n**R ({r_iterations} iterations):** Mean {np.mean(rs):.3f}, Std {np.std(rs):.3f}")
        # Note train/test proportion
        # x_train, y_train, x_test, y_test = split_train_test(xs, ys, proportion=0.8)

    def plt(self):
        feature_names, xs, ys = FeatureLabelReducer(self.df_dict).get_x_y_data()
        xs = xs[:, 3:]

        # Prepare DataFrame
        data = pd.DataFrame(xs, columns=feature_names[3:])
        data['iAUC'] = ys
        # If you have a user_id array, replace range(len(xs)) with it
        data['user_id'] = range(len(xs))

        sns.set(style="whitegrid")

        # --- Plot 1: Variability per feature ---
        plt.figure(figsize=(15, 5))
        for i, feature in enumerate(feature_names[3:]):
            plt.subplot(1, len(feature_names[3:]), i + 1)
            sns.scatterplot(x=data[feature], y=data['iAUC'])
            plt.xlabel(feature)
            if i == 0:
                plt.ylabel('iAUC')
            plt.title(f'{feature} vs iAUC')

        plt.suptitle('iAUC Variability for single user', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # --- Plot 2: Individual logs per feature vs iAUC ---
        # Prepare DataFrame
        data = pd.DataFrame(xs, columns=feature_names[3:])
        data['iAUC'] = ys
        data['meal_hour'] = data['meal_hour'].astype(int)  # for labels

        # Colors for energy components
        colors = ['skyblue', 'orange', 'green']  # carbs, protein, fat
        energy_labels = ['Carbohydrate', 'Protein', 'Fat']

        logs_per_fig = 16
        num_figs = int(np.ceil(len(data) / logs_per_fig))

        for fig_idx in range(num_figs):
            start_idx = fig_idx * logs_per_fig
            end_idx = min((fig_idx + 1) * logs_per_fig, len(data))
            subset = data.iloc[start_idx:end_idx]

            fig, axes = plt.subplots(4, 4, figsize=(16, 12))
            axes = axes.flatten()

            for ax_idx, (_, row) in enumerate(subset.iterrows()):
                # Energy components
                energy_vals = [row['Carbohydrate'], row['Protein'], row['Fat']]
                # Stacked energy bar
                axes[ax_idx].bar(0, energy_vals[0], color=colors[0])
                axes[ax_idx].bar(0, energy_vals[1], bottom=energy_vals[0], color=colors[1])
                axes[ax_idx].bar(0, energy_vals[2], bottom=sum(energy_vals[:2]), color=colors[2])

                # iAUC bar next to it
                axes[ax_idx].bar(0.5, row['iAUC'], color='gray', alpha=0.7, label='iAUC')

                axes[ax_idx].set_xticks([0, 0.5])
                axes[ax_idx].set_xticklabels([f'Meal {row["meal_hour"]}h', 'iAUC'])
                axes[ax_idx].set_ylabel('Energy / iAUC')
                axes[ax_idx].set_title(f'Log {start_idx + ax_idx + 1}', fontsize=10)

            # Remove empty subplots
            for j in range(len(subset), 16):
                fig.delaxes(axes[j])

            # Single legend for energy components + iAUC
            handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors] + [
                plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.7)]
            labels = energy_labels + ['iAUC']
            fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=10)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
    def train(self):
        pass

    def predict(self):
        pass


if __name__ == "__main__":
    base_file_path = "../data/CGMacros/pickle/"
    df_dict = dict()
    for pkl in ["cgm", "dynamic_user", "log", "static_user"]:
        df_dict[pkl] = load_dataframe(base_file_path + pkl + ".pkl")

    user_id = 1
    user_model = LoggingHistoryModel(user_id, df_dict)
    user_model.train_mini_model()