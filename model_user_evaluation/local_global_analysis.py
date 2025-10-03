"""
Analysis of Local/Global Results file.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

CGMacro_USER_GROUPS = {
    "healthy": [1,  2, 4, 6, 15, 17, 18, 19, 21, 27, 31, 32, 33, 34, 48],
    "prediabetes": [7, 8,  9, 10, 11, 13, 16, 20, 22, 23, 26, 29, 41, 43, 44, 45],
    "t2dm": [3, 5, 12, 14, 28, 30, 35, 36, 38, 39, 42, 46, 47, 49]
}

class LocalGlobalAnalysis:
    def __init__(self, local_global_arr: list[dict]):
        self.local_global_arr = local_global_arr

        self.local_errors, self.global_errors = [], []
        self.y_test, self.local_predictions, self.global_predictions = [], [], []
        for user in self.local_global_arr:
            self.local_errors.extend(np.array(user["y_test"]) - np.array(user["local_predictions"]))
            self.global_errors.extend(np.array(user["y_test"]) - np.array(user["global_predictions"]))
            self.y_test.extend(user["y_test"])
            self.local_predictions.extend(user["local_predictions"])
            self.global_predictions.extend(user["global_predictions"])

        self.r_l, self.p_l = pearsonr(self.local_predictions, self.y_test)
        self.r_g, self.p_g = pearsonr(self.global_predictions, self.y_test)

    def error_distribution_by_model(self):
        plt.hist(self.local_errors, bins=50, alpha=0.5, label=f"Local, R = {self.r_l}")
        plt.hist(self.global_errors, bins=50, alpha=0.5, label=f"Global, R = {self.r_g}")
        plt.legend()
        plt.savefig("error_distribution_by_model")
        plt.close()

    def full_actual_expected(self):
        """
        Actual Expected for local, and for global, with pearson R in legend label.
        """
        plt.plot(self.y_test, self.local_predictions, 'o', label=f"Local, R = {self.r_l}")
        plt.plot(self.y_test, self.global_predictions, 'o', label=f"Global, R = {self.r_g}")

        plt.xlabel("Actual")
        plt.ylabel("Prediction")

        min_val = min(min(self.y_test), min(self.local_predictions), min(self.global_predictions))
        max_val = max(max(self.y_test), max(self.local_predictions), max(self.global_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label="y=x", alpha=0.8)

        plt.legend()
        plt.savefig("full_actual_expected")
        plt.close()


    def user_level_error_scatter(self):
        """
        x axis: user's global error, y axis: user local error.
        Each point is 1 user.
        If points fall below the diagonal line, local is better; above, global is better.
        """
        plt.plot(self.global_errors, self.local_errors, 'o')
        plt.xlabel("Global Error")
        plt.ylabel("Local Error")
        min_val = min(min(self.global_errors), min(self.local_errors))
        max_val = max(max(self.global_errors), max(self.local_errors))
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label="y=x", alpha=0.8)
        plt.title("Raw User Level Error")
        plt.savefig("user_level_error_scatter")
        plt.close()

    def user_level_error_abs_scatter(self):
        """
        x axis: user's global error, y axis: user local error.
        Each point is 1 user.
        If points fall below the diagonal line, local is better; above, global is better.
        """
        g_e = np.abs(self.global_errors)
        l_e = np.abs(self.local_errors)
        plt.plot(g_e, l_e, 'o')
        plt.xlabel("Global Error")
        plt.ylabel("Local Error")
        min_val = min(min(g_e), min(l_e))
        max_val = max(max(g_e), max(l_e))
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label="y=x", alpha=0.8)
        plt.title("Absolute User Level Error")
        plt.savefig("user_level_error_abs_scatter")
        plt.close()


    # ---- CHATGPT ANALYSIS ----- #
    def boxplot_user_errors(self):
        """
        Boxplot of per-user absolute errors for local vs global.
        """
        user_local_mae = []
        user_global_mae = []
        for user in self.local_global_arr:
            user_local_mae.append(np.mean(np.abs(np.array(user["y_test"]) - np.array(user["local_predictions"]))))
            user_global_mae.append(np.mean(np.abs(np.array(user["y_test"]) - np.array(user["global_predictions"]))))

        plt.boxplot([user_local_mae, user_global_mae], labels=["Local", "Global"])
        plt.ylabel("Mean Absolute Error per User")
        plt.title("Distribution of User-Level Errors")
        plt.savefig("boxplot_user_errors")
        plt.close()

    def error_cdf(self):
        """
        CDF of absolute errors for local vs global.
        """
        l_e = np.sort(np.abs(self.local_errors))
        g_e = np.sort(np.abs(self.global_errors))
        l_cdf = np.arange(1, len(l_e)+1) / len(l_e)
        g_cdf = np.arange(1, len(g_e)+1) / len(g_e)

        plt.plot(l_e, l_cdf, label="Local")
        plt.plot(g_e, g_cdf, label="Global")
        plt.xlabel("Absolute Error")
        plt.ylabel("CDF")
        plt.title("Error CDF (Local vs Global)")
        plt.legend()
        plt.savefig("error_cdf")
        plt.close()

    def heatmap_user_differences_by_group(self, user_groups):
        """
        Heatmap of MAE differences per user, grouped by cohort.
        Positive = Local better, Negative = Global better.
        """
        user_ids = []
        diffs = []
        group_labels = []

        # Compute per-user differences
        for user in self.local_global_arr:
            uid = user["user_id"]
            y_test = np.array(user["y_test"])
            local_mae = np.mean(np.abs(y_test - np.array(user["local_predictions"])))
            global_mae = np.mean(np.abs(y_test - np.array(user["global_predictions"])))
            diff = global_mae - local_mae

            user_ids.append(uid)
            diffs.append(diff)

            # Find which group the user belongs to
            for group_name, ids in user_groups.items():
                if uid in ids:
                    group_labels.append(group_name)
                    break
            else:
                group_labels.append("unknown")

        # Sort users by group for nicer visualization
        sorted_idx = np.argsort(group_labels)
        user_ids = np.array(user_ids)[sorted_idx]
        diffs = np.array(diffs)[sorted_idx]
        group_labels = np.array(group_labels)[sorted_idx]

        plt.figure(figsize=(12, 3))
        plt.imshow([diffs], aspect="auto", cmap="bwr", interpolation="nearest")
        plt.colorbar(label="Global MAE - Local MAE")

        # Compute tick positions at group centers
        xticks, xticklabels = [], []
        boundaries = []

        for grp in np.unique(group_labels):
            indices = np.where(group_labels == grp)[0]
            if len(indices) > 0:
                center = (indices[0] + indices[-1]) / 2
                xticks.append(center)
                xticklabels.append(grp)
                boundaries.append(indices[-1] + 0.5)

        # Add vertical lines between groups
        for b in boundaries[:-1]:  # exclude last since it's the end of axis
            plt.axvline(x=b, color="black", linestyle="--", alpha=0.7)

        plt.xticks(xticks, xticklabels, rotation=0)
        plt.yticks([])
        plt.title("Per-User Error Differences Grouped by Cohort")
        plt.tight_layout()
        plt.savefig("heatmap_user_differences_by_group")
        plt.close()

    def heatmap_user_rscores_by_group(self, user_groups):
        """
        Two heatmaps of per-user Pearson R scores, grouped by cohort.
        One for Local predictions, one for Global predictions.
        """
        user_ids = []
        local_rs, global_rs = [], []
        group_labels = []

        # Compute per-user R scores
        for user in self.local_global_arr:
            uid = user["user_id"]
            y_test = np.array(user["y_test"])
            local_preds = np.array(user["local_predictions"])
            global_preds = np.array(user["global_predictions"])

            # Handle edge cases (constant arrays → NaN R)
            try:
                r_local, _ = pearsonr(local_preds, y_test)
            except Exception:
                r_local = np.nan
            try:
                r_global, _ = pearsonr(global_preds, y_test)
            except Exception:
                r_global = np.nan

            user_ids.append(uid)
            local_rs.append(r_local)
            global_rs.append(r_global)

            # Find which group the user belongs to
            for group_name, ids in user_groups.items():
                if uid in ids:
                    group_labels.append(group_name)
                    break
            else:
                group_labels.append("unknown")

        # Sort users by group for consistent visualization
        sorted_idx = np.argsort(group_labels)
        user_ids = np.array(user_ids)[sorted_idx]
        local_rs = np.array(local_rs)[sorted_idx]
        global_rs = np.array(global_rs)[sorted_idx]
        group_labels = np.array(group_labels)[sorted_idx]

        # Helper to plot one heatmap
        def plot_heatmap(values, title):
            plt.imshow([values], aspect="auto", cmap="viridis", vmin=-1, vmax=1, interpolation="nearest")
            plt.colorbar(label="Pearson R")

            # Compute tick positions at group centers
            xticks, xticklabels = [], []
            boundaries = []
            for grp in np.unique(group_labels):
                indices = np.where(group_labels == grp)[0]
                if len(indices) > 0:
                    center = (indices[0] + indices[-1]) / 2
                    xticks.append(center)
                    xticklabels.append(grp)
                    boundaries.append(indices[-1] + 0.5)

            plt.xticks(xticks, xticklabels, rotation=0)
            plt.yticks([])
            plt.title(title)

            # Add vertical lines between groups
            for b in boundaries[:-1]:
                plt.axvline(x=b, color="black", linestyle="--", alpha=0.7)

        # Plot side-by-side
        plt.figure(figsize=(14, 4))

        plt.subplot(1, 2, 1)
        plot_heatmap(local_rs, "Local Model Pearson R")

        plt.subplot(1, 2, 2)
        plot_heatmap(global_rs, "Global Model Pearson R")

        plt.tight_layout()
        plt.savefig("heatmap_rscores_by_group")
        plt.close()

    def weighted_ensemble_scatter_by_group(self, user_groups: dict[str, list], local_weights):
        group_dict: dict = {}
        for user in self.local_global_arr:
            uid = user["user_id"]
            for label, users in user_groups.items():
                if uid in users:
                    if label in group_dict:
                        group_dict[label].append(user)
                    else:
                        group_dict[label] = [user]
                    break

        xs, ys = [], []
        for lw in local_weights:
            ensemble = lambda lp, gp: lp * lw + gp * (1-lw)

            prediction_actual: dict = {}
            for user_group_label, users in group_dict.items():
                prediction_actual[user_group_label] = {
                    "predictions": [],
                    "actual": []
                }
                for user in users:
                    prediction_actual[user_group_label]["predictions"].extend(ensemble(np.array(user["local_predictions"]), np.array(user["global_predictions"])))
                    prediction_actual[user_group_label]["actual"].extend(user["y_test"])

            user_group_rs: dict = {}
            for user_group_label, pa_dict in prediction_actual.items():
                # ---
                r, _ = pearsonr(pa_dict["predictions"], pa_dict["actual"])
                user_group_rs[user_group_label] = r
                print(user_group_label, "LW", lw, "R:", r)

            xs.append(lw)
            ys.append(user_group_rs)

        healthy = [d['healthy'] for d in ys]
        t2dm = [d['t2dm'] for d in ys]
        prediabetes = [d['prediabetes'] for d in ys]

        # plot
        plt.figure(figsize=(8, 5))
        plt.plot(xs, healthy, marker='o', label='Healthy')
        plt.plot(xs, t2dm, marker='o', label='T2DM')
        plt.plot(xs, prediabetes, marker='o', label='Prediabetes')

        plt.xlabel('Local Weight')
        plt.ylabel('R')
        plt.title('R Score change with Local Prediction Weight')
        plt.legend()
        plt.grid(True)
        plt.savefig("weighted_ensemble_scatter_by_group")
        plt.close()

    def _save_fig(self, title: str):
        plt.savefig(f"/figures/new/{title}.png")




if __name__ == "__main__":
    with open('data/local_global_results.json', 'r') as file:
        data = json.load(file)
    analyser = LocalGlobalAnalysis(data)

    analyser.error_distribution_by_model()
    analyser.full_actual_expected()
    analyser.user_level_error_scatter()
    analyser.user_level_error_abs_scatter()

    analyser.boxplot_user_errors()
    analyser.error_cdf()
    analyser.heatmap_user_differences_by_group(CGMacro_USER_GROUPS)
    analyser.heatmap_user_rscores_by_group(CGMacro_USER_GROUPS)
    analyser.weighted_ensemble_scatter_by_group(CGMacro_USER_GROUPS, np.linspace(0, 1, 15))




