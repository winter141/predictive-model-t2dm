import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

CGMacro_USER_GROUPS = {
    "healthy": [1,  2, 4, 6, 15, 17, 18, 19, 21, 27, 31, 32, 33, 34, 48],
    "prediabetes": [7, 8,  9, 10, 11, 13, 16, 20, 22, 23, 26, 29, 41, 43, 44, 45],
    "t2dm": [3, 5, 12, 14, 28, 30, 35, 36, 38, 39, 42, 46, 47, 49]
}

# Open and read the JSON file
with open('data/data.json', 'r') as file:
    data = json.load(file)  # Parse JSON into a Python dictionary


all_dict = dict()
for lw, entry in data.items():
    healthy, prediabetes, t2dm = [], [], []
    for user_dict in entry:
        if len(user_dict["predictions"]) > 2:
            r, p = pearsonr(user_dict["predictions"], user_dict["y_test"])
        else:
            continue
        u_id = user_dict["user_id"]
        if u_id in CGMacro_USER_GROUPS["healthy"]:
            healthy.append(r)
        elif u_id in CGMacro_USER_GROUPS["prediabetes"]:
            prediabetes.append(r)
        elif u_id in CGMacro_USER_GROUPS["t2dm"]:
            t2dm.append(r)
    all_dict[lw] = {
        "healthy": healthy,
        "prediabetes": prediabetes,
        "t2dm": t2dm
    }


for lw_key in [0.03, 0.14, 0.21, 0.28, 0.34, 0.41, 0.48, 0.55, 0.62, 0.69, 0.76, 0.83, 0.9, 0.97]:
    del all_dict[f"Local Weight {lw_key:.2f}"]

def results_dict_to_df(results_dict):
    records = []
    for weight, groups in results_dict.items():
        for group, r_list in groups.items():
            for r in r_list:
                records.append({
                    "Weight": weight.replace("Local Weight ", ""),  # strip prefix
                    "Group": group,
                    "R": float(r)
                })
    return pd.DataFrame(records)

# Convert dict to DataFrame
df = results_dict_to_df(all_dict)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# assume df exists with columns: "Weight", "Group", "R"

for show_box_plot in [True, False]:
    df["Weight"] = df["Weight"].astype(float)
    sorted_weights = sorted(df["Weight"].unique())

    groups = ["healthy", "prediabetes", "t2dm"]
    colors = sns.color_palette("Set2", n_colors=len(groups))
    palette = dict(zip(groups, colors))

    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    # --- Boxplots ---
    if show_box_plot:
        sns.boxplot(data=df, x="Weight", y="R", hue="Group",
                    order=sorted_weights, palette=palette, ax=ax)

        # --- Stripplot (optional points) ---
        sns.stripplot(data=df, x="Weight", y="R", hue="Group",
                      order=sorted_weights, palette=palette,
                      dodge=True, jitter=True, alpha=0.25, size=4, ax=ax, linewidth=0,
                      legend=False)

    # --- Group means ---
    means = df.groupby(["Weight", "Group"])["R"].mean().reset_index()
    pivot = means.pivot(index="Weight", columns="Group", values="R").reindex(sorted_weights)

    # --- Overall mean ---
    overall_mean = df.groupby("Weight")["R"].mean().reindex(sorted_weights)

    x_positions = list(range(len(sorted_weights)))

    line_handles = []
    for grp in groups:
        if grp in pivot.columns:
            y = pivot[grp].values
            (ln,) = ax.plot(x_positions, y, marker="o", linewidth=2.5,
                            color=palette[grp], label=f"{grp} mean")
            line_handles.append(ln)

            # --- Show values above each point ---
            if not show_box_plot:
                for xi, yi in zip(x_positions, y):
                    ax.text(xi, yi + 0.01, f"{yi:.2f}", ha='center', va='bottom', fontsize=9, color=palette[grp])

    # --- Overall line (black dashed) ---
    (overall_ln,) = ax.plot(x_positions, overall_mean.values, marker="o",
                            color="black", linestyle="--", linewidth=2.5, label="Overall mean")
    line_handles.append(overall_ln)

    # Show values above overall line points
    if not show_box_plot:
        for xi, yi in zip(x_positions, overall_mean.values):
            ax.text(xi, yi + 0.01, f"{yi:.2f}", ha='center', va='bottom', fontsize=9, color='black')

    # --- Axis labels ---
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{w:.2f}" for w in sorted_weights], rotation=45)
    ax.set_xlabel("Personalization Weight")
    ax.set_ylabel("R Score")
    ax.set_title("R Distribution by Group and Personalization Weight")

    # --- Legend ---
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(handles=line_handles, title="Mean Lines", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(f"figures/user_group_we_model_{"boxplot" if show_box_plot else "line"}", bbox_inches='tight')
    plt.close()
    # plt.show()


