import matplotlib.pyplot as plt
import json
import numpy as np

def open_data():
    # Open and load JSON file
    with open('data.json', 'r') as file:
        return json.load(file)

def plt_cartesian_product(data: list[dict]):
    features = ["Su", "Fm", "Tg", "Tf", "Ft", "Ui"]

    data = sorted(data, key=lambda x: -x["mean_R"])

    fig, ax = plt.subplots(figsize=(6, 10))

    for i, row in enumerate(data):
        for j, feature in enumerate(features):
            if feature in row["nickname"]:
                ax.plot(j, i, 'o', markersize=5, markerfacecolor='none', markeredgecolor='green', markeredgewidth=2)
            else:
                ax.plot(j, i, 'o', markersize=5, markerfacecolor='white', markeredgecolor='gray', alpha=0.5)
        # Add R score at the end of the row
        ax.text(len(features) + 0.2, i, f"{row['mean_R']:.3f}", va='center')
        ax.text(len(features) + 0.5, i, f"{row['std_R']:.3f}", va='center')


    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([row['nickname'] for row in data])
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.invert_yaxis()  # highest R at top if desired
    ax.set_xlim(-0.5, len(features) + 1)
    ax.set_xlabel("Features")
    ax.set_title("Feature Inclusion vs R Score")
    plt.show()


if __name__ == "__main__":
    data = open_data()
    d: list[dict]= []
    for model_results in data:
        rs = model_results["result"]
        d.append({
            "nickname": model_results["nickname"],
            "mean_R": np.mean(rs),
            "std_R": np.std(rs)
        })
    plt_cartesian_product(d)
