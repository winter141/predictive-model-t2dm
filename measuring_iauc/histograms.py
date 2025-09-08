"""
Plot the iAUC histograms to get an idea on how well the actual values match with research studies.

RANGES FROM RESEARCH:


"""
import numpy as np
import matplotlib.pyplot as plt



def hist(data: dict[str, list], overlay=True):
    if overlay:
        for label, ys in data.items():
            plt.hist(ys, alpha=0.5, label=label + stats(ys))
        plt.ylabel("Frequency")
        plt.xlabel("iAUC (mmol/L")
        plt.legend()
        plt.title("iAUC 2 hours PPGR")
        plt.show()
    else:
        for label, ys in data.items():
            plt.hist(ys)
            plt.ylabel("Frequency")
            plt.xlabel("iAUC (mmol/L")
            plt.title(f"iAUC 2 hours PPGR, {label + stats(ys)}")
            plt.show()

def box_whiskers(data: dict[str, list], overlay: bool = True):
    if overlay:
        # Plot all boxplots together with labels
        plt.boxplot([ys for ys in data.values()], labels=list(data.keys()))
        plt.ylabel("Frequency")
        plt.xlabel("iAUC (mmol/L)")
        plt.title("iAUC 2 hours PPGR")
        for i, (label, ys) in enumerate(data.items(), start=1):
            s = stats(ys)
            plt.text(i, max(ys) + (0.05 * max(ys)), s,
                     ha='center', va='bottom', fontsize=8)
        plt.show()
    else:
        # Plot one boxplot per figure
        for label, ys in data.items():
            plt.boxplot(ys)
            plt.ylabel("Frequency")
            plt.xlabel("iAUC (mmol/L)")
            plt.title(f"iAUC 2 hours PPGR - {label + stats(ys)}")
            plt.show()


def stats(y: list) -> str:
    arr = np.array(y)
    mean = arr.mean()
    p25 = np.percentile(arr, 25)
    p50 = np.percentile(arr, 50)  # same as median
    p75 = np.percentile(arr, 75)
    std = arr.std()

    return (f": Mean: {mean:.2f}, "
            f"25th: {p25:.2f}, 50th: {p50:.2f}, 75th: {p75:.2f}, "
            f"Std: {std:.2f}")



def categorize_iAUC(values):
    """Convert raw iAUC values into counts per category: LOW, MEDIUM, HIGH"""
    low = np.sum(values < 183)
    medium = np.sum((values >= 183) & (values <= 375))
    high = np.sum(values > 375)
    return [low, medium, high]

def bar_groups_by_iAUC(data: dict[str, np.ndarray], iAUC_bins: list[str]):
    """
    data: dict[str, np.ndarray] - raw iAUC arrays for each health group
    iAUC_bins: list[str] - ['LOW', 'MEDIUM', 'HIGH']
    """
    health_groups = list(data.keys())
    n_groups = len(health_groups)
    x = np.arange(len(iAUC_bins))  # positions for LOW, MEDIUM, HIGH
    width = 0.2  # width of each bar

    # Preprocess counts per category
    counts_dict = {group: categorize_iAUC(values) for group, values in data.items()}

    fig, ax = plt.subplots(figsize=(8,5))

    # Plot each health group
    for i, group in enumerate(health_groups):
        bars = ax.bar(x + i*width, counts_dict[group], width, label=group)
        # Add numbers on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,  # 1 unit above bar
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x + width*(n_groups-1)/2)
    ax.set_xticklabels(iAUC_bins)
    ax.set_ylabel("Number of samples")
    ax.set_title("Distribution of iAUC Categories by Health Group")
    ax.legend()
    plt.show()


if __name__ == '__main__':
    # Load data in

    healthy_y = np.load("../data/CGMacros/feature_label/healthy/y.npy", allow_pickle=True)
    prediabetes_y = np.load("../data/CGMacros/feature_label/prediabetes/y.npy", allow_pickle=True)
    t2dm_y = np.load("../data/CGMacros/feature_label/t2dm/y.npy", allow_pickle=True)

    data_dict = {"healthy": healthy_y, "prediabetes": prediabetes_y, "t2dm": t2dm_y}

    # hist(data_dict, overlay=True)
    # box_whiskers(data_dict, overlay=True)
    bar_groups_by_iAUC(data_dict, ["LOW (iAUC < 183mmol/L)", "MEDIUM (iAUC [183, 375]mmol/L)", "HIGH (iAUC > 375mmol/L)"])