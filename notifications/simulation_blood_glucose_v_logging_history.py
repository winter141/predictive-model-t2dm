"""
Simulate with different datasets.

For now lets just do the CGMacros and UC_HT_T1DM.

Note the UC_HT_T1DM only has carbs as the food logs.

UC_HT_T1DM is about five days

CGMacros is about 10 days

So we can simulate for 3 days.

Not the most efficient, but it does not matter :)
"""
import json

from matplotlib import cm, pyplot as plt
import pandas as pd
import sys
from pathlib import Path
from datetime import timedelta
from enum import Enum
import re
import os
from sklearn.metrics import ConfusionMatrixDisplay
from configurations import CRON_MIN_SCHEDULING, MAX_NOTIFICATIONS, SEGMENTED_SCHEDULES, MIN_MINUTES_APART
from live_notification_blood_glucose import notify_max_min_ratio, notify_rate_of_change, notify_cusum, \
    notify_sustained_rise, notify_predictive_residual, notify_hybrid
from calculate_schedule_times import ScheduleCalculatorBase, WeightedAverageSegmentedCalculator
from typing import Optional
import numpy as np
import math
sys.path.append(str(Path(__file__).resolve().parent))


# Done in measuring_iauc/filter_particpants. It is by A1c which is the method identified in the CGMacros documentation.
CGMacro_USER_GROUPS = {
    "healthy": [1,  2, 4, 6, 15, 17, 18, 19, 21, 27, 31, 32, 33, 34, 48],
    "prediabetes": [7, 8,  9, 10, 11, 13, 16, 20, 22, 23, 26, 29, 41, 43, 44, 45],
    "t2dm": [3, 5, 12, 14, 28, 30, 35, 36, 38, 39, 42, 46, 47, 49]
}

class LogDifferenceMethod(Enum):
    WEIGHTED_AVG = "Weighted Average"  # In segment
    LARGEST_LOG = "Largest Log"    # In segment


def compute_logging_history_difference(log_df: pd.DataFrame, days: int, difference_methods: list[LogDifferenceMethod], user_ids = None) -> list[dict]:
    """
    TODO: For logs, lets just look at the Carbohydrates instead of the energy, as UC_HT_T1DM only does Carbs.

    Example output:
    [
        {
            'day': 0,
            'logging_history_differences': [[None, None, None], [None, None, None], ...]
        }, 
        {
            'day': 1,
            'logging_history_differences': [[11.936, 143.0, 98.8048], [30.0, 57.0, 96.0], [None, 136.0, 26.0], ... ]
        },
        ...
    ]
    
    """
    users = log_df["UserID"].unique() if user_ids is None else user_ids
    all_days = []

    users = [1]
    for day in range(1, days + 1):
        logging_history_differences: dict = {}
        for difference_method in difference_methods:
            logging_history_differences[difference_method.value] = []

        for user_id in users:
            f_log_df = log_df[(log_df["UserID"] == user_id) & (log_df["Carbohydrate"] > 0)]

            start_time = min(f_log_df["Timestamp"].to_numpy())
            end_time = pd.to_datetime(start_time) + timedelta(days=day)

            collected_logs = f_log_df[(f_log_df["Timestamp"] >= start_time) & (f_log_df["Timestamp"] <= end_time)]

            # Filters to be aligned with calculators
            collected_logs = collected_logs.copy()
            collected_logs["Timestamp"] = collected_logs["Timestamp"].dt.strftime("%H:%M")
            
            all_collected_logs = list(collected_logs[["Timestamp", "Carbohydrate"]].itertuples(index=False, name=None))
            
            # Compare logs
            start_time = end_time
            end_time = start_time + timedelta(days=1)

            compared_logs = f_log_df[(f_log_df["Timestamp"] > start_time) & (f_log_df["Timestamp"] <= end_time)]
            compared_logs = compared_logs.copy()

            compared_logs["Timestamp"] = compared_logs["Timestamp"].dt.strftime("%H:%M")
            all_compare_logs = list(compared_logs[["Timestamp", "Carbohydrate"]].itertuples(index=False, name=None))

            # Logging history differences
            scheduled_times = WeightedAverageSegmentedCalculator().calculate_schedule_times(all_collected_logs)
            scheduled_minute_times = [ScheduleCalculatorBase._time_to_minutes(t) for t in scheduled_times]
            compare_minute_times = [(ScheduleCalculatorBase._time_to_minutes(t), amount) for t, amount in all_compare_logs]
            for difference_method in difference_methods:
                logging_history_differences[difference_method.value].append(scheduled_actual_difference(scheduled_minute_times, compare_minute_times, difference_method))

            # Live blood glucose differences
            # For this we are just looking at spikes 

        # Day + 2 as we collected a day and evaluated a day, and days start from 0.
        all_days.append({"day": day + 2, "logging_history_differences": logging_history_differences})

    return all_days


def scheduled_actual_difference(scheduled: list[int], actual: list[tuple[int, float]], method: LogDifferenceMethod) -> list[Optional[int]]:
    """
    Measures the difference between the scheduled time and the actual time.

    scheduled: [minutes, carbohydrates] OR we could do [minutes, kilocalories]
    """
    segments = [[] for _ in SEGMENTED_SCHEDULES]
    for (minutes, energy) in actual:
        for i, (start, end) in enumerate(SEGMENTED_SCHEDULES):
            if minutes >= start and minutes < end:
                segments[i].append((minutes, energy))

    scheduled_none = [None for i in range(len(SEGMENTED_SCHEDULES))]
    for sch_mins in scheduled:
        if sch_mins is not None:
            for i, (start, end) in enumerate(SEGMENTED_SCHEDULES):
                if sch_mins >= start and sch_mins < end:
                    scheduled_none[i] = sch_mins

    if method == LogDifferenceMethod.WEIGHTED_AVG:
        actual_minutes = []
        for segment in segments:
            totalEnergy = sum([energy for _, energy in segment])
            if len(segment) == 0:
                actual_minutes.append(None)
            else:
                time = 0
                for minutes, energy in segment:
                    weight = energy / totalEnergy
                    time += weight * minutes
                actual_minutes.append(time)
       

        differences = []
        for i, act in enumerate(actual_minutes):
            if act is None or scheduled_none[i] is None:
                differences.append(None)
            else:
                diff = abs(act - scheduled_none[i])
                differences.append(diff)
        return differences
        
            
    elif method == LogDifferenceMethod.LARGEST_LOG:
        largest_logs = []
        for segment in segments:
            if len(segment) == 0:
                largest_logs.append(None)
            else:
                largest_log = 0
                for (mins, amount) in segment:
                    if amount > largest_log:
                        largest_log = mins
                largest_logs.append(mins)
        differences = []
        for i, act in enumerate(largest_logs):
            if act is None or scheduled_none[i] is None:
                differences.append(None)
            else:
                diff = abs(act - scheduled_none[i])
                differences.append(diff) 
        return differences


def compute_live_blood_glucose_difference(cgm_df: pd.DataFrame, log_df: pd.DataFrame, user_ids = None) -> list[dict]:
    """
    TODO fix reading/Reading.

    Note this is going to be MASSIVE!
    """

    methods = {
        "max_min_ratio": notify_max_min_ratio,
        "rate_of_change": notify_rate_of_change,
        "cusum": notify_cusum,
        "kalman_residual": notify_predictive_residual,
        "sustained_rise": notify_sustained_rise,
        "hybrid": notify_hybrid,
    }


    users = cgm_df["UserID"].unique() if user_ids is None else user_ids
    
    return_notifications: list[dict] = []
    for user_id in users:
        # For each 30 mins get last 60 mins of readings
        f_cgm_df = cgm_df[(cgm_df["UserID"] == user_id)]
        times = f_cgm_df["Timestamp"].to_numpy()
        time = pd.to_datetime(min(times)) 
        max_time = pd.to_datetime(max(times))

        # Ex: [{"start_time": Date, "end_time": Date, "Send notification": bool, "log_in_period": bool}, ...]
        send_notification_checks: list[dict] = []

        while time < max_time:
            # Note: the times are ordered
            upper_bound_time = time + timedelta(minutes=CRON_MIN_SCHEDULING)
            lower_bound_time = time - timedelta(minutes=CRON_MIN_SCHEDULING)
            selected_readings = f_cgm_df[(pd.to_datetime(f_cgm_df["Timestamp"]) >= lower_bound_time) & (pd.to_datetime(f_cgm_df["Timestamp"]) < upper_bound_time)]["reading"].dropna().to_numpy()
            food_logged = _check_food_loggged_in_period(log_df, user_id, lower_bound_time, upper_bound_time)

            notification_check = {
                "start_time": lower_bound_time,
                "end_time": upper_bound_time,
                "food_logged": food_logged
            }

            # Add different methods for checking if a notification should be sent
            for key_label, func in methods.items():
                notification_check[key_label] = func(selected_readings)

            send_notification_checks.append(notification_check)
            
            time = upper_bound_time
        return_notifications.append({
            "user": user_id,
            "send_notification_checks": send_notification_checks
        })
    return return_notifications

        

def _check_food_loggged_in_period(log_df: pd.DataFrame, user_id, lower_bound_time, upper_bound_time) -> bool:
    return log_df[
            (log_df["UserID"] == user_id) & 
            (log_df["Carbohydrate"] > 0) & 
            (pd.to_datetime(log_df["Timestamp"]) >= lower_bound_time) & 
            (pd.to_datetime(log_df["Timestamp"]) < upper_bound_time)
        ].shape[0] > 0
            

def blood_glucose_notification_confusion_matrix(notification_times: list[dict], method: str):
    """
    Look at FF, TT, FT, TF
    """
    return_results: list[dict] = []
    for user_data in notification_times:
        FF, TT, FT, TF = 0, 0, 0, 0
        send_notifications: list[dict] = user_data["send_notification_checks"]

        for notification_checks in send_notifications:
            sent: bool = notification_checks[method]
            logged: bool = notification_checks["food_logged"]
            if sent and logged:
                TT += 1
            elif not sent and logged:
                FT += 1
            elif sent and not logged:
                TF += 1
            else:
                FF += 1
        return_results.append({
            "user": user_data["user"],
            "TT": TT,
            "FF": FF,
            "FT": FT,
            "TF": TF
        })
    return return_results        


def plot_blood_glucose_confusion_matrix(confusion_matrix: list[dict], additional_title: str = ""):
    """
    For CGMacros, lets look at 3 different groups.
    """
    TT = sum(d['TT'] for d in confusion_matrix)
    FF = sum(d['FF'] for d in confusion_matrix)
    FT = sum(d['FT'] for d in confusion_matrix)
    TF = sum(d['TF'] for d in confusion_matrix)

    # Build a 2x2 confusion matrix
    cm = np.array([[TT, FT],
                [TF, FF]])

    # Display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=["True", "False"])

    fig, ax = plt.subplots(figsize=(10, 6))
    disp.plot(cmap="Blues", ax=ax, colorbar=True)

    title = f"{additional_title} Blood glucose notification confusion matrix | Method: {method}"

    ax.set_xlabel("Notification Sent", fontsize=12)
    ax.set_ylabel("Food logged in period", fontsize=12)
    ax.set_title(title, fontsize=14, wrap=True)

    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
    path = os.path.join("figures", safe_title + ".png")
    
    plt.savefig(path)
    plt.show()
    plt.close()
    return TT, FF, FT, TF
  
def plot_logging_history_results(logging_history_data, additional_title: str = ""):
    num_series = MAX_NOTIFICATIONS

    days = []
    all_series: list[dict[str, list]] = []

    for day_entry in logging_history_data:
        day = day_entry['day']
        days.append(day)
        data: dict[str, list] = day_entry['logging_history_differences'] 
        series_values = {key: [[] for _ in range(num_series)] for key in data.keys()}

        # For each series, collect values for this day
        for key in data.keys():
            for user_diffs in data[key]:
                for i, diff in enumerate(user_diffs):
                    if diff is not None:
                        if math.isnan(diff):
                            print(f"\n\nERROR: {diff} \n\n")
                        series_values[key][i].append(diff)

        avg_series: dict[str, list] = {key: [np.mean(s) for s in series_values[key]] for key in data.keys()}
        all_series.append(avg_series)

    # Plot each series
    keys = list(all_series[0].keys())
    labels = ["breakfast", "lunch", "dinner"]
    days = np.arange(len(all_series))  # numeric positions for x-axis

    bar_width = 0.25  # width of each bar
    offsets = np.linspace(-bar_width, bar_width, len(labels))  # shifts for breakfast/lunch/dinner

    for method in keys:
        plt.figure(figsize=(8, 5))
        for i, label in enumerate(labels):
            values = [s[method][i] for s in all_series]
            plt.bar(days + offsets[i], values, width=bar_width, label=label)
        
        # Compute averages across breakfast/lunch/dinner for each day
        averages = [np.mean(s[method]) for s in all_series]

        # Overlay line plot with dots for averages
        plt.plot(
            days, averages,
            marker="o", linestyle="-", color="black", linewidth=2,
            label="Daily Average"
        )

        plt.xlabel("Day")
        plt.ylabel("Difference (minutes)")
        title = f"{additional_title} Schedule/Notification Difference per Day | Method: {method}"
        plt.title(title)
        plt.xticks(days, [f"Day {d + 2}" for d in days])  # label each day
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
        path = os.path.join("notifications", "figures", safe_title + ".png")

        plt.savefig(path)
        plt.show()
        plt.close()


if __name__ == "__main__":
    CGMacros_cgm_df = pd.read_pickle("../data/CGMacros/pickle/cgm.pkl")
    CGMacros_log_df = pd.read_pickle("../data/CGMacros/pickle/log.pkl")

    UC_HT_T1DM_cgm_df = pd.read_pickle("../data/UC_HT_T1DM/pickle/cgm.pkl")
    UC_HT_T1DM_log_df = pd.read_pickle("../data/UC_HT_T1DM/pickle/log.pkl")

    # diffs = compute_logging_history_difference(CGMacros_log_df, 9, [LogDifferenceMethod.WEIGHTED_AVG, LogDifferenceMethod.LARGEST_LOG])
    # plot_logging_history_results(diffs, additional_title="ALL")

    # diffs = compute_logging_history_difference(CGMacros_log_df, 9, [LogDifferenceMethod.WEIGHTED_AVG, LogDifferenceMethod.LARGEST_LOG], user_ids=CGMacro_USER_GROUPS["healthy"])
    # plot_logging_history_results(diffs, additional_title="Healthy")

    # diffs = compute_logging_history_difference(CGMacros_log_df, 9, [LogDifferenceMethod.WEIGHTED_AVG, LogDifferenceMethod.LARGEST_LOG], user_ids=CGMacro_USER_GROUPS["prediabetes"])
    # plot_logging_history_results(diffs, additional_title="Prediabetes")

    # diffs = compute_logging_history_difference(CGMacros_log_df, 9, [LogDifferenceMethod.WEIGHTED_AVG, LogDifferenceMethod.LARGEST_LOG], user_ids=CGMacro_USER_GROUPS["t2dm"])
    # plot_logging_history_results(diffs, additional_title="T2DM")
    # collect_sample_days_data(UC_HT_T1DM_cgm_df, UC_HT_T1DM_log_df)
    
    r = compute_live_blood_glucose_difference(CGMacros_cgm_df, CGMacros_log_df)

    # print(r)

    # with open('data/live_bg_diff.json', 'w') as json_file:
    #     json.dump(r, json_file, indent=4, default=str)
    # print(r)

    for method in ["max_min_ratio", "rate_of_change", "cusum", "kalman_residual", "sustained_rise", "hybrid"]:


        confusion_matrix = blood_glucose_notification_confusion_matrix(r, method)
        #
        #
        # # confusion_matrix = [
        # #     {'user': 1, 'TT': 2, 'FF': 408, 'FT': 75, 'TF': 6}, {'user': 2, 'TT': 21, 'FF': 443, 'FT': 42, 'TF': 62},
        # # ]
        #
        TT, FF, FT, TF = plot_blood_glucose_confusion_matrix(confusion_matrix, additional_title="ALL")
        #

        if TT > 0 and FT > 0 and FF > 0 and TF > 0:
            print("\n" + "-" * 20 + method + "-" * 20 + "\n")
            pc = lambda x : f"{(x * 100):.2f}%"

            total = TT + FF + FT + TF

            print("Total Intervals:", total)
            for c, label in zip([TT, FF, FT, TF], ["TT", "FF", "FT", "TF"]):
                print(f"{label} proportion: {pc(c / total)}")


            print("Of the notifications that were sent")
            print(f"{pc(TT / (TT + TF))} were sent in an interval where food was logged")
            print(f"{pc(TF / (TT + TF))} were sent in an interval where food was not logged")


            print("of the notification that were not sent")
            print(f"{pc(FT / (FT + FF))} should have been sent")

    # Difference example input
    # sch, actual = [423, 803, 1212], [(779, 94.0), (1046, 27.0), (1182, 1.0), (1307, 22.0), (562, 73.0), (796, 28.0), (1265, 1.0), (1278, 14.0), (534, 24.0), (745, 93.0), (1196, 42.0), (1258, 32.0), (607, 66.0)]
    # diffs = scheduled_actual_difference(sch, actual, LogDifferenceMethod.LARGEST_LOG)
    