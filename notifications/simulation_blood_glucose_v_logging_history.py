"""
Simulate with different datasets.

For now lets just do the CGMacros and UC_HT_T1DM.

Note the UC_HT_T1DM only has carbs as the food logs.

UC_HT_T1DM is about five days

CGMacros is about 10 days

So we can simulate for 3 days.

Not the most efficient, but it does not matter :)
"""
from matplotlib import pyplot as plt
import pandas as pd
import sys
from pathlib import Path
from datetime import timedelta
from enum import Enum

from sklearn.metrics import ConfusionMatrixDisplay
from live_blood_glucose_configurations import CRON_MIN_SCHEDULING
from live_notification_blood_glucose import send_reminder_blood_glucose
from schedule_configurations import SEGMENTED_SCHEDULES, MIN_MINUTES_APART
from calculate_schedule_times import ScheduleCalculatorBase, WeightedAverageSegmentedCalculator
from typing import Optional
import numpy as np
import math
sys.path.append(str(Path(__file__).resolve().parent))


# Done in measuring_iauc/filter_particpants. It is by A1c which is the method identified in the CGMacros documentation.
CGMacroUserGroups = {
    "healthy": [1,  2, 4, 6, 15, 17, 18, 19, 21, 27, 31, 32, 33, 34, 48],
    "prediabetes": [7, 8,  9, 10, 11, 13, 16, 20, 22, 23, 26, 29, 41, 43, 44, 45],
    "t2dm": [3, 5, 12, 14, 28, 30, 35, 36, 38, 39, 42, 46, 47, 49]
}

class LogDifferenceMethod(Enum):
    WEIGHTED_AVG = 0,  # In segment
    LARGEST_LOG = 1    # In segment


def compute_logging_history_difference(cgm_df: pd.DataFrame, log_df: pd.DataFrame, days: int, difference_method: LogDifferenceMethod) -> list[dict]:
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
    users = cgm_df["UserID"].unique()
    all_days = []
    for day in range(days + 1):
        logging_history_differences = []
        for user_id in users:
            f_cgm_df = cgm_df[(cgm_df["UserID"] == user_id)]
            f_log_df = log_df[(log_df["UserID"] == user_id) & (log_df["Carbohydrate"] > 0)]

            start_time = min(f_cgm_df["Timestamp"].to_numpy())
            end_time = pd.to_datetime(start_time) + timedelta(days=day)

            collected_logs = f_log_df[(f_log_df["Timestamp"] >= start_time) & (f_log_df["Timestamp"] <= end_time)]
            collected_cgm = f_cgm_df[(f_cgm_df["Timestamp"] >= start_time) & (f_cgm_df["Timestamp"] <= end_time)]


            # Filters to be aligned with calculators
            collected_logs = collected_logs.copy()
            collected_cgm = collected_cgm.copy()
            collected_logs["Timestamp"] = collected_logs["Timestamp"].dt.strftime("%H:%M")
            collected_cgm["Timestamp"] = collected_cgm["Timestamp"].dt.strftime("%H:%M")
            
            all_collected_logs = list(collected_logs[["Timestamp", "Carbohydrate"]].itertuples(index=False, name=None))
            
            # Compare logs
            start_time = end_time
            end_time = start_time + timedelta(days=1)
            compared_logs = f_log_df[(f_log_df["Timestamp"] > start_time) & (f_log_df["Timestamp"] <= end_time)]
            compared_cgm = f_cgm_df[(f_cgm_df["Timestamp"] > start_time) & (f_cgm_df["Timestamp"] <= end_time)]
            compared_logs = compared_logs.copy()
            compared_cgm = compared_cgm.copy()

            compared_logs["Timestamp"] = compared_logs["Timestamp"].dt.strftime("%H:%M")
            compared_cgm["Timestamp"] = compared_cgm["Timestamp"].dt.strftime("%H:%M")
            all_compare_logs = list(compared_logs[["Timestamp", "Carbohydrate"]].itertuples(index=False, name=None))

            # Logging history differences
            scheduled_times = WeightedAverageSegmentedCalculator().calculate_schedule_times(all_collected_logs)
            scheduled_minute_times = [ScheduleCalculatorBase._time_to_minutes(t) for t in scheduled_times]
            compare_minute_times = [(ScheduleCalculatorBase._time_to_minutes(t), amount) for t, amount in all_compare_logs]
            logging_history_differences.append(scheduled_actual_difference(scheduled_minute_times, compare_minute_times, difference_method))

            # Live blood glucose differences
            # For this we are just looking at spikes 
        all_days.append({"day": day, "logging_history_differences": logging_history_differences})

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


def compute_live_blood_glucose_difference(cgm_df: pd.DataFrame, log_df: pd.DataFrame) -> list[dict]:
    """
    TODO fix reading/Reading.

    Note this is going to be MASSIVE!
    """
    users = cgm_df["UserID"].unique()
    
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
            selected_readings = f_cgm_df[(pd.to_datetime(f_cgm_df["Timestamp"]) >= lower_bound_time) & (pd.to_datetime(f_cgm_df["Timestamp"]) < upper_bound_time)]["reading"].to_numpy()
            notification_sent = send_reminder_blood_glucose(selected_readings)
            food_logged = _check_food_loggged_in_period(log_df, user_id, lower_bound_time, upper_bound_time)

            notification_check = {
                "start_time": lower_bound_time,
                "end_time": upper_bound_time,
                "notification_sent": notification_sent,
                "food_logged": food_logged
            }
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
            

def blood_glucose_notification_confusin_matrix(notification_times: list[dict]):
    """
    Look at FF, TT, FT, TF
    """
    return_results: list[dict] = []
    for user_data in notification_times:
        FF, TT, FT, TF = 0, 0, 0, 0
        send_notifications: list[dict] = user_data["send_notification_checks"]

        for notification_checks in send_notifications:
            sent: bool = notification_checks["notification_sent"]
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


def plot_blood_glucose_confusion_matrix(confusion_matrix: list[dict]):
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

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", ax=ax, colorbar=True)

    ax.set_xlabel("Notification Sent", fontsize=12)
    ax.set_ylabel("Food logged in period", fontsize=12)
    ax.set_title("Blood glucose notification confusion matrix", fontsize=14)

    plt.show()
  
def plotter(data_by_day):
    # Number of series (3 in your case)
    num_series = 3

    # Prepare lists to hold the values per series
    days = []
    all_series = []

    for day_entry in data_by_day:
        day = day_entry['day']
        days.append(day)
        data = day_entry['logging_history_differences'] 
        series_values = [[] for _ in range(num_series)]

        # For each series, collect values for this day
        for user_diffs in data:
            for i, diff in enumerate(user_diffs):
                if diff is not None:
                    if math.isnan(diff):
                        print(f"\n\nERROR: {diff} \n\n")
                    series_values[i].append(diff)

        avg_series = [np.mean(s) for s in series_values]
        all_series.append(avg_series)

    # Plot each series
    for i, label in enumerate(["breakfast", "lunch", "dinner"]):
        plt.plot(days, [s[i] for s in all_series], marker='o', label=label)

    plt.xlabel('Day')
    plt.ylabel('Difference')
    plt.title('Differences per Day')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    CGMacros_cgm_df = pd.read_pickle("data/CGMacros/pickle/cgm.pkl")
    CGMacros_log_df = pd.read_pickle("data/CGMacros/pickle/log.pkl")

    UC_HT_T1DM_cgm_df = pd.read_pickle("data/UC_HT_T1DM/pickle/cgm.pkl")
    UC_HT_T1DM_log_df = pd.read_pickle("data/UC_HT_T1DM/pickle/log.pkl")

    # diffs = compute_logging_history_difference(CGMacros_cgm_df, CGMacros_log_df, 5, LogDifferenceMethod.WEIGHTED_AVG)
    # diffs2 = compute_logging_history_difference(CGMacros_cgm_df, CGMacros_log_df, 5, LogDifferenceMethod.LARGEST_LOG)
    # print(diffs)
    # plotter(diffs)
    # plotter(diffs2)
    # collect_sample_days_data(UC_HT_T1DM_cgm_df, UC_HT_T1DM_log_df)
    
    # r = compute_live_blood_glucose_difference(CGMacros_cgm_df, CGMacros_log_df)
    # a_r = blood_glucose_notification_confusin_matrix(r)
    # print(a_r)

    confusion_matrix = [
        {'user': 1, 'TT': 2, 'FF': 408, 'FT': 75, 'TF': 6}, {'user': 2, 'TT': 21, 'FF': 443, 'FT': 42, 'TF': 62}, 
        {'user': 3, 'TT': 27, 'FF': 368, 'FT': 43, 'TF': 48}, {'user': 4, 'TT': 14, 'FF': 356, 'FT': 88, 'TF': 18}, 
        {'user': 5, 'TT': 8, 'FF': 341, 'FT': 76, 'TF': 57}, {'user': 6, 'TT': 0, 'FF': 383, 'FT': 75, 'TF': 24}, 
        {'user': 7, 'TT': 7, 'FF': 264, 'FT': 26, 'TF': 50}, {'user': 8, 'TT': 5, 'FF': 392, 'FT': 55, 'TF': 40}, 
        {'user': 9, 'TT': 8, 'FF': 385, 'FT': 67, 'TF': 19}, {'user': 10, 'TT': 14, 'FF': 428, 'FT': 64, 'TF': 33}, 
        {'user': 11, 'TT': 17, 'FF': 400, 'FT': 55, 'TF': 22}, {'user': 12, 'TT': 16, 'FF': 375, 'FT': 46, 'TF': 52}, 
        {'user': 13, 'TT': 9, 'FF': 432, 'FT': 66, 'TF': 21}, {'user': 14, 'TT': 13, 'FF': 474, 'FT': 59, 'TF': 29}, 
        {'user': 15, 'TT': 17, 'FF': 448, 'FT': 72, 'TF': 38}, {'user': 16, 'TT': 5, 'FF': 459, 'FT': 51, 'TF': 28}, 
        {'user': 17, 'TT': 9, 'FF': 426, 'FT': 75, 'TF': 30}, {'user': 18, 'TT': 9, 'FF': 829, 'FT': 63, 'TF': 19}, 
        {'user': 19, 'TT': 6, 'FF': 379, 'FT': 77, 'TF': 20}, {'user': 20, 'TT': 11, 'FF': 444, 'FT': 64, 'TF': 23}, 
        {'user': 21, 'TT': 28, 'FF': 442, 'FT': 47, 'TF': 21}, {'user': 22, 'TT': 6, 'FF': 479, 'FT': 75, 'TF': 28}, 
        {'user': 23, 'TT': 20, 'FF': 412, 'FT': 62, 'TF': 46}, {'user': 26, 'TT': 27, 'FF': 410, 'FT': 41, 'TF': 66}, 
        {'user': 27, 'TT': 10, 'FF': 395, 'FT': 56, 'TF': 24}, {'user': 28, 'TT': 6, 'FF': 538, 'FT': 62, 'TF': 19}, 
        {'user': 29, 'TT': 15, 'FF': 507, 'FT': 45, 'TF': 24}, {'user': 30, 'TT': 7, 'FF': 419, 'FT': 43, 'TF': 47}, 
        {'user': 31, 'TT': 5, 'FF': 366, 'FT': 77, 'TF': 10}, {'user': 32, 'TT': 13, 'FF': 361, 'FT': 23, 'TF': 63}, 
        {'user': 33, 'TT': 31, 'FF': 398, 'FT': 31, 'TF': 69}, {'user': 34, 'TT': 17, 'FF': 360, 'FT': 75, 'TF': 38}, 
        {'user': 35, 'TT': 9, 'FF': 420, 'FT': 37, 'TF': 14}, {'user': 36, 'TT': 19, 'FF': 486, 'FT': 37, 'TF': 29}, 
        {'user': 38, 'TT': 11, 'FF': 403, 'FT': 50, 'TF': 20}, {'user': 39, 'TT': 14, 'FF': 481, 'FT': 48, 'TF': 31}, 
        {'user': 41, 'TT': 11, 'FF': 372, 'FT': 75, 'TF': 19}, {'user': 42, 'TT': 3, 'FF': 426, 'FT': 37, 'TF': 18}, 
        {'user': 43, 'TT': 16, 'FF': 357, 'FT': 88, 'TF': 24}, {'user': 44, 'TT': 16, 'FF': 380, 'FT': 65, 'TF': 24}, 
        {'user': 45, 'TT': 21, 'FF': 408, 'FT': 41, 'TF': 49}, {'user': 46, 'TT': 13, 'FF': 404, 'FT': 42, 'TF': 31}, 
        {'user': 47, 'TT': 7, 'FF': 451, 'FT': 51, 'TF': 14}, {'user': 48, 'TT': 21, 'FF': 472, 'FT': 50, 'TF': 36}, 
        {'user': 49, 'TT': 19, 'FF': 395, 'FT': 37, 'TF': 60}
        ]
    
    plot_blood_glucose_confusion_matrix(confusion_matrix)
    # Difference example input
    # sch, actual = [423, 803, 1212], [(779, 94.0), (1046, 27.0), (1182, 1.0), (1307, 22.0), (562, 73.0), (796, 28.0), (1265, 1.0), (1278, 14.0), (534, 24.0), (745, 93.0), (1196, 42.0), (1258, 32.0), (607, 66.0)]
    # diffs = scheduled_actual_difference(sch, actual, LogDifferenceMethod.LARGEST_LOG)
    