"""
Sandbox file.

"""
import pandas

from process_data import load_dataframe
import pandas as pd
import matplotlib.pyplot as plt


def plt_cgm_with_logs(cgm_window: pandas.DataFrame, logs: pandas.DataFrame):
    cgm_window['NZT'] = pd.to_datetime(cgm_window['NZT'])
    cgm_window = cgm_window.sort_values('NZT')
    logs['Timestamp'] = pd.to_datetime(logs['Timestamp'])
    logs = logs.sort_values('Timestamp')

    # Extract time for x-axis
    cgm_window['Time_str'] = cgm_window['NZT'].dt.strftime('%H:%M')
    logs['Time_str'] = logs['Timestamp'].dt.strftime('%H:%M')

    # Convert time to numeric: minutes since midnight
    cgm_window['Time_num'] = cgm_window['NZT'].dt.hour * 60 + cgm_window['NZT'].dt.minute
    logs['Time_num'] = logs['Timestamp'].dt.hour * 60 + logs['Timestamp'].dt.minute

    # Get the date for the title
    plot_date = cgm_window['NZT'].dt.date.iloc[0]

    plt.plot(cgm_window["Time_num"], cgm_window["value"], 'o--')

    # Plot logs
    n = len(logs)
    y = [10 for _ in range(n)]

    plt.plot(logs["Time_num"], y, 'o')
    # Add a label on every point
    for i, log in logs.iterrows():  # or any text you want
        plt.text(log["Time_num"], 10.1 if i % 2 == 0 else 9.5, log["FoodItem"], rotation=45, fontsize=8, ha='left', va='bottom')

    plt.xlabel("Time")
    plt.title(f"CGM Values on {plot_date}")

    # Use the numeric positions for ticks, but label with corresponding HH:MM strings
    hours = [2, 4, 6, 8, 10, 12, 14, 16 , 18, 20, 22]
    labels = [f"{'0' if x < 10 else ''}{x}:00" for x in hours]
    ticks = [x * 60 for x in hours]

    plt.xticks(ticks=ticks, labels=labels)

    plt.show()


if __name__ == "__main__":
    # df = load_dataframe()
    # print(df.loc[0])
    # print(df.loc[0, "cgm_window"])
    # plt_cgm_window(df.loc[0, "cgm_window"])

    cgm_df = load_dataframe("./data/raw_cgm.pkl")
    UserID = "AE22VM"
    date_selected = "2024-10-08"

    cgm_df = cgm_df[cgm_df["UserID"] == UserID]
    cgm_df = cgm_df[cgm_df['NZT'].dt.date == pd.to_datetime(date_selected).date()]

    log_df = load_dataframe("./data/raw_log.pkl")
    log_df = log_df[log_df["UserID"] == UserID]
    log_df = log_df[log_df["Date"].dt.date == pd.to_datetime(date_selected).date()]


    plt_cgm_with_logs(cgm_df, log_df)