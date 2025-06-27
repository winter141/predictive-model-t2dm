"""
Create pdf showing CGM readings and logged food for a particular user.
"""
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from process_data import load_dataframe
import pandas as pd
from datetime import timedelta
from process_data import FeatureLabelReducer


def finalize_and_save(fig, pdf):
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


class UserPDFGenerator:

    def __init__(self, static_user, log_df, cgm_df, start_date: str, end_date: str, file_out, UserID: Union[str, int], dynamic_user=None, food_name="FoodItem"):
        """
        :param user_cgm_df:
        :param user_log_df:
        :param start_date: like yyyy/mm/dd  "2024-10-08"
        :param end_date: like yyyy/mm/dd "2024-10-08"
        :param file_out:
        """
        self.UserID = UserID
        self.file_out = file_out
        self.start_date = start_date
        self.end_date = end_date
        self.food_name = food_name


        user_cgm_df = cgm_df[cgm_df["UserID"] == UserID].copy()
        user_log_df = log_df[log_df["UserID"] == UserID].copy()

        user_cgm_df['Timestamp'] = pd.to_datetime(user_cgm_df['Timestamp'])
        user_cgm_df = user_cgm_df
        user_cgm_df['Time_num'] = user_cgm_df['Timestamp'].dt.hour * 60 + user_cgm_df['Timestamp'].dt.minute
        self.user_cgm_df = user_cgm_df

        user_log_df['Timestamp'] = pd.to_datetime(user_log_df['Timestamp'])
        user_log_df = user_log_df.sort_values('Timestamp')
        user_log_df['Time_num'] = user_log_df['Timestamp'].dt.hour * 60 + user_log_df['Timestamp'].dt.minute
        self.user_log_df = user_log_df

    def generate_cgm_logs(self):
        self.file_out += "_cgm_logs"
        with PdfPages(f"{self.file_out}.pdf") as pdf:
            for d in pd.date_range(start=self.start_date, end=self.end_date):
                self._plt_cgm_by_date(d, pdf)
        print(f"PDF report '{self.file_out}' generated successfully!")

    def generate_food_specific(self, food: str):
        self.file_out += "_food_specific"
        with PdfPages(f"{self.file_out}.pdf") as pdf:
            self._plt_food_specific(food, pdf)
        print(f"PDF report '{self.file_out}' generated successfully!")

    def generate_food_consistency(self, food: str):
        self.file_out += "_food_consistency"
        with PdfPages(f"{self.file_out}.pdf") as pdf:
            self._plt_food_consistency(food, pdf)
        print(f"PDF report '{self.file_out}' generated successfully!")

    def _plt_cgm_by_date(self, selected_date, pdf):
        date_filtered_cgm = self.user_cgm_df[self.user_cgm_df['Timestamp'].dt.date == pd.to_datetime(selected_date).date()].copy()

        # Add vertical lines for logs
        date_filtered_logs = self.user_log_df[self.user_log_df['Timestamp'].dt.date == pd.to_datetime(selected_date).date()].copy()

        self._plt_cgm_logs(date_filtered_cgm, date_filtered_logs, selected_date, pdf)

    def _plt_food_specific(self, selected_food: str, pdf):
        """
        Find all instances of that food type.
        Plot graphs for 4 hour window (2 hour before, 2 after) with all logs in that time.

        TODO later maybe: multiple plots on one page
        :param food:
        :return:
        """
        for _, log in self.user_log_df.iterrows():
            if log[self.food_name] == selected_food:
                timestamp = log["Timestamp"]
                cgm_window, log_window = self._cgm_log_df_in_timeframe(2, 2, timestamp)
                self._plt_cgm_logs(cgm_window, log_window, f"{timestamp}: {selected_food}", pdf)


        pass

    def _plt_food_consistency(self, selected_food: str, pdf):
        fig, ax = plt.subplots(figsize=(12, 8))

        for _, log in self.user_log_df.iterrows():
            if log[self.food_name] == selected_food:
                timestamp = log["Timestamp"]
                cgm_window, log_window = self._cgm_log_df_in_timeframe(1, 2, timestamp)
                ppgr_window, _ = self._cgm_log_df_in_timeframe(0, 2, timestamp)
                iAUC = FeatureLabelReducer.reduce_cgm_window_to_area(ppgr_window.copy())
                if not np.isnan(iAUC):
                    ax.plot(cgm_window["Time_num"] - log["Time_num"], cgm_window["reading"], 'o--', label=f"{timestamp} iAUC: {iAUC:.0f} mmol/L")

        ax.axvline(x=0, linestyle='--', label=f"{selected_food} logged")
        ax.set_title(f"{selected_food} log consistency for {self.UserID}")
        ax.set_xlabel("Minutes Since Food Log")
        ax.set_ylabel("Glucose Level (mmol/L)")
        ax.legend()
        finalize_and_save(fig, pdf)
        plt.close(fig)

    def _plt_cgm_logs(self, cgm_window, log_window, title, pdf):
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(cgm_window["Time_num"], cgm_window["reading"], 'o--')

        # Use the numeric positions for ticks, but label with corresponding HH:MM strings
        hours = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
        labels = [f"{'0' if x < 10 else ''}{x}:00" for x in hours]
        ticks = [x * 60 for x in hours]
        ax.set_xticks(ticks=ticks, labels=labels)

        for _, log in log_window.iterrows():
            ax.plot([log["Time_num"] for _ in range(2)], [5, 130], '-', label=log[self.food_name])


        ax.legend()
        ax.set_title(title)
        finalize_and_save(fig, pdf)


    def _cgm_log_df_in_timeframe(self, hours_before: float, hours_after: float, timestamp):
        """
        :return: Tuple of
            - CGM Dataframe
            - Log Dataframe
        """
        cgm_mask = (
                (self.user_cgm_df['Timestamp'] >= timestamp - timedelta(hours=hours_before)) &
                (self.user_cgm_df['Timestamp'] <= timestamp + timedelta(hours=hours_after))
        )
        cgm_window = self.user_cgm_df[cgm_mask]

        log_mask = (
                (self.user_log_df['Timestamp'] >= timestamp - timedelta(hours=hours_before)) &
                (self.user_log_df['Timestamp'] <= timestamp + timedelta(hours=hours_after))
        )
        log_window = self.user_log_df[log_mask]
        return cgm_window, log_window


if __name__ == "__main__":
    UserID = 1
    file_out = f"./results/CGMacros/{UserID}"

    # Old Dataset
    # cgm_df = load_dataframe("data/old/pickle/cgm.pkl")
    # log_df = load_dataframe("data/old/pickle/log.pkl")
    # static_user = load_dataframe("data/old/pickle/log.pkl")


    # New Dataset
    cgm_df = load_dataframe("./data/CGMacros/pickle/cgm.pkl")
    log_df = load_dataframe("./data/CGMacros/pickle/log.pkl")
    static_user = load_dataframe("./data/CGMacros/pickle/static_user.pkl")
    dynamic_user = load_dataframe("./data/CGMacros/pickle/dynamic_user.pkl")

    # FoodItem frequency
    #print(log_df['FoodItem'].value_counts())

    generator = UserPDFGenerator(static_user, log_df, cgm_df, "2020-05-01", "2020-05-11", file_out, UserID, food_name="Meal Type")
    # generator.generate_food_consistency("Dinner")
    generator.generate_cgm_logs()














