from datetime import timedelta

import pandas
import pandas as pd


def process_xlsx(filepath) -> dict[str, pandas.DataFrame]:
    """
    :param filepath:
    :return: Dictionary of Food Data/CGM data dataframes
    """
    return pd.read_excel(filepath, sheet_name=None)


def join_df(dataframes: dict[str, pandas.DataFrame]):
    log_df = dataframes["TEI_Cleaned"]
    cgm_df = dataframes["CGM_Cleaned"]

    # Ensure datetime
    log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'])
    cgm_df['NZT'] = pd.to_datetime(cgm_df['NZT'])

    # Function to find matching CGM rows for each log entry
    def get_cgm_window(row):
        uid = row['UserID']
        ts = row['Timestamp']
        mask = (
                (cgm_df['UserID'] == uid) &
                (cgm_df['NZT'] >= ts - timedelta(hours=2)) &
                (cgm_df['NZT'] <= ts + timedelta(hours=2))
        )
        return cgm_df[mask]

    # Add a new column with the matched CGM DataFrames
    log_df['cgm_window'] = log_df.apply(get_cgm_window, axis=1)

    return log_df


if __name__ == "__main__":
    # 4541 rows x 12 columns
    # Sex, UserID, Date, Time, Timestamp, FoodItem, Energy, Carbohydrate, Protein, Fat, Tag, Weight
    filepath = "./data/CGM_TEI_Cleaned(1).xlsx"
    file_out = "./data/log_with_cgm.pkl"
    dataframes = process_xlsx(filepath)
    df = join_df(dataframes)
    df.to_pickle(file_out)
