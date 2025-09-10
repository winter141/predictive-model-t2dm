"""
Simulate with different datasets.

For now lets just do the CGMacros and UC_HT_T1DM.

Note the UC_HT_T1DM only has carbs as the food logs.

UC_HT_T1DM is about five days

CGMacros is about 10 days

So we can simulate for 3 days.
"""
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

# Method: Record logging history and blood glucose for five days.
# - Compare the times between the five days after with the predictions

# - Logging history only
# - Blood glucose history only
# - Logging and blood glucose history
# - Blood glucose live (ensure 30min delay)

# For the logging history lets just try with the weighted average segmented schedule

def collect_sample_days_data(cgm_df: pd.DataFrame, log_df: pd.DataFrame, days: tuple[int, int] = (3, 3)):
    """
    Collect from public datasets.int
    days: (int, int) -> (collection_days, test_days)
    """
    users = cgm_df["UserID"].unique()
    for user_id in users:      
        f_cgm_df = cgm_df[cgm_df["UserID"] == user_id]
        f_log_df = log_df[log_df["UserID"] == user_id]

        
    return None

  

if __name__ == "__main__":
    CGMacros_cgm_df = pd.read_pickle("data/CGMacros/pickle/cgm.pkl")
    CGMacros_log_df = pd.read_pickle("data/CGMacros/pickle/log.pkl")

    UC_HT_T1DM_cgm_df = pd.read_pickle("data/UC_HT_T1DM/pickle/cgm.pkl")
    UC_HT_T1DM_log_df = pd.read_pickle("data/UC_HT_T1DM/pickle/log.pkl")

    collect_sample_days_data(CGMacros_cgm_df, CGMacros_log_df)
    collect_sample_days_data(UC_HT_T1DM_cgm_df, UC_HT_T1DM_log_df)