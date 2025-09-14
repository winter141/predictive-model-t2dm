"""
Now we look at just considering the live blood glucose that sends a reminder if blood glucose is climbing quickly.
"""

# Let's take the min and max reading in past 30mins and look at the signed percentage change
from configurations import PERECNTAGE_CHANGE_THRESHOLD


def send_reminder_blood_glucose(readings: list[float]) -> bool:
    if len(readings) == 0:
        return False
    return bool((max(readings) / min(readings)) >= PERECNTAGE_CHANGE_THRESHOLD)
