# --- LOGGING SCHEDULING --- #
MIN_MINUTES_APART = 120
MAX_NOTIFICATIONS = 3
BUCKET_MINUTE_SIZE = 15
NOTIFICATION_MINUTE_BUFFER = 0  # We want to send the notification a little after they usually eat

SEGMENTED_SCHEDULES = [(0, 11 * 60), (11*60, 16*60), (16 * 60, 24 * 60)]

INITIAL_SCHEDULE_TIMES = [8 * 60, 12.5 * 60, 18 * 60]
# -------------------------- #

# ---- LIVE BLOOD GLUCOSE --- #
# Check every 30mins
CRON_MIN_SCHEDULING = 30

# max/min 
PERECNTAGE_CHANGE_THRESHOLD = 1.5
# ---------------------------- #