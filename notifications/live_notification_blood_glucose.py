"""
Now we look at just considering the live blood glucose that sends a reminder if blood glucose is climbing quickly.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

# Let's take the min and max reading in past 30mins and look at the signed percentage change
from configurations import PERECNTAGE_CHANGE_THRESHOLD


def notify_max_min_ratio(readings: list[float], threshold=PERECNTAGE_CHANGE_THRESHOLD) -> bool:
    if len(readings) == 0:
        return False
    return bool((max(readings) / min(readings)) >= threshold)


# __________________ BELOW ARE UNTESTED ____________________ #

# --- Method 1: Rate of Change Detector ---
# Based on: Zecchin et al., "Meal detection in diabetes by continuous glucose monitoring sensors: a change detection approach," 2013.
def notify_rate_of_change(
    cgm_values: list[float],
    sampling_interval_min: int = 60,
    threshold_mmol_per_hour = 1,
    min_points: int = 2
) -> bool:
    """
    Detect if glucose rise is steep. Returns True if rise rate (mmol/L per hour) >= threshold.
    - cgm_values: list of mmol/L ordered oldest->newest
    - sampling_interval_min: minutes between samples (e.g. 60 for hourly blocks)
    - threshold_mmol_per_hour: default tuned to be sensitive for hourly blocks (~1.0 mmol/L/hr)
    - min_points: minimum samples required (2)
    Good default: threshold_mmol_per_hour = 1.0 (≈18 mg/dL per hour).
    """
    if len(cgm_values) < min_points:
        return False

    elapsed_min = sampling_interval_min * (len(cgm_values) - 1)
    if elapsed_min <= 0:
        return False

    rate_per_hour = (cgm_values[-1] - cgm_values[0]) / elapsed_min * 60.0  # mmol/L per hour
    return rate_per_hour >= threshold_mmol_per_hour


# --- Method 2: CUSUM Change-Point Detection ---
# Based on: Zecchin et al., "A new algorithm for glucose monitoring in diabetes based on CUSUM control charts," 2012.
def notify_cusum(
    cgm_values: list[float],
    sampling_interval_min: int = 60,
    k  = None,
    h = None,
    min_points: int = 3
) -> bool:
    """
    CUSUM over differences. Values in mmol/L.
    - k: reference value (mmol). If None, it's set to 0.5 * std(diff) or 0.05 whichever larger.
    - h: decision threshold (mmol). If None, set to 3*std(diff) or 0.2 whichever larger.
    Works better with >=3 points.
    """
    if len(cgm_values) < min_points:
        return False

    diffs = np.diff(cgm_values)  # per-sample differences in mmol
    std_diff = float(np.std(diffs)) if len(diffs) > 0 else 0.0

    if k is None:
        k = max(0.05, 0.5 * std_diff)  # reference step (mmol)
    if h is None:
        h = max(0.2, 3.0 * std_diff)   # decision threshold (mmol)

    s_pos, s_neg = 0.0, 0.0
    for d in diffs:
        s_pos = max(0.0, s_pos + d - k)
        s_neg = min(0.0, s_neg + d + k)
        if s_pos > h:
            return True
        if abs(s_neg) > h:
            return True
    return False


# --- Method 3: Kalman Filter Residual Detector ---
# Based on: Turksoy et al., "Meal detection in patients with type 1 diabetes: a Kalman filter based approach," 2013.
def notify_predictive_residual(
    cgm_values: list[float],
    threshold_mmol = 0.5,
    min_points: int = 3
) -> bool:
    """
    Fit a linear trend to earlier points (all except last) and predict the last sample.
    If last - predicted_last > threshold_mmol => notify (sudden unexpected rise).
    Returns (bool, residual).
    - threshold_mmol default ~0.5 mmol (≈9 mg/dL), adjust upwards for noisy users.
    """
    if len(cgm_values) < min_points:
        return False, 0.0


    # Use previous n-1 points to predict the nth
    X = np.arange(len(cgm_values) - 1).reshape(-1, 1)  # times 0..n-2
    y = np.array(cgm_values[:-1])
    model = LinearRegression()
    model.fit(X, y)
    t_pred = np.array([[len(cgm_values) - 1]])  # predict the index of the last sample
    pred_last = float(model.predict(t_pred)[0])
    residual = cgm_values[-1] - pred_last
    return residual >= threshold_mmol


# --- Method 5: First Derivative + Slope Duration ---
# Based on: Cinar et al., "Meal detection and meal size estimation for artificial pancreas systems," 2018.
def notify_sustained_rise(
    cgm_values: list[float],
    sampling_interval_min: int = 60,
    slope_threshold_mmol_per_hour = 1,
    consecutive_intervals: int = 1,
    min_points: int = 2
) -> bool:
    """
    Detect if there are `consecutive_intervals` consecutive positive slopes above threshold.
    - slope_threshold_mmol_per_hour default 1.0 mmol/hr (adjustable)
    - consecutive_intervals is how many consecutive per-sample intervals must exceed threshold
    Note: for hourly blocks, set slope_threshold_mmol_per_hour ~ 0.5-1.5 depending on sensitivity.
    """
    if len(cgm_values) < min_points:
        return False

    # Convert per-hour threshold to per-interval threshold:
    slope_threshold_per_interval = slope_threshold_mmol_per_hour * (sampling_interval_min / 60.0)

    diffs = np.diff(cgm_values)  # mmol per interval
    count = 0
    for d in diffs:
        if d >= slope_threshold_per_interval:
            count += 1
            if count >= consecutive_intervals:
                return True
        else:
            count = 0
    return False


# ------------------------
# 6) Hybrid rule (relative + absolute)
# ------------------------
def notify_hybrid(
    cgm_values: list[float],
    ratio_threshold: float = 1.4,
    abs_threshold_mmol: float = 1.0
) -> bool:
    """
    Combine relative and absolute thresholds:
     - notify if max/min >= ratio_threshold AND (max - min) >= abs_threshold_mmol
    This reduces false positives from small relative changes at low absolute values.
    """
    if len(cgm_values) == 0:
        return False
    mn, mx = min(cgm_values), max(cgm_values)
    if mn <= 0:
        return False
    return (mx / mn >= ratio_threshold) and ((mx - mn) >= abs_threshold_mmol)

def notify_decision(
    readings: list[float],
    sampling_interval_min: int = 60,
    prefer: str = "auto"
) -> dict:
    """
    Runs several detectors and returns a summary:
    Returns dict:
      { "max_min":bool, "roc":bool, "cusum":bool, "pred_res":(bool,resid), "sustained":bool, "hybrid":bool }
    'prefer' controls fallback: "auto" leaves it to detectors; "safe" returns True if any method triggers;
    You can decide notification policy based on this summary.
    """
    out = {}
    # always compute max/min (fast, robust)
    out["max_min"] = notify_max_min_ratio(readings, threshold=1.5)

    # ROC sensible default: 1 mmol/hour
    out["roc"] = notify_rate_of_change(readings, sampling_interval_min=sampling_interval_min,
                                       threshold_mmol_per_hour=1.0, min_points=2)

    # CUSUM adaptive
    out["cusum"] = notify_cusum(readings, sampling_interval_min=sampling_interval_min)

    # predictive residual (returns (bool,resid))
    pred_flag, resid = notify_predictive_residual(readings, sampling_interval_min=sampling_interval_min,
                                                  threshold_mmol=0.5)
    out["pred_residual"] = {"flag": pred_flag, "residual": resid}

    # sustained rise: require 1 consecutive interval by default
    out["sustained"] = notify_sustained_rise(readings, sampling_interval_min=sampling_interval_min,
                                             slope_threshold_mmol_per_hour=1.0, consecutive_intervals=1)

    # hybrid stricter rule
    out["hybrid"] = notify_hybrid(readings, ratio_threshold=1.4, abs_threshold_mmol=1.0)

    # decide overall (configurable)
    if prefer == "auto":
        # conservative policy: require either hybrid OR (roc OR cusum OR pred_residual)
        overall = out["hybrid"] or (out["roc"] or out["cusum"] or out["pred_residual"]["flag"])
    elif prefer == "any":
        overall = any([v if not isinstance(v, dict) else v["flag"] for v in out.values()])
    elif prefer == "maxmin_only":
        overall = out["max_min"]
    else:
        overall = out["hybrid"] or out["roc"]

    out["notify_overall"] = bool(overall)
    return out


# --- Method 4: Machine Learning Classifier (Logistic Regression) ---
# Based on: Perez-Gandia et al., "Artificial neural networks for meal detection in type 1 diabetes," 2010.
# Here we stub with logistic regression probability > 0.5.
# from sklearn.linear_model import LogisticRegression
#
# def train_meal_classifier(X_train, y_train):
#     model = LogisticRegression()
#     model.fit(X_train, y_train)
#     return model
#
# def notify_ml(model, feature_vector):
#     """
#     Use trained classifier to predict meal vs. no meal.
#     """
#     prob = model.predict_proba([feature_vector])[0,1]
#     return prob > 0.5

