"""
Apple Watch / Apple Health Stress Classifier
=============================================
Apple Watch has no public REST API. Its data lives in Apple HealthKit,
which only iOS apps can read directly. This script gets that data into
Python via the Health app's export:

  iPhone -> Health app -> profile icon (top right) -> Export All Health Data
  -> unzip the file -> you'll find "export.xml" inside "apple_health_export/"

Then call load_from_export_xml("export.xml") below.

If you instead have a live aggregator (Terra API, Human API, Validic, etc.)
that syncs HealthKit to the cloud, just write a small function that returns
a DataFrame with the same columns as load_from_export_xml() produces
(hrv_sdnn, resting_hr, sleep_hours, respiratory_rate, heart_rate_avg, spo2,
steps, active_energy) - everything below this point works unchanged.

Install deps:
    pip install pandas numpy --break-system-packages
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 1. Load & aggregate raw Apple Health export into one row per day
# ---------------------------------------------------------------------------

HK_TYPES = {
    "heart_rate_avg": "HKQuantityTypeIdentifierHeartRate",
    "resting_hr": "HKQuantityTypeIdentifierRestingHeartRate",
    "hrv_sdnn": "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
    "respiratory_rate": "HKQuantityTypeIdentifierRespiratoryRate",
    "spo2": "HKQuantityTypeIdentifierOxygenSaturation",
    "steps": "HKQuantityTypeIdentifierStepCount",
    "active_energy": "HKQuantityTypeIdentifierActiveEnergyBurned",
    "sleep": "HKCategoryTypeIdentifierSleepAnalysis",
}

# how to combine multiple same-day readings per metric
AGG = {
    "heart_rate_avg": "mean",
    "resting_hr": "mean",
    "hrv_sdnn": "mean",
    "respiratory_rate": "mean",
    "spo2": "mean",
    "steps": "sum",
    "active_energy": "sum",
}

ASLEEP_VALUES = {
    "HKCategoryValueSleepAnalysisAsleepCore",
    "HKCategoryValueSleepAnalysisAsleepDeep",
    "HKCategoryValueSleepAnalysisAsleepREM",
    "HKCategoryValueSleepAnalysisAsleepUnspecified",
    "HKCategoryValueSleepAnalysisAsleep",
}


def load_from_export_xml(path: str) -> pd.DataFrame:
    """Parse Apple Health's export.xml into a tidy per-day DataFrame."""
    buckets = {k: [] for k in HK_TYPES}

    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "Record":
            rtype = elem.get("type")
            for key, hk_id in HK_TYPES.items():
                if rtype == hk_id:
                    buckets[key].append({
                        "start": pd.to_datetime(elem.get("startDate")),
                        "end": pd.to_datetime(elem.get("endDate")),
                        "value": elem.get("value"),
                    })
        elem.clear()

    daily = {}

    for key, how in AGG.items():
        if not buckets[key]:
            continue
        df = pd.DataFrame(buckets[key])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = df["start"].dt.date
        daily[key] = df.groupby("date")["value"].agg(how)

    if buckets["sleep"]:
        df = pd.DataFrame(buckets["sleep"])
        df = df[df["value"].isin(ASLEEP_VALUES)]
        df["duration_hr"] = (df["end"] - df["start"]).dt.total_seconds() / 3600
        df["date"] = df["end"].dt.date  # attribute to wake-up date
        daily["sleep_hours"] = df.groupby("date")["duration_hr"].sum()

    return pd.DataFrame(daily).sort_index()


# ---------------------------------------------------------------------------
# 2. Stress scoring
# ---------------------------------------------------------------------------

@dataclass
class StressResult:
    score: float        # 0-100, higher = more stressed
    level: str           # "Low" / "Medium" / "High" / "Unknown"
    contributors: dict   # per-metric signed z-scores, for transparency

# sign shows direction of "more stress"; magnitude is relative importance
WEIGHTS = {
    "hrv_sdnn": -0.30,         # lower HRV -> more stress (strongest signal)
    "resting_hr": 0.25,         # higher resting HR -> more stress
    "sleep_hours": -0.20,       # less sleep -> more stress
    "respiratory_rate": 0.10,   # higher resp rate -> more stress
    "heart_rate_avg": 0.10,     # higher daytime avg HR -> more stress
    "spo2": -0.05,              # lower blood oxygen -> more stress
}


def build_baseline(daily: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Rolling personal baseline (mean/std) per metric, so scoring adapts to you."""
    baseline = pd.DataFrame(index=daily.index)
    for col in daily.columns:
        baseline[f"{col}_mean"] = daily[col].rolling(window, min_periods=5).mean()
        baseline[f"{col}_std"] = daily[col].rolling(window, min_periods=5).std()
    return baseline


def _score(row: pd.Series, baseline_row: pd.Series) -> StressResult:
    contributors, total, total_weight = {}, 0.0, 0.0

    for metric, weight in WEIGHTS.items():
        if metric not in row or pd.isna(row.get(metric)):
            continue
        mean = baseline_row.get(f"{metric}_mean")
        std = baseline_row.get(f"{metric}_std")
        if mean is None or std in (None, 0) or pd.isna(mean) or pd.isna(std):
            continue
        z = (row[metric] - mean) / std
        signed_z = z * np.sign(weight)  # positive = more stress
        contributors[metric] = round(float(signed_z), 2)
        total += abs(weight) * signed_z
        total_weight += abs(weight)

    if total_weight == 0:
        return StressResult(score=50.0, level="Unknown", contributors={})

    avg_z = total / total_weight
    score = 100 / (1 + np.exp(-avg_z))  # logistic squash, 50 = baseline

    level = "Low" if score < 35 else "Medium" if score < 65 else "High"
    return StressResult(score=round(float(score), 1), level=level, contributors=contributors)


def classify_history(daily: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Score every day in a DataFrame produced by load_from_export_xml()."""
    baseline = build_baseline(daily, window=window)
    rows = []
    for date, row in daily.iterrows():
        res = _score(row, baseline.loc[date])
        rows.append({"date": date, "stress_score": res.score, "stress_level": res.level, **res.contributors})
    return pd.DataFrame(rows).set_index("date")


def classify_snapshot(hrv_sdnn=None, resting_hr=None, sleep_hours=None,
                       respiratory_rate=None, heart_rate_avg=None, spo2=None,
                       baseline: dict = None) -> StressResult:
    """
    Classify one reading against a personal baseline dict, e.g.:
        {"hrv_sdnn_mean": 45, "hrv_sdnn_std": 8,
         "resting_hr_mean": 58, "resting_hr_std": 4,
         "sleep_hours_mean": 7.2, "sleep_hours_std": 0.8}
    Build a real baseline once from classify_history()/build_baseline(),
    then reuse it here for quick one-off checks.
    """
    row = pd.Series({
        "hrv_sdnn": hrv_sdnn, "resting_hr": resting_hr, "sleep_hours": sleep_hours,
        "respiratory_rate": respiratory_rate, "heart_rate_avg": heart_rate_avg, "spo2": spo2,
    })
    return _score(row, pd.Series(baseline or {}))


# ---------------------------------------------------------------------------
# 3. Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Real usage, once you've exported your Health data ---
    # daily = load_from_export_xml("export.xml")
    # history = classify_history(daily)
    # print(history.tail(14))

    # --- Quick demo, no export needed ---
    demo_baseline = {
        "hrv_sdnn_mean": 45, "hrv_sdnn_std": 8,
        "resting_hr_mean": 58, "resting_hr_std": 4,
        "sleep_hours_mean": 7.2, "sleep_hours_std": 0.8,
        "respiratory_rate_mean": 15, "respiratory_rate_std": 1.5,
    }
    today = classify_snapshot(
        hrv_sdnn=30,       # well below baseline -> stress signal
        resting_hr=66,     # above baseline -> stress signal
        sleep_hours=5.5,   # short night -> stress signal
        respiratory_rate=17,
        baseline=demo_baseline,
    )
    print(today)
