"""
Section 4.1 / 5 (first half): the naive baseline and its MAE.

This has to exist and produce a number BEFORE any correction model work,
per the master prompt's build order — it's "the number to beat," not an
afterthought computed after the model already looks good.
"""

import pandas as pd

import storage
from timeutils import IST


def scored_dataframe() -> pd.DataFrame:
    """All logged rows that have a real, joined actual arrival — i.e. every
    row the naive baseline and the model can both be scored against fairly."""
    rows = storage.fetch_scored_logs()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    for col in ("naive_predicted_arrival_ist", "model_predicted_arrival_ist",
                "confidence_low_ist", "confidence_high_ist", "actual_arrival_ist",
                "poll_timestamp_ist"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_convert(IST)
    return df


def naive_mae_minutes(df: pd.DataFrame = None) -> float:
    """MAE, in minutes, of (scheduled time + last known delay) vs. actual
    arrival — computed over logged holdout data, exactly as Section 5
    specifies. Returns None if there isn't a single scored row yet, rather
    than a fake 0.0 that would look like a (very good, very fake) result."""
    df = scored_dataframe() if df is None else df
    if df.empty:
        return None
    valid = df.dropna(subset=["naive_predicted_arrival_ist", "actual_arrival_ist"])
    if valid.empty:
        return None
    errors_min = (valid["actual_arrival_ist"] - valid["naive_predicted_arrival_ist"]).dt.total_seconds().abs() / 60.0
    return float(errors_min.mean())


def naive_mae_report() -> dict:
    df = scored_dataframe()
    mae = naive_mae_minutes(df)
    return {
        "n_scored_rows": 0 if df.empty else len(df.dropna(subset=["naive_predicted_arrival_ist", "actual_arrival_ist"])),
        "naive_mae_min": mae,
    }


if __name__ == "__main__":
    report = naive_mae_report()
    if report["naive_mae_min"] is None:
        print("No scored rows yet (no train has completed a logged run with an actual "
              "arrival). Run the pipeline until at least one full LKO->NDLS run is logged.")
    else:
        print(f"Naive baseline MAE over {report['n_scored_rows']} scored rows: "
              f"{report['naive_mae_min']:.1f} minutes")
