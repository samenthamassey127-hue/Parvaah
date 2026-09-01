"""
Section 5 — the actual point of the pilot.

Computes, on real logged holdout data (never on training rows):
  - naive baseline MAE
  - correction model MAE
  - improvement %
  - conformal interval coverage (does the reported interval actually contain
    the true arrival at the rate it claims to?)

Appends one row per run to config.RESULTS_HISTORY_CSV so the dashboard's
"Model performance" tab has a re-runnable history, not a one-off number.

Explicitly refuses to report a model MAE or coverage number when there
isn't a trained model / enough scored data yet, rather than printing a
misleadingly clean 0.0 or 100% — an unmeasured number is not a result.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from naive_baseline import scored_dataframe, naive_mae_minutes
from correction_model import build_training_frame, FEATURE_COLS, load_model, predict_arrival


def _model_predictions_on_holdout(holdout_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    preds, lows, highs = [], [], []
    for _, row in holdout_df.iterrows():
        features = {c: row[c] for c in FEATURE_COLS}
        point, low, high = predict_arrival(bundle, row["naive_predicted_arrival_ist"], features)
        preds.append(point)
        lows.append(low)
        highs.append(high)
    out = holdout_df.copy()
    out["harness_model_pred"] = preds
    out["harness_conf_low"] = lows
    out["harness_conf_high"] = highs
    return out


def run_harness(holdout_frac: float = 0.3, seed: int = 42) -> dict:
    df = scored_dataframe()
    result = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_scored_rows": 0 if df.empty else len(df),
        "naive_mae_min": None,
        "model_mae_min": None,
        "improvement_pct": None,
        "conformal_target_coverage": 1 - config.CONFORMAL_ALPHA,
        "conformal_observed_coverage": None,
        "n_holdout_rows": 0,
        "status": None,
    }

    if df.empty:
        result["status"] = "NO_DATA — nothing logged yet. Run pipeline.py first."
        _append_result(result)
        return result

    train_frame = build_training_frame(df)
    if len(train_frame) < 10:
        result["naive_mae_min"] = naive_mae_minutes(df)
        result["status"] = (f"NAIVE_ONLY — only {len(train_frame)} usable scored rows; "
                             f"not enough to hold out a fair test split for the model yet.")
        _append_result(result)
        return result

    # Holdout split is by ROW here for simplicity; if/when you have enough
    # completed runs, switch this to holding out entire (train_number,
    # service_date) journeys instead, so the model is never scored on a
    # journey it partially saw during training.
    _, holdout_df = train_test_split(train_frame, test_size=holdout_frac, random_state=seed)

    result["n_holdout_rows"] = len(holdout_df)
    result["naive_mae_min"] = float(
        (holdout_df["actual_arrival_ist"] - holdout_df["naive_predicted_arrival_ist"])
        .dt.total_seconds().abs().div(60.0).mean()
    )

    try:
        bundle = load_model()
    except FileNotFoundError:
        result["status"] = "NO_TRAINED_MODEL — run correction_model.py to train one first."
        _append_result(result)
        return result

    scored = _model_predictions_on_holdout(holdout_df, bundle)
    scored["model_error_min"] = (
        (scored["actual_arrival_ist"] - scored["harness_model_pred"]).dt.total_seconds().abs() / 60.0
    )
    result["model_mae_min"] = float(scored["model_error_min"].mean())
    result["improvement_pct"] = 100.0 * (result["naive_mae_min"] - result["model_mae_min"]) / result["naive_mae_min"]

    within = (
        (scored["actual_arrival_ist"] >= scored["harness_conf_low"])
        & (scored["actual_arrival_ist"] <= scored["harness_conf_high"])
    )
    result["conformal_observed_coverage"] = float(within.mean())
    result["status"] = "OK"

    _append_result(result)
    return result


def _append_result(result: dict):
    row_df = pd.DataFrame([result])
    header = not os.path.exists(config.RESULTS_HISTORY_CSV)
    row_df.to_csv(config.RESULTS_HISTORY_CSV, mode="a", header=header, index=False)


def latest_result() -> dict:
    if not os.path.exists(config.RESULTS_HISTORY_CSV):
        return {"status": "NO_RUNS_YET"}
    df = pd.read_csv(config.RESULTS_HISTORY_CSV)
    return df.iloc[-1].to_dict()


def results_history() -> pd.DataFrame:
    if not os.path.exists(config.RESULTS_HISTORY_CSV):
        return pd.DataFrame()
    return pd.read_csv(config.RESULTS_HISTORY_CSV)


if __name__ == "__main__":
    r = run_harness()
    print(f"Status: {r['status']}")
    print(f"Scored rows in DB: {r['n_scored_rows']} | Holdout rows this run: {r['n_holdout_rows']}")
    if r["naive_mae_min"] is not None:
        print(f"Naive MAE:  {r['naive_mae_min']:.1f} min")
    if r["model_mae_min"] is not None:
        print(f"Model MAE:  {r['model_mae_min']:.1f} min")
    if r["improvement_pct"] is not None:
        print(f"Improvement: {r['improvement_pct']:.1f}%  "
              f"(published Indian-Railways ML studies report ~20-30% as a realistic range — "
              f"treat that as a comparison point, not a target to force)")
    if r["conformal_observed_coverage"] is not None:
        print(f"{int(r['conformal_target_coverage']*100)}% interval — observed coverage: "
              f"{r['conformal_observed_coverage']*100:.1f}%")
