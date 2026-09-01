"""
Section 4.2 (correction model) + 4.3 (conformal intervals).

Design choice, spelled out rather than hidden in code comments: the model
does NOT predict absolute arrival time directly. It predicts the RESIDUAL
(actual_arrival - naive_predicted_arrival, in minutes) and the final ETA is
naive_predicted + predicted_residual. This keeps the model's job small and
well-scoped (predict "how wrong is the naive guess, in minutes, given these
conditions") rather than re-deriving the whole schedule, which matters a lot
at pilot data volumes (a few hundred rows).

Model selection follows the master prompt exactly:
  - fewer than RF_MIN_ROWS logged rows -> LinearRegression on 3-4 features
  - RF_MIN_ROWS or more -> RandomForestRegressor

Conformal calibration: tries `mapie` first (as the master prompt names it),
falls back to a manual split-conformal implementation if mapie isn't
installed in your environment — the two are mathematically the same
procedure (absolute residuals on a held-out calibration split -> take the
(1-alpha) quantile -> +/- that around the point prediction), so falling back
does not change what guarantee you're getting.
"""

import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import config
from naive_baseline import scored_dataframe

FEATURE_COLS = ["reported_delay_min", "distance_remaining_km", "weather_visibility_km", "hour_of_day"]


def build_training_frame(df: pd.DataFrame = None) -> pd.DataFrame:
    df = scored_dataframe() if df is None else df
    if df.empty:
        return df
    df = df.dropna(subset=["naive_predicted_arrival_ist", "actual_arrival_ist"]).copy()
    df["hour_of_day"] = pd.to_datetime(df["poll_timestamp_ist"]).dt.hour
    df["weather_visibility_km"] = df["weather_visibility_km"].fillna(10.0)  # clear-day default
    df["residual_min"] = (
        (df["actual_arrival_ist"] - df["naive_predicted_arrival_ist"]).dt.total_seconds() / 60.0
    )
    return df.dropna(subset=FEATURE_COLS + ["residual_min"])


def train_correction_model(df: pd.DataFrame = None, min_calibration_rows: int = 20):
    """Returns a model bundle dict, or None if there isn't enough data yet.
    Also WRITES that bundle to config.MODEL_PATH so pipeline.py's next poll
    can use it — but only if training actually produced something usable."""
    train_df = build_training_frame(df)
    n = len(train_df)
    if n < min_calibration_rows * 2:
        print(f"Only {n} usable scored rows — need at least {min_calibration_rows * 2} "
              f"(some for training, some for conformal calibration) before a correction "
              f"model can be trained responsibly. Falling back to naive baseline only.")
        return None

    # Split BEFORE fitting: one part to fit the regressor, a separate,
    # never-fitted-on part to calibrate the conformal interval. Calibrating
    # on training-set residuals would understate the interval — that's the
    # one shortcut this pilot explicitly should not take.
    fit_df, cal_df = train_test_split(train_df, test_size=0.3, random_state=42)

    X_fit, y_fit = fit_df[FEATURE_COLS].values, fit_df["residual_min"].values
    X_cal, y_cal = cal_df[FEATURE_COLS].values, cal_df["residual_min"].values

    if n >= config.RF_MIN_ROWS:
        model = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
        model_type = "random_forest"
    else:
        model = LinearRegression()
        model_type = "linear"
    model.fit(X_fit, y_fit)

    conformal_quantile_min = _split_conformal_quantile(model, X_cal, y_cal, alpha=config.CONFORMAL_ALPHA)

    bundle = {
        "model": model,
        "model_type": model_type,
        "feature_cols": FEATURE_COLS,
        "conformal_quantile_min": conformal_quantile_min,
        "conformal_alpha": config.CONFORMAL_ALPHA,
        "n_train_rows": n,
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    joblib.dump(bundle, config.MODEL_PATH)
    return bundle


def _split_conformal_quantile(model, X_cal, y_cal, alpha: float) -> float:
    """Split conformal prediction (Section 4.3): nonconformity score is the
    absolute residual-of-the-residual on held-out calibration data; the
    interval half-width is the (1-alpha) empirical quantile of those scores.
    This is the same math `mapie.regression.MapieRegressor(method='base',
    cv='split')` runs — using it directly here means one fewer dependency
    for a pilot at this data scale, with an identical guarantee."""
    try:
        # If mapie IS installed, prefer it for the calibration step so the
        # guarantee is coming from an audited library, not homegrown code.
        from mapie.regression import MapieRegressor
        mapie_model = MapieRegressor(estimator=model, cv="prefit")
        mapie_model.fit(X_cal, y_cal)
        _, y_pis = mapie_model.predict(X_cal, alpha=alpha)
        half_widths = (y_pis[:, 1, 0] - y_pis[:, 0, 0]) / 2.0
        return float(np.median(half_widths))
    except ImportError:
        preds = model.predict(X_cal)
        abs_resid = np.abs(y_cal - preds)
        n = len(abs_resid)
        # finite-sample-corrected quantile level, standard split-conformal adjustment
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(abs_resid, q_level))


def load_model() -> dict:
    return joblib.load(config.MODEL_PATH)  # raises FileNotFoundError if untrained — caller handles it


def predict_arrival(bundle: dict, naive_pred_dt, features: dict):
    """Returns (point_prediction_dt, low_dt, high_dt)."""
    x = np.array([[features[c] for c in bundle["feature_cols"]]])
    residual_min = float(bundle["model"].predict(x)[0])
    point = naive_pred_dt + timedelta(minutes=residual_min)
    half_width = bundle["conformal_quantile_min"]
    low = point - timedelta(minutes=half_width)
    high = point + timedelta(minutes=half_width)
    return point, low, high


if __name__ == "__main__":
    bundle = train_correction_model()
    if bundle:
        print(f"Trained {bundle['model_type']} on {bundle['n_train_rows']} rows. "
              f"{int((1 - bundle['conformal_alpha']) * 100)}% conformal half-width: "
              f"{bundle['conformal_quantile_min']:.1f} min. Saved to {config.MODEL_PATH}")
