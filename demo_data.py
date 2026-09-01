"""
================================================================================
 DEMO / SYNTHETIC DATA ONLY — NOT REAL LOGGED DATA
================================================================================
This module exists for ONE reason: to let you smoke-test naive_baseline.py,
correction_model.py, accuracy_harness.py, and dashboard.py end-to-end TODAY,
before you have real logged rows from pipeline.py. It has nothing to do with
answering "does this system actually predict better than naive." That
question can ONLY be answered from real data logged by pipeline.py against
the real RailRadar/OpenWeatherMap APIs over the real 2-3 week window.

Safety rail: this script REFUSES to write into config.DB_PATH unless you
explicitly set PRAVAAH_DB_PATH to a path containing the word "demo" first.
That's a deliberate speed bump, not a suggestion — synthetic and real rows
must never end up in the same database, or every number the harness prints
afterwards becomes unfalsifiable.

Usage:
    PRAVAAH_DB_PATH=data/demo_pravaah.sqlite python demo_data.py --days 60
    PRAVAAH_DB_PATH=data/demo_pravaah.sqlite python accuracy_harness.py
================================================================================
"""

import argparse
import random
from datetime import timedelta

import numpy as np

import config
import storage
from timeutils import combine_service_date_and_hhmm, iso_ist

TOTAL_DISTANCE_KM = 497.0
# rough progress fractions along ROUTE_ORDER, for interpolating "last station"
_STATION_FRACTIONS = {code: i / (len(config.ROUTE_ORDER) - 1) for i, code in enumerate(config.ROUTE_ORDER)}


def _assert_demo_db():
    if "demo" not in config.DB_PATH.lower():
        raise RuntimeError(
            "Refusing to run: PRAVAAH_DB_PATH does not contain 'demo'.\n"
            "Set it explicitly, e.g.:\n"
            "  PRAVAAH_DB_PATH=data/demo_pravaah.sqlite python demo_data.py\n"
            "This guard exists so synthetic rows can never contaminate the real "
            "logged dataset the pilot's accuracy claim depends on."
        )


def _nearest_station_for_progress(frac: float) -> str:
    best_code, best_diff = config.ORIGIN, 1.0
    for code, f in _STATION_FRACTIONS.items():
        if abs(f - frac) < best_diff:
            best_code, best_diff = code, abs(f - frac)
    return best_code


def _is_winter(month: int) -> bool:
    return month in (12, 1)  # Dec-Jan fog season, as called out in Section 3.2


def simulate_journey(rng: random.Random, np_rng: np.random.Generator, train_number: str, service_date_str: str):
    """Simulates one LKO->NDLS run and returns a list of poll rows plus the
    true actual arrival. The simulated 'physics': a base random delay, PLUS
    a fog penalty that only shows up as low visibility on the intermediate
    stations and only partially reflected in the delay reported so far — this
    is what gives a correction model something real to learn versus naive
    (which only ever sees 'delay so far', not 'visibility right now')."""
    train_info = config.TRAIN_ROSTER[train_number]
    sched_dep = combine_service_date_and_hhmm(service_date_str, train_info["approx_dep"])
    sched_arr = combine_service_date_and_hhmm(service_date_str, train_info["approx_arr"])
    month = sched_dep.month

    base_delay = max(0.0, np_rng.normal(loc=8, scale=12))
    fog_penalty = 0.0
    if _is_winter(month) and sched_dep.hour <= 9 or (_is_winter(month) and sched_arr.hour <= 9):
        # overnight/early-morning trains in winter catch the fog window
        if rng.random() < 0.55:
            fog_penalty = max(0.0, np_rng.normal(loc=45, scale=25))
    true_final_delay_min = base_delay + fog_penalty

    n_polls = rng.randint(8, 14)
    rows = []
    for i in range(n_polls):
        frac = min(0.98, (i + 1) / n_polls)
        poll_time = sched_dep + timedelta(seconds=frac * (sched_arr - sched_dep).total_seconds() * 0.9)
        last_code = _nearest_station_for_progress(frac)
        distance_remaining = TOTAL_DISTANCE_KM * (1 - frac)

        # delay "so far" grows towards true_final_delay but the fog portion
        # of it is disproportionately back-loaded (fog often gets WORSE deep
        # into the run, e.g. near Moradabad/Hapur in the early hours) —
        # so a mid-journey report understates where things end up.
        reported_so_far = base_delay * frac + fog_penalty * (frac ** 2.2) + np_rng.normal(0, 4)
        reported_so_far = max(0.0, reported_so_far)

        if _is_winter(month) and (poll_time.hour <= 8 or poll_time.hour >= 22) and fog_penalty > 0:
            visibility_km = max(0.05, np_rng.normal(loc=1.2, scale=0.8))
            precip_mm = 0.0
        elif rng.random() < 0.08:
            visibility_km = max(1.0, np_rng.normal(loc=4.0, scale=1.5))
            precip_mm = max(0.0, np_rng.normal(loc=2.0, scale=1.5))
        else:
            visibility_km = max(3.0, np_rng.normal(loc=9.0, scale=1.5))
            precip_mm = 0.0

        naive_pred = sched_arr + timedelta(minutes=reported_so_far)

        rows.append({
            "poll_timestamp_ist": iso_ist(poll_time),
            "train_number": train_number,
            "service_date": service_date_str,
            "last_station_code": last_code,
            "distance_remaining_km": round(distance_remaining, 1),
            "reported_delay_min": round(reported_so_far, 1),
            "weather_visibility_km": round(visibility_km, 2),
            "weather_precip_mm": round(precip_mm, 2),
            "naive_predicted_arrival_ist": iso_ist(naive_pred),
            "model_predicted_arrival_ist": None,   # filled in later once a model exists
            "confidence_low_ist": None,
            "confidence_high_ist": None,
            "api_last_update_ist": iso_ist(poll_time),
            "actual_arrival_ist": None,             # joined below, all at once
        })

    actual_arrival = sched_arr + timedelta(minutes=true_final_delay_min)
    return rows, actual_arrival


def generate(days: int = 60, seed: int = 7):
    _assert_demo_db()
    storage.init_db()
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    from timeutils import now_ist
    start = now_ist() - timedelta(days=days)
    total_rows = 0
    for d in range(days):
        service_date = (start + timedelta(days=d)).strftime("%Y-%m-%d")
        for train_number in config.PILOT_TRAINS:
            rows, actual_arrival = simulate_journey(rng, np_rng, train_number, service_date)
            for r in rows:
                storage.insert_eta_log(r)
            storage.record_actual_arrival(train_number, service_date, iso_ist(actual_arrival))
            total_rows += len(rows)
    print(f"[DEMO DATA] Wrote {total_rows} synthetic rows across {days} days x "
          f"{len(config.PILOT_TRAINS)} trains into {config.DB_PATH}")
    print("[DEMO DATA] Remember: this is fake. Delete this DB before real logging starts, "
          "and never point the dashboard's live view at it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic demo data (NOT real).")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    generate(args.days, args.seed)
