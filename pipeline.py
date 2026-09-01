"""
Section 3: the data pipeline. This is the FIRST thing to get running per
the master prompt's build order — nothing downstream can be evaluated
without real logged rows.

Field parsing below matches RailRadar's actually-documented response shape
(https://railradar.in/docs/live-train-status) as of Aug 2026 — verify
against their current docs before relying on this in production, since
public API responses do drift.

Requires (set as environment variables before running):
    RAILRADAR_API_KEY   (from https://railradar.in — free sandbox, no card)
    OPENWEATHER_API_KEY (from https://openweathermap.org/api — free tier)

Run standalone for a quick one-off poll of the pilot trains:
    python pipeline.py --once

Run the long-lived loop (2-3 week logging window per Section 2.2):
    python pipeline.py --loop
"""

import argparse
import time
import sys
from datetime import timedelta

import requests

import config
import storage
from timeutils import now_ist, iso_ist, parse_api_timestamp, combine_service_date_and_hhmm

try:
    from correction_model import load_model, predict_arrival
    _MODEL_AVAILABLE = True
except Exception:
    _MODEL_AVAILABLE = False


# ---------------------------------------------------------------------------
# RailRadar client
# ---------------------------------------------------------------------------
class RailRadarClient:
    """Thin wrapper over RailRadar's documented REST API (Section 3.1).
    Every field pulled out of the response below is named exactly as in
    their docs' response example — if RailRadar changes their schema,
    update the extraction in get_live_status(), not the rest of the
    pipeline, which only ever sees this class's normalized return dict."""

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or config.RAILRADAR_API_KEY
        self.base_url = base_url or config.RAILRADAR_BASE_URL
        if not self.api_key:
            raise RuntimeError(
                "RAILRADAR_API_KEY not set. This client will not fabricate "
                "live train data — get a free sandbox key at railradar.in "
                "and set the env var."
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def get_live_status(self, train_number: str, service_date: str = None) -> dict:
        """GET /v1/trains/{number}/live

        `service_date` is optional — omitting it lets RailRadar auto-detect
        the current run from IST time, which is more reliable than us
        guessing "today" for an overnight train that departed yesterday.
        The authoritative service date actually used comes back as
        data['startDate'] and is what the caller should log against.
        """
        params = {}
        if service_date:
            params["date"] = service_date
        resp = self.session.get(
            f"{self.base_url}/trains/{train_number}/live", params=params, timeout=10
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("success"):
            raise RuntimeError(f"RailRadar API error for {train_number}: {payload.get('error')}")
        data = payload["data"]

        route = data.get("route", [])
        current = data.get("currentLocation", {}) or {}
        last_code = current.get("stationCode")

        # Locate the destination stop in the returned route array to read
        # its scheduled/actual arrival directly, rather than deriving it
        # from a static config table — this is the "let the live API's
        # numbers override the reference table" behaviour the master
        # prompt asks for.
        dest_entry = next((s for s in route if s.get("stationCode") == config.DESTINATION), None)
        scheduled_arrival_raw = dest_entry["scheduledArrival"] if dest_entry else None
        actual_arrival_raw = dest_entry.get("actualArrival") if dest_entry else None

        distance_remaining_km = self._distance_remaining(data, route, current)

        return {
            "status": data.get("status"),                       # running | not-started | completed | cancelled
            "last_station_code": last_code,
            "distance_remaining_km": distance_remaining_km,
            "reported_delay_min": data.get("delayMinutes"),
            "scheduled_arrival_raw": scheduled_arrival_raw,      # ISO string, already correctly date-rolled
            "actual_arrival_raw": actual_arrival_raw,            # ISO string once the train has actually arrived, else None
            "api_last_update_raw": data.get("lastUpdatedAt"),
            "start_date": data.get("startDate"),                 # authoritative service_date per RailRadar
        }

    @staticmethod
    def _distance_remaining(data: dict, route: list, current: dict):
        total_km = (data.get("train") or {}).get("distance")
        if total_km is None or not route:
            return None
        last_code = current.get("stationCode")
        seg_progress = current.get("segmentProgress") or 0.0
        idx = next((i for i, s in enumerate(route) if s.get("stationCode") == last_code), None)
        if idx is None:
            return None
        current_km = route[idx].get("distance", 0) or 0
        if idx + 1 < len(route):
            next_km = route[idx + 1].get("distance", current_km) or current_km
            current_km = current_km + seg_progress * (next_km - current_km)
        return round(max(0.0, total_km - current_km), 1)

    def get_timetable(self, train_number: str) -> dict:
        """GET /v1/trains/{number} — full scheduled timetable. Not called on
        every poll (would burn budget for no benefit, since the live
        endpoint's own route[] already carries scheduledArrival); useful if
        you want to independently verify config.py's approximate dep/arr
        times before your first real poll of a new train."""
        resp = self.session.get(f"{self.base_url}/trains/{train_number}", timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("success"):
            raise RuntimeError(f"RailRadar API error for {train_number}: {payload.get('error')}")
        return payload["data"]


# ---------------------------------------------------------------------------
# Weather client
# ---------------------------------------------------------------------------
class WeatherClient:
    """OpenWeatherMap wrapper (Section 3.2). Fog/visibility is pulled out
    explicitly because it's flagged as the single most important
    India-specific feature for this corridor. OWM's free tier (60 calls/min,
    1,000,000/month) is generous relative to RailRadar's — it is not the
    binding budget constraint on poll frequency; RailRadar is."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.OPENWEATHER_API_KEY
        if not self.api_key:
            raise RuntimeError("OPENWEATHER_API_KEY not set.")

    def get_conditions(self, lat: float, lon: float) -> dict:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        visibility_m = data.get("visibility")  # OWM returns meters, max 10000
        precip_mm = 0.0
        if "rain" in data:
            precip_mm += data["rain"].get("1h", 0.0)
        if "snow" in data:
            precip_mm += data["snow"].get("1h", 0.0)
        return {
            "visibility_km": (visibility_m / 1000.0) if visibility_m is not None else None,
            "precip_mm": precip_mm,
        }


# ---------------------------------------------------------------------------
# Budget-conscious active-window gating
# ---------------------------------------------------------------------------
def _approx_active_window(train_number: str, service_date: str):
    """A coarse window, from config.py's APPROXIMATE dep/arr times only —
    used purely to decide WHETHER to spend a RailRadar call, never to log
    or predict anything. Once we do call the live API, its own numbers are
    authoritative everywhere else in the pipeline."""
    info = config.TRAIN_ROSTER[train_number]
    dep = combine_service_date_and_hhmm(service_date, info["approx_dep"])
    arr_field = info["approx_arr"] or info["approx_dep"]  # fallback if unknown
    arr = combine_service_date_and_hhmm(service_date, arr_field if isinstance(arr_field, str) and ":" in arr_field else "23:59")
    buf = timedelta(minutes=config.ACTIVE_WINDOW_BUFFER_MIN)
    return dep - buf, arr + buf


def _should_poll_now(train_number: str, service_date: str) -> bool:
    if storage.has_actual_arrival(train_number, service_date):
        return False  # already logged this run's arrival — don't spend budget re-checking it
    start, end = _approx_active_window(train_number, service_date)
    return start <= now_ist() <= end


# ---------------------------------------------------------------------------
# Polling logic
# ---------------------------------------------------------------------------
def poll_train(rr_client: RailRadarClient, wx_client: WeatherClient, train_number: str):
    """One poll of one train: fetch live status + weather at its current
    position, compute naive + (if trained) model predictions, log the full
    Section 3.4 row. `date` is intentionally NOT forced here — RailRadar
    auto-detects the current run and tells us the correct service_date back,
    which matters most for overnight trains like the Lucknow Mail."""
    status = rr_client.get_live_status(train_number)
    service_date = status["start_date"] or now_ist().strftime("%Y-%m-%d")

    last_code = status["last_station_code"]
    station = config.STATIONS.get(last_code, config.STATIONS[config.ORIGIN])
    wx = wx_client.get_conditions(station["lat"], station["lon"])

    if not status["scheduled_arrival_raw"]:
        print(f"[{train_number}] no destination stop found in live route yet — skipping this poll",
              file=sys.stderr)
        return None

    scheduled_arrival_dt = parse_api_timestamp(status["scheduled_arrival_raw"])
    naive_pred = scheduled_arrival_dt + timedelta(minutes=status["reported_delay_min"] or 0)

    model_pred_ist, conf_low_ist, conf_high_ist = None, None, None
    if _MODEL_AVAILABLE:
        try:
            model = load_model()
            features = {
                "reported_delay_min": status["reported_delay_min"] or 0,
                "distance_remaining_km": status["distance_remaining_km"] or 0,
                "weather_visibility_km": wx["visibility_km"] if wx["visibility_km"] is not None else 10.0,
                "hour_of_day": now_ist().hour,
            }
            pred_dt, low_dt, high_dt = predict_arrival(model, naive_pred, features)
            model_pred_ist, conf_low_ist, conf_high_ist = iso_ist(pred_dt), iso_ist(low_dt), iso_ist(high_dt)
        except FileNotFoundError:
            pass  # no trained model yet — expected in the pipeline's early days

    row = {
        "poll_timestamp_ist": iso_ist(now_ist()),
        "train_number": train_number,
        "service_date": service_date,
        "last_station_code": last_code,
        "distance_remaining_km": status["distance_remaining_km"],
        "reported_delay_min": status["reported_delay_min"],
        "weather_visibility_km": wx["visibility_km"],
        "weather_precip_mm": wx["precip_mm"],
        "naive_predicted_arrival_ist": iso_ist(naive_pred),
        "model_predicted_arrival_ist": model_pred_ist,
        "confidence_low_ist": conf_low_ist,
        "confidence_high_ist": conf_high_ist,
        "api_last_update_ist": iso_ist(parse_api_timestamp(status["api_last_update_raw"])) if status["api_last_update_raw"] else None,
        "actual_arrival_ist": None,
    }
    storage.insert_eta_log(row)

    # Ground truth from RailRadar's OWN record of the actual arrival at NDLS
    # (dest_entry.actualArrival), not a guess based on when we happened to
    # poll — more precise, and it's what the join in Section 3.4 needs.
    if status["actual_arrival_raw"]:
        actual_dt = parse_api_timestamp(status["actual_arrival_raw"])
        storage.record_actual_arrival(train_number, service_date, iso_ist(actual_dt))

    return row


def poll_once(train_numbers=None, respect_budget_gate: bool = True):
    storage.init_db()
    rr = RailRadarClient()
    wx = WeatherClient()
    train_numbers = train_numbers or config.PILOT_TRAINS
    service_date_guess = now_ist().strftime("%Y-%m-%d")
    results = []
    for tn in train_numbers:
        if respect_budget_gate and not _should_poll_now(tn, service_date_guess):
            continue
        try:
            row = poll_train(rr, wx, tn)
            if row:
                results.append(row)
        except requests.HTTPError as e:
            print(f"[{tn}] API error, skipping this poll: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[{tn}] unexpected error, skipping this poll: {e}", file=sys.stderr)
    return results


def run_loop(interval_min: int = config.POLL_INTERVAL_MIN):
    """Sleep loop, per Section 7. Skips (does not call RailRadar for) any
    train outside its own active journey window, so a long-running loop
    doesn't burn the free sandbox's 1,000 req/month sitting idle overnight.
    See config.ACTIVE_WINDOW_BUFFER_MIN / README for the request-budget math.
    """
    print(f"PRAVAAH pipeline: checking {config.PILOT_TRAINS} every {interval_min} min "
          f"(RailRadar only called while a train is in its active journey window). Ctrl+C to stop.")
    while True:
        rows = poll_once()
        if rows:
            print(f"{now_ist().strftime('%H:%M:%S')} IST — logged {len(rows)} row(s): "
                  f"{[r['train_number'] for r in rows]}")
        time.sleep(interval_min * 60)

    # --- APScheduler variant (uncomment to use instead of the loop above,
    #     useful if this needs to keep running longer than one script call) ---
    # from apscheduler.schedulers.blocking import BlockingScheduler
    # sched = BlockingScheduler(timezone="Asia/Kolkata")
    # sched.add_job(poll_once, "interval", minutes=interval_min)
    # sched.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRAVAAH LKO-NDLS data pipeline")
    parser.add_argument("--once", action="store_true", help="poll all pilot trains one time and exit")
    parser.add_argument("--loop", action="store_true", help="poll continuously (2-3 week collection window)")
    parser.add_argument("--interval", type=int, default=config.POLL_INTERVAL_MIN)
    parser.add_argument("--ignore-budget-gate", action="store_true",
                         help="poll even outside the approximate active window (uses more API budget; "
                              "mainly useful for a single manual test poll)")
    args = parser.parse_args()

    if args.loop:
        run_loop(args.interval)
    else:
        storage.init_db()
        rows = poll_once(respect_budget_gate=not args.ignore_budget_gate)
        print(f"Logged {len(rows)} row(s).")
