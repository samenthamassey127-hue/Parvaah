"""
PRAVAAH pilot — shared configuration.

Everything here (station distances, dep/arr times) is a STARTING REFERENCE
ONLY, per the master prompt. The pipeline overwrites these with live values
from the rail-data API on every poll — nothing here should ever be logged
as if it were a verified live number.
"""

import os

# ---------------------------------------------------------------------------
# Route skeleton — identity + coordinates for weather lookups only.
# Distance/scheduled-time fields are left as None; the pipeline fills them
# in from the live API's timetable response (see pipeline.fetch_timetable).
# ---------------------------------------------------------------------------
STATIONS = {
    "LKO":  {"name": "Lucknow (Charbagh)", "lat": 26.85, "lon": 80.95, "km_from_lko": 0},
    "HRI":  {"name": "Hardoi",             "lat": 27.40, "lon": 80.13, "km_from_lko": None},
    "SPN":  {"name": "Shahjahanpur",       "lat": 27.88, "lon": 79.91, "km_from_lko": None},
    "BE":   {"name": "Bareilly",           "lat": 28.35, "lon": 79.43, "km_from_lko": None},
    "RMU":  {"name": "Rampur Jn",          "lat": 28.80, "lon": 79.03, "km_from_lko": None},
    "MB":   {"name": "Moradabad",          "lat": 28.84, "lon": 78.78, "km_from_lko": None},
    "HPU":  {"name": "Hapur Jn",           "lat": 28.73, "lon": 77.78, "km_from_lko": None},
    "GZB":  {"name": "Ghaziabad Jn",       "lat": 28.67, "lon": 77.45, "km_from_lko": None},
    "NDLS": {"name": "New Delhi",          "lat": 28.64, "lon": 77.22, "km_from_lko": None},
}
ROUTE_ORDER = ["LKO", "HRI", "SPN", "BE", "RMU", "MB", "HPU", "GZB", "NDLS"]
ORIGIN, DESTINATION = "LKO", "NDLS"

# ---------------------------------------------------------------------------
# Train roster — approximate times ONLY, per master prompt Section 2.2.
# `active` marks the pilot's initial 1-2 train focus (Section 2, "don't poll
# all eight from day one"). Add the rest once the pilot trains log cleanly.
# ---------------------------------------------------------------------------
TRAIN_ROSTER = {
    "22489": {"name": "Vande Bharat Express",       "tier": "semi-high-speed", "approx_dep": "13:50", "approx_arr": "20:10", "active": True},
    "82501": {"name": "Tejas Express (IRCTC)",      "tier": "premium fast",    "approx_dep": "06:10", "approx_arr": "12:35", "active": False},
    "12003": {"name": "Lucknow Swarna Shatabdi",    "tier": "premium day",     "approx_dep": "15:30", "approx_arr": None,    "active": False},
    "20505": {"name": "AC Superfast Express",       "tier": "AC express",      "approx_dep": "05:55", "approx_arr": "13:38", "active": False},
    "12419": {"name": "Gomti Express",              "tier": "superfast",       "approx_dep": "05:45", "approx_arr": "15:00", "active": False},
    "12229": {"name": "Lucknow Mail",                "tier": "overnight mail",  "approx_dep": "22:00", "approx_arr": "06:55+1", "active": True},
    "12391": {"name": "Shramjeevi Express",          "tier": "overnight superfast", "approx_dep": "20:15", "approx_arr": "04:45+1", "active": False},
    "15733": {"name": "Farakka Express",             "tier": "overnight express", "approx_dep": "18:55", "approx_arr": "06:05+1", "active": False},
}
PILOT_TRAINS = [num for num, t in TRAIN_ROSTER.items() if t["active"]]  # ["22489", "12229"]

# ---------------------------------------------------------------------------
# Paths / API config
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("PRAVAAH_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.environ.get("PRAVAAH_DB_PATH", os.path.join(DATA_DIR, "pravaah.sqlite"))
# Set PRAVAAH_DB_PATH to a separate file (e.g. data/demo_pravaah.sqlite) when
# running demo_data.py, so synthetic rows can NEVER land in the same
# database as real logged pipeline data. See demo_data.py's own warnings.
RESULTS_HISTORY_CSV = os.path.join(DATA_DIR, "results_history.csv")
MODEL_PATH = os.path.join(DATA_DIR, "correction_model.joblib")

RAILRADAR_BASE_URL = os.environ.get("RAILRADAR_BASE_URL", "https://api.railradar.in/v1")
RAILRADAR_API_KEY = os.environ.get("RAILRADAR_API_KEY")  # set this before running pipeline.py for real
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# IST is UTC+5:30, no DST. All timestamps stored in the DB are IST ISO strings
# (see timeutils.now_ist / to_ist) so every train ETA the dashboard shows is
# unambiguously on the IST grid, regardless of what timezone the API/host is in.
IST_OFFSET_MINUTES = 5 * 60 + 30

# Poll cadence (minutes) while a pilot train is between LKO dep and NDLS arr.
# RailRadar's free sandbox tier is 1,000 requests/month. Gated to each pilot
# train's actual active journey window (not 24/7), 2 trains at 20-min
# intervals costs ~1,370 req/month if run every day, or ~965 for a 21-day
# window — under budget for the pilot's 2-3 week collection run, but with
# little margin. Go to 25-30 min, or track 1 train instead of 2, for more
# headroom; see README for the exact math.
POLL_INTERVAL_MIN = 20

# Minutes of buffer before scheduled departure / after scheduled arrival
# during which the pipeline WILL poll (to catch early departures / late
# arrivals). Outside [dep - buffer, arr + buffer], pipeline.py skips the
# RailRadar call entirely rather than spending budget on a train that isn't
# running yet.
ACTIVE_WINDOW_BUFFER_MIN = 30

# Minimum logged (predicted, actual) rows before switching from linear
# regression to Random Forest as the correction model (Section 4.2).
RF_MIN_ROWS = 300

# Conformal prediction target coverage (Section 4.3): 90% interval.
CONFORMAL_ALPHA = 0.10
