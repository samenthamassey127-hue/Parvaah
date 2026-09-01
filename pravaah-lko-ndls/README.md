# PRAVAAH — Lucknow–New Delhi ETA Accuracy Pilot

Real-data pilot for one corridor: does this system predict LKO→NDLS arrivals
more accurately than "scheduled time + last known delay"? Every file here
exists to answer that, in the order below.

## A constraint worth stating up front

This code was built in a sandboxed environment **with no live network
access**, so it has been tested with pandas/scikit-learn on **synthetic
demo data only** (`demo_data.py`) — the naive-baseline, model-training, and
harness logic all run end-to-end and produce sane numbers on that synthetic
set. `pipeline.py` and `dashboard.py` (which need `requests`/live APIs and
`dash`/`dash-leaflet` respectively) are written against documented, real
APIs but haven't been run against a live network from here. You'll run the
real thing in your own environment (Colab, per the original tech stack) with
real API keys. That's not a shortcut — it's the same thing Section 1 asks
for: nothing here should be presented as a measured result until it's run
on real logged data by you.

## Setup

```bash
pip install -r requirements.txt
export RAILRADAR_API_KEY=your_key_here
export OPENWEATHER_API_KEY=your_key_here
```

## Putting this in your own GitHub repo

```bash
# unzip this download, then from inside the folder:
git init
git add .
git commit -m "PRAVAAH LKO-NDLS pilot: pipeline, baseline, model, harness, dashboard"

# create an empty repo on github.com first (no README/license, so there's
# no merge conflict), then:
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

`.gitignore` already excludes `data/` (your logged SQLite DB, trained model,
results CSV) and any `.env` file — those are real measured data / secrets,
not source, and shouldn't go in the repo. To use this from Colab afterward:
```python
!git clone https://github.com/<your-username>/<your-repo-name>.git
%cd <your-repo-name>
!pip install -r requirements.txt -q
```
then continue with the API-key/Drive-mount steps below.

## Running this in Google Colab (live data)

### 1. Get your two API keys
- **RailRadar**: go to railradar.in → Sign up (free, no card) → dashboard
  gives you a key like `rr_live_xxxxxxxx`. Free sandbox = **1,000 requests/
  month** — see the budget note below, this is the binding constraint on
  how often you can poll.
- **OpenWeatherMap**: openweathermap.org/api → free tier API key. Its quota
  (60 calls/min, 1M/month) is generous and not the bottleneck here.

### 2. Get the code into Colab
Either:
```python
!unzip pravaah_pilot.zip   # after uploading the zip via the Colab file pane
%cd pravaah_pilot
```
or push it to a GitHub repo first and `!git clone` it — nicer if you want
version history on your logged data too.

### 3. Install dependencies
```python
!pip install -r requirements.txt -q
```

### 4. Set your API keys as Colab **Secrets**, not plaintext in a cell
Click the key icon in Colab's left sidebar → add `RAILRADAR_API_KEY` and
`OPENWEATHER_API_KEY` → toggle "Notebook access" on. Then:
```python
import os
from google.colab import userdata
os.environ["RAILRADAR_API_KEY"] = userdata.get("RAILRADAR_API_KEY")
os.environ["OPENWEATHER_API_KEY"] = userdata.get("OPENWEATHER_API_KEY")
```
This keeps your keys out of the notebook file itself (which matters if you
ever share or commit the notebook).

### 5. Mount Drive so your logged data survives across sessions
Colab's local disk is wiped every time the runtime disconnects — and for a
2-3 week collection window, it *will* disconnect. Point the DB at Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
os.environ["PRAVAAH_DATA_DIR"] = "/content/drive/MyDrive/pravaah_data"
```
Do this **before** importing `config` / running anything else, since
`config.py` reads `PRAVAAH_DATA_DIR` at import time.

### 6. Test one poll
```python
!python pipeline.py --once
```
You should see `Logged N row(s).` — if `N` is 0, it's most likely because
neither pilot train (22489, 12229) is in its scheduled journey window right
now, which is correct, budget-conserving behavior, not a bug. Use
`--ignore-budget-gate` to force a test poll outside the window if you just
want to confirm the API keys work.

### 7. The part that needs an honest caveat: continuous polling for 2-3 weeks
```python
!python pipeline.py --loop
```
This cell will run for as long as the Colab runtime stays connected — and a
free-tier Colab runtime disconnects after ~90 min idle, or ~12 hours
regardless, and immediately if the browser tab closes. It is **not** a
reliable way to run an unattended multi-week job. Two honest options:

- **Manual/best-effort**: re-open the notebook and re-run `--loop` (or
  `--once` a few times a day) whenever you can, over the 2-3 week window.
  You'll get partial, gappy coverage — usable, just not complete.
- **Actually unattended (recommended)**: run the polling loop somewhere
  that stays on independent of your browser — a free-tier GitHub Actions
  scheduled workflow (`cron`, every 20-30 min) calling
  `python pipeline.py --once --ignore-budget-gate` (do the active-window
  gating check yourself in the workflow's schedule instead, or leave the
  gate on — either works) and committing the growing `data/pravaah.sqlite`
  back to the repo, or a small always-on free-tier VM. Then use Colab
  purely for `correction_model.py` / `accuracy_harness.py` / `dashboard.py`
  — reading whatever's in the DB (pull from the repo, or read straight from
  the same Drive folder if you mount it from both places).

Either way, whatever's actually landing in `data/pravaah.sqlite` is what
`accuracy_harness.py` will honestly score — gaps just mean fewer scored
rows, not wrong ones.

### Request-budget math (2 trains, RailRadar free sandbox)
Gated to each train's actual journey window (not 24/7):

| Poll interval | Requests/day (2 trains) | Requests/month | Under 1,000/mo? |
|---|---|---|---|
| 10 min | ~92 | ~2,750 | No |
| 15 min | ~61 | ~1,830 | No |
| 20 min | ~46 | ~1,370 | Tight — OK for a 21-day window (~965), not a full month |
| 25 min | ~37 | ~1,100 | Close |
| 30 min | ~30 | ~915 | Comfortable |

`config.POLL_INTERVAL_MIN` defaults to 20. If you want more margin, either
raise the interval to 25-30, or track just one train (per Section 2.2's
"don't poll all eight from day one" — the same logic applies to going from
2 down to 1 if budget is tight).

## Run order (matches the master prompt's build order exactly — don't skip ahead)

### 1. Logging pipeline (Section 3) — do this first, for real, for 2-3 weeks
```bash
python pipeline.py --once          # one poll of the two pilot trains (22489, 12229), sanity check
python pipeline.py --loop          # continuous polling every 15 min, per Section 2.2's collection window
```
Rows land in `data/pravaah.sqlite` (`eta_logs` table), matching Section 3.4's
schema exactly. When a pilot train's `last_station_code` reaches `NDLS`, the
pipeline automatically joins the actual arrival back onto every earlier
prediction for that run — that join is what makes the row "scoreable."

### 2. Naive baseline + its MAE (Section 4.1 / 5) — the number to beat
```bash
python naive_baseline.py
```
Prints `None` / an honest "no scored rows yet" message until at least one
full run has actually completed and been logged. It will not print a fake
number to look finished early.

### 3. Correction model + conformal calibration (Section 4.2/4.3) — once you have rows
```bash
python correction_model.py
```
Trains `LinearRegression` on 3-4 features below `RF_MIN_ROWS` (300) usable
rows, `RandomForestRegressor` at/above it — this is decided automatically
from your actual row count, not hardcoded. Calibrates a 90% conformal
interval via `mapie` if installed, else a manual split-conformal fallback
(identical math, see the docstring in `correction_model.py`). Saves to
`data/correction_model.joblib`; `pipeline.py`'s next poll picks it up
automatically.

### 4. Accuracy harness (Section 5) — the actual point of the pilot
```bash
python accuracy_harness.py
```
Computes naive MAE vs. model MAE vs. actual arrivals on a real holdout split
of your logged data, the improvement %, and conformal coverage (does the
90% interval actually contain the true arrival ~90% of the time?). Appends
one row to `data/results_history.csv` per run — that CSV is what the
dashboard's "Model performance" tab reads, so re-running this as more data
comes in builds a real trend line, not a one-off screenshot.

### 5. Dashboard (Section 6) — last
```bash
python dashboard.py
```
Three tabs: Passenger view (per-train cards, ETA + 90% interval, freshness
indicator, route map, "I'm on this train" button), Control room view (raw
table of every tracked train's current state), Model performance (the
harness's naive-vs-model MAE, improvement %, and conformal coverage —
showing "not enough data yet" honestly rather than a placeholder number).

Run `python pipeline.py --loop` in a **separate process** alongside the
dashboard — the dashboard only reads the DB, it never calls RailRadar/OWM
itself.

## Testing everything before real data exists

```bash
PRAVAAH_DB_PATH=data/demo_pravaah.sqlite python demo_data.py --days 60
PRAVAAH_DB_PATH=data/demo_pravaah.sqlite python correction_model.py
PRAVAAH_DB_PATH=data/demo_pravaah.sqlite python accuracy_harness.py
```
`demo_data.py` refuses to run unless `PRAVAAH_DB_PATH` contains the word
"demo" — a deliberate guard so synthetic rows can never land in the same
database as real logged data. **Delete `data/demo_pravaah.sqlite` before
starting real logging**, and never point `dashboard.py` at it.

On 60 days of synthetic data (2 trains, ~1,300 rows) this produced:
naive MAE 6.6 min, model MAE 4.6 min, 29.4% improvement, 91.4% observed
coverage on a 90% interval. **This number is fake** — it's the pipeline
proving it works, not the pilot's result. Your real number, from
`pipeline.py --loop` run for real over real trains, is the actual
deliverable per Section 8.

## Files

| File | Section | Purpose |
|---|---|---|
| `config.py` | 2 | Station table, train roster (approximate — API overrides at runtime), paths |
| `timeutils.py` | — | IST-everywhere time handling, so every ETA is unambiguous on the IST grid |
| `storage.py` | 3.4 | SQLite schema (`eta_logs`, `crowdsource_reports`) + access layer |
| `pipeline.py` | 3 | RailRadar + weather clients, polling loop, the (predicted, actual) join |
| `naive_baseline.py` | 4.1 / 5 | Naive MAE — the number to beat |
| `correction_model.py` | 4.2 / 4.3 | Linear → Random Forest + split conformal intervals |
| `accuracy_harness.py` | 5 | Naive-vs-model MAE, improvement %, conformal coverage, results history |
| `demo_data.py` | — | Synthetic data for smoke-testing only — quarantined from the real DB |
| `dashboard.py` | 6 | Dash app: Passenger / Control room / Model performance tabs |

## Known gaps to close before this is production-grade

- `pipeline.py`'s RailRadar field names (`lastStationCode`, `delayMinutes`, etc.)
  are best-guess placeholders — check RailRadar's current API docs at
  integration time and adjust `RailRadarClient.get_live_status`.
- The accuracy harness currently holds out individual *rows*; once you have
  enough completed journeys, switch the split to hold out entire
  (train_number, service_date) journeys so the model is never scored on a
  journey it partially saw in training.
- `dashboard.py`'s crowdsource lat/lon fields are plain number inputs; wiring
  up the browser's actual geolocation API is a client-side Dash callback
  left as a next step, noted in the code.
