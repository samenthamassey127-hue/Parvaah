"""
SQLite persistence layer. Two tables:

  eta_logs  — one row per poll, per Section 3.4's schema exactly, plus a
              few bookkeeping columns (service_date, id, actual_arrival_ist
              which starts NULL and gets filled in once the train reaches
              NDLS — that fill-in IS the join Section 3.4 describes).

  crowdsource_reports — the "I'm on this train" layer (Section 3.3).

Kept as plain SQLite via the stdlib `sqlite3` module — no ORM — because at
pilot scale (a couple of trains, a few polls/hour) a CSV-with-extra-steps is
all this needs, and the master prompt explicitly says not to over-build here.
"""

import sqlite3
from contextlib import contextmanager

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS eta_logs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    poll_timestamp_ist      TEXT NOT NULL,   -- when this row was logged
    train_number            TEXT NOT NULL,
    service_date            TEXT NOT NULL,   -- 'YYYY-MM-DD' departure date, disambiguates repeated daily runs
    last_station_code       TEXT,
    distance_remaining_km   REAL,
    reported_delay_min      REAL,
    weather_visibility_km   REAL,
    weather_precip_mm       REAL,
    naive_predicted_arrival_ist TEXT,
    model_predicted_arrival_ist TEXT,
    confidence_low_ist      TEXT,
    confidence_high_ist     TEXT,
    api_last_update_ist     TEXT,            -- freshness indicator source (Section 6)
    actual_arrival_ist      TEXT             -- NULL until the train reaches NDLS
);
CREATE INDEX IF NOT EXISTS idx_eta_logs_train_service
    ON eta_logs(train_number, service_date);

CREATE TABLE IF NOT EXISTS crowdsource_reports (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    report_timestamp_ist TEXT NOT NULL,
    train_number        TEXT NOT NULL,
    service_date        TEXT NOT NULL,
    user_id             TEXT NOT NULL,       -- pseudonymous; consent required at submission time
    reported_lat        REAL,
    reported_lon        REAL,
    nearest_station_code TEXT
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def insert_eta_log(row: dict) -> int:
    cols = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    with get_conn() as conn:
        cur = conn.execute(
            f"INSERT INTO eta_logs ({cols}) VALUES ({placeholders})",
            list(row.values()),
        )
        return cur.lastrowid


def record_actual_arrival(train_number: str, service_date: str, actual_arrival_ist: str) -> int:
    """The join described in Section 3.4: back-fill actual_arrival_ist onto
    every prediction row already logged for this train's run, so each one
    becomes a scoreable (predicted, actual) pair."""
    with get_conn() as conn:
        cur = conn.execute(
            """UPDATE eta_logs
               SET actual_arrival_ist = ?
               WHERE train_number = ? AND service_date = ? AND actual_arrival_ist IS NULL""",
            (actual_arrival_ist, train_number, service_date),
        )
        return cur.rowcount


def insert_crowdsource_report(row: dict) -> int:
    cols = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    with get_conn() as conn:
        cur = conn.execute(
            f"INSERT INTO crowdsource_reports ({cols}) VALUES ({placeholders})",
            list(row.values()),
        )
        return cur.lastrowid


def fetch_all_eta_logs():
    """Returns list[sqlite3.Row] of every logged poll, across all trains."""
    with get_conn() as conn:
        return conn.execute("SELECT * FROM eta_logs ORDER BY poll_timestamp_ist").fetchall()


def fetch_scored_logs():
    """Only rows that have a real actual arrival joined — i.e. the rows the
    accuracy harness (Section 5) is allowed to score against."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM eta_logs WHERE actual_arrival_ist IS NOT NULL ORDER BY poll_timestamp_ist"
        ).fetchall()


def has_actual_arrival(train_number: str, service_date: str) -> bool:
    """True once this train's run for this service_date already has a joined
    actual arrival — the pipeline uses this to stop polling (and stop
    spending API budget on) a run that's already finished."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT 1 FROM eta_logs
               WHERE train_number = ? AND service_date = ? AND actual_arrival_ist IS NOT NULL
               LIMIT 1""",
            (train_number, service_date),
        ).fetchone()
        return row is not None


def fetch_latest_per_train():
    """Latest poll row for each train — what the dashboard's live view reads."""
    with get_conn() as conn:
        return conn.execute(
            """SELECT e.* FROM eta_logs e
               INNER JOIN (
                   SELECT train_number, MAX(poll_timestamp_ist) AS max_ts
                   FROM eta_logs GROUP BY train_number
               ) latest
               ON e.train_number = latest.train_number AND e.poll_timestamp_ist = latest.max_ts"""
        ).fetchall()
