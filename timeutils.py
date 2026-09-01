"""
Single source of truth for time handling.

Every timestamp that gets logged, joined, or shown on the dashboard goes
through here first. Rail data APIs and weather APIs are not guaranteed to
return timestamps in IST (some return UTC, some return naive local time) —
mixing those silently is exactly how an ETA pipeline ends up 5.5 hours wrong
and no one notices. So: convert on the way IN, store IST everywhere, format
IST on the way OUT.
"""

from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> datetime:
    return datetime.now(IST)


def to_ist(dt: datetime) -> datetime:
    """Convert any aware datetime to IST. Raises if dt is naive — naive
    timestamps from an API must be explicitly tagged with their source
    timezone before they reach this function; guessing is how bugs happen."""
    if dt.tzinfo is None:
        raise ValueError(
            "to_ist() received a naive datetime. Tag it with its source "
            "timezone (e.g. dt.replace(tzinfo=timezone.utc)) before converting."
        )
    return dt.astimezone(IST)


def parse_api_timestamp(raw: str, assume_tz: timezone = timezone.utc) -> datetime:
    """Parse a timestamp string from an external API into IST.

    Many rail-data APIs (RailRadar included, per their docs) return either
    ISO-8601 with an explicit offset, or a bare 'YYYY-MM-DD HH:MM:SS' that is
    implicitly UTC. Handle both; only fall back to `assume_tz` when the
    string carries no offset of its own.
    """
    raw = raw.strip()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        # Bare 'YYYY-MM-DD HH:MM:SS' with no offset info.
        dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=assume_tz)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=assume_tz)
    return to_ist(dt)


def combine_service_date_and_hhmm(service_date: str, hhmm: str) -> datetime:
    """Build an IST datetime from a service date ('YYYY-MM-DD') and a
    scheduled clock time ('HH:MM' or 'HH:MM+1' for next-day arrivals like the
    Lucknow Mail's 06:55+1)."""
    plus_day = 0
    if hhmm.endswith("+1"):
        plus_day = 1
        hhmm = hhmm[:-2]
    base = datetime.strptime(service_date, "%Y-%m-%d")
    hh, mm = map(int, hhmm.split(":"))
    dt = base.replace(hour=hh, minute=mm, tzinfo=IST) + timedelta(days=plus_day)
    return dt


def fmt_ist(dt: datetime) -> str:
    """Human-readable IST string for the dashboard, e.g. '06:55 IST (31 Aug)'."""
    dt = to_ist(dt)
    return dt.strftime("%H:%M IST (%d %b)")


def iso_ist(dt: datetime) -> str:
    """Canonical string form for storage in SQLite/CSV — always IST, always
    with explicit offset so a re-read of the DB can never silently drift."""
    return to_ist(dt).isoformat()
