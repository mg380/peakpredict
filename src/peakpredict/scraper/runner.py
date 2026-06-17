"""A4 — scrape orchestration and CLI.

Drives roster ingestion (A2) then career ingestion (A3) into the raw store,
resumably: ``raw.scrape_state`` tracks each athlete's status so a re-run skips
``done`` athletes. Writes are idempotent upserts (stable ``perf_id``), so a
crash mid-run never corrupts the store. Athlete fetches are throttled.

Run: ``python -m peakpredict.scraper.runner --events 40,50,70 --sexes 1,2``
"""

from __future__ import annotations

import argparse
import hashlib
import time

import duckdb
from selenium.common.exceptions import WebDriverException

from ..common.event_maps import event_name
from ..common.io import RAW_DB_PATH, connect, init_raw_store
from ..common.logging import get_logger
from .career import scrape_athlete
from .roster import scrape_roster
from .session import SessionManager

log = get_logger("scraper.runner")


def _stable_perf_id(row: dict) -> int:
    key = (
        f"{row['pid']}|{row['event_id']}|{row['indoor']}|{row['perf_date']}"
        f"|{row['mark_raw']}|{row['round_pos']}"
    )
    return int(hashlib.sha1(key.encode()).hexdigest()[:15], 16)  # 60 bits, fits BIGINT


def upsert_event(con: duckdb.DuckDBPyConnection, event_id: str, indoor: bool) -> None:
    con.execute(
        "INSERT INTO raw.event (event_id, name, indoor) VALUES (?, ?, ?) "
        "ON CONFLICT (event_id) DO NOTHING",
        [event_id, event_name(event_id), indoor],
    )


def upsert_athletes(con: duckdb.DuckDBPyConnection, athletes: list[dict]) -> int:
    for a in athletes:
        con.execute(
            "INSERT INTO raw.athlete (pid, name, country, sex, dob, url, scraped_at) "
            "VALUES (?, ?, ?, ?, ?, ?, now()) ON CONFLICT (pid) DO UPDATE SET "
            "name=excluded.name, country=excluded.country, sex=excluded.sex, "
            "dob=excluded.dob, url=excluded.url, scraped_at=excluded.scraped_at",
            [a["pid"], a["name"], a["country"], a["sex"], a["dob"], a["url"]],
        )
        # Only initialize state for new athletes; never reset an existing status.
        con.execute(
            "INSERT INTO raw.scrape_state (pid, status, attempts, updated_at) "
            "VALUES (?, 'pending', 0, now()) ON CONFLICT (pid) DO NOTHING",
            [a["pid"]],
        )
    return len(athletes)


def upsert_performances(con: duckdb.DuckDBPyConnection, rows: list[dict]) -> int:
    for r in rows:
        con.execute(
            "INSERT INTO raw.performance (perf_id, pid, event_id, indoor, perf_date, mark_raw, "
            "mark, wind, record_flag, round_pos, competition, location, scraped_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now()) "
            "ON CONFLICT (perf_id) DO UPDATE SET mark=excluded.mark, wind=excluded.wind, "
            "record_flag=excluded.record_flag, round_pos=excluded.round_pos, "
            "competition=excluded.competition, location=excluded.location, "
            "scraped_at=excluded.scraped_at",
            [
                _stable_perf_id(r), r["pid"], r["event_id"], r["indoor"], r["perf_date"],
                r["mark_raw"], r["mark"], r["wind"], r["record_flag"], r["round_pos"],
                r["competition"], r["location"],
            ],
        )
    return len(rows)


def pending_athletes(con: duckdb.DuckDBPyConnection, limit: int | None = None) -> list[tuple]:
    """(pid, sex) for athletes not yet 'done' — the resumability core."""
    query = (
        "SELECT s.pid, a.sex FROM raw.scrape_state s JOIN raw.athlete a USING (pid) "
        "WHERE s.status <> 'done' ORDER BY s.pid"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return con.execute(query).fetchall()


def _mark_state(con, pid: int, status: str, error: str | None) -> None:
    con.execute(
        "UPDATE raw.scrape_state SET status=?, last_error=?, attempts=attempts+1, "
        "updated_at=now() WHERE pid=?",
        [status, error, pid],
    )


def ingest_roster(con, session, events: list[str], sexes: list[int], indoor: bool) -> int:
    total = 0
    for event_id in events:
        upsert_event(con, event_id, indoor)
        for sex in sexes:
            athletes = scrape_roster(session, event_id, sex, indoor=indoor)
            n = upsert_athletes(con, athletes)
            total += n
            log.info("roster event=%s sex=%s -> %d athletes", event_id, sex, n)
    return total


# markers of a dead browser/driver (the chromedriver process itself can die, which
# surfaces as a urllib3/requests connection error, not a WebDriverException)
_SESSION_DEAD_MARKERS = (
    "invalid session id",
    "session deleted",
    "no such window",
    "max retries exceeded",
    "connection refused",
    "not reachable",
    "disconnected",
    "failed to establish",
)


def _is_session_dead(exc: Exception) -> bool:
    if isinstance(exc, WebDriverException):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _SESSION_DEAD_MARKERS)


def _update_physical(con, pid: int, profile: dict) -> None:
    """Write tilastopaja-profile height/weight (overrides other sources where present)."""
    h, w = profile.get("height_cm"), profile.get("weight_kg")
    if h is None and w is None:
        return
    con.execute(
        "UPDATE raw.athlete SET height_cm = COALESCE(?, height_cm), "
        "weight_kg = COALESCE(?, weight_kg), physical_source = 'tilastopaja' WHERE pid = ?",
        [h, w, pid],
    )


def _scrape_one(con, session, pid: int, sex: int) -> str:
    """Scrape+store one athlete. Returns 'done', 'failed', or 'session_dead'."""
    try:
        rows, profile = scrape_athlete(session, pid, sex)
        upsert_performances(con, rows)
        _update_physical(con, pid, profile)
        _mark_state(con, pid, "done", None)
        log.info("athlete %s -> %d performances", pid, len(rows))
        return "done"
    except Exception as exc:  # noqa: BLE001 - record and continue, do not abort the run
        _mark_state(con, pid, "failed", str(exc)[:300])
        line = str(exc).splitlines()[0][:120]
        if _is_session_dead(exc):
            log.warning("athlete %s failed (browser/driver down): %s", pid, line)
            return "session_dead"
        log.warning("athlete %s failed: %s", pid, line)
        return "failed"


def _recover_session(session) -> bool:
    """Recreate the browser, reusing the session cookies where possible (minimises logins)."""
    try:
        session.recycle()
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("session recovery failed: %s", exc)
        return False


def ingest_careers(
    con, session, *, throttle: float, limit: int | None, restart_every: int = 400
) -> int:
    pend = pending_athletes(con, limit=limit)
    log.info("career scrape: %d athlete(s) pending", len(pend))
    done = 0
    for i, (pid, sex) in enumerate(pend):
        # proactively recycle the browser to avoid the slow memory leak that kills
        # chromedriver on very long runs
        if i and restart_every and i % restart_every == 0:
            log.info("recycling browser session after %d athletes", i)
            _recover_session(session)
        status = _scrape_one(con, session, int(pid), int(sex))
        if status == "session_dead":
            # the browser/driver died; recover and retry this athlete once
            time.sleep(throttle * 2)
            if _recover_session(session):
                status = _scrape_one(con, session, int(pid), int(sex))
        if status == "done":
            done += 1
        time.sleep(throttle)
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="peakpredict scraper (Component A)")
    parser.add_argument("--events", default="40,50,70", help="comma-separated event ids")
    parser.add_argument("--sexes", default="1,2", help="comma-separated sexes (1=men,2=women)")
    parser.add_argument("--indoor", action="store_true", help="scrape indoor lists")
    parser.add_argument("--limit", type=int, default=None, help="max athlete careers this run")
    parser.add_argument("--throttle", type=float, default=2.0, help="seconds between athletes")
    parser.add_argument("--roster-only", action="store_true")
    parser.add_argument("--career-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume (this is the default): athletes already marked done are always skipped",
    )
    parser.add_argument("--db", default=str(RAW_DB_PATH), help="DuckDB path")
    args = parser.parse_args(argv)

    events = [e.strip() for e in args.events.split(",") if e.strip()]
    sexes = [int(s) for s in args.sexes.split(",") if s.strip()]

    con = connect(args.db)
    init_raw_store(con)
    session = SessionManager()
    try:
        session.login()
        if not args.career_only:
            ingest_roster(con, session, events, sexes, args.indoor)
        if not args.roster_only:
            ingest_careers(con, session, throttle=args.throttle, limit=args.limit)
    finally:
        session.close()
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
