from datetime import date

from peakpredict.common.io import connect, init_raw_store
from peakpredict.scraper.runner import (
    _stable_perf_id,
    pending_athletes,
    upsert_athletes,
    upsert_performances,
)


def _con(tmp_path):
    con = connect(tmp_path / "store.duckdb")
    init_raw_store(con)
    return con


def _athletes():
    return [
        {"pid": 1, "name": "A", "country": "USA", "sex": 1, "dob": date(1990, 1, 1), "url": "u1"},
        {"pid": 2, "name": "B", "country": "GBR", "sex": 2, "dob": date(1992, 5, 5), "url": "u2"},
    ]


def _perf(pid=1):
    return {
        "pid": pid, "event_id": "40", "indoor": False, "perf_date": date(2010, 6, 1),
        "mark_raw": "9.99 +0.5", "mark": 9.99, "wind": 0.5, "record_flag": None,
        "round_pos": "1", "competition": "GP", "location": "Rome",
    }


def test_upsert_athletes_initializes_pending_state(tmp_path):
    con = _con(tmp_path)
    upsert_athletes(con, _athletes())
    assert con.execute("SELECT count(*) FROM raw.athlete").fetchone()[0] == 2
    pend = pending_athletes(con)
    assert sorted(pend) == [(1, 1), (2, 2)]
    con.close()


def test_resume_skips_done(tmp_path):
    con = _con(tmp_path)
    upsert_athletes(con, _athletes())
    con.execute("UPDATE raw.scrape_state SET status='done' WHERE pid=1")
    assert pending_athletes(con) == [(2, 2)]
    con.close()


def test_reingest_roster_does_not_reset_done(tmp_path):
    con = _con(tmp_path)
    upsert_athletes(con, _athletes())
    con.execute("UPDATE raw.scrape_state SET status='done' WHERE pid=1")
    upsert_athletes(con, _athletes())  # roster re-run
    assert pending_athletes(con) == [(2, 2)]  # pid 1 stays done
    con.close()


def test_performance_upsert_is_idempotent(tmp_path):
    con = _con(tmp_path)
    upsert_athletes(con, _athletes())
    upsert_performances(con, [_perf()])
    upsert_performances(con, [_perf()])  # same row again
    assert con.execute("SELECT count(*) FROM raw.performance").fetchone()[0] == 1
    con.close()


def test_stable_perf_id_is_deterministic():
    assert _stable_perf_id(_perf()) == _stable_perf_id(_perf())
    assert _stable_perf_id(_perf(1)) != _stable_perf_id(_perf(2))
