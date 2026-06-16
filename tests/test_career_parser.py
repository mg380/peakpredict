from pathlib import Path

from peakpredict.scraper.career import parse_career, parse_mark_cell

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "specs" / "peak-performance-predictor" / "spike" / "athlete_45032.html"
)


def test_parse_mark_cell_splits_mark_wind_record():
    raw, mark, wind, rec = parse_mark_cell("9.72\xa0\xa0+1.7\xa0\xa0WR")
    assert mark == 9.72
    assert wind == 1.7
    assert rec == "WR"


def test_parse_mark_cell_negative_wind_no_record():
    _, mark, wind, rec = parse_mark_cell("10.40\xa0-2.0")
    assert mark == 10.40
    assert wind == -2.0
    assert rec is None


def test_parse_mark_cell_strips_annotation_suffix():
    _, mark, wind, _ = parse_mark_cell("9.14+")
    assert mark == 9.14
    assert wind is None


def test_parse_mark_cell_minutes_to_seconds():
    _, mark, _, _ = parse_mark_cell("1:43.20")
    assert mark == 103.20


def test_parse_mark_cell_empty():
    raw, mark, wind, rec = parse_mark_cell("")
    assert (raw, mark, wind, rec) == ("", None, None, None)


def test_parse_career_on_fixture():
    rows = parse_career(FIXTURE.read_text(encoding="utf-8"), pid=45032)
    assert len(rows) > 200
    assert all(r["pid"] == 45032 for r in rows)

    # 100m (event 40) present; Bolt's best ~9.58
    e40 = [r["mark"] for r in rows if r["event_id"] == "40" and r["mark"] is not None]
    assert e40
    assert min(e40) == __import__("pytest").approx(9.58, abs=0.02)

    # wind and record flags are parsed from the mark cell
    assert any(r["wind"] is not None for r in rows)
    assert any(r["record_flag"] and "WR" in r["record_flag"] for r in rows)

    # dates assemble from year header + day/month, within Bolt's career span
    dated = [r["perf_date"] for r in rows if r["perf_date"] is not None]
    assert dated
    assert all(2000 <= d.year <= 2018 for d in dated)
