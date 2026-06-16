from datetime import date
from pathlib import Path

from peakpredict.scraper.roster import _parse_dob, parse_roster

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "roster_70_2_outdoor.html"


def test_parse_dob_two_digit_year():
    assert _parse_dob("18 Feb 57") == date(1957, 2, 18)


def test_parse_dob_future_year_corrected():
    # a '68' that would parse as 2068 must roll back a century
    d = _parse_dob("01 Jan 68")
    assert d is not None and d.year == 1968


def test_parse_dob_blank():
    assert _parse_dob("") is None


def test_parse_roster_on_fixture():
    athletes = parse_roster(FIXTURE.read_text(encoding="utf-8"), sex=2)
    assert len(athletes) > 300

    pids = [a["pid"] for a in athletes]
    assert all(isinstance(p, int) for p in pids)
    assert len(set(pids)) == len(pids)  # de-duplicated

    # DOB extracted for the vast majority (age features depend on it)
    assert sum(1 for a in athletes if a["dob"]) / len(athletes) > 0.9

    # a known 400m athlete is present and correctly parsed
    koch = [a for a in athletes if a["name"] == "Marita Koch"]
    assert koch and koch[0]["country"] == "GER"
    assert all(a["sex"] == 2 for a in athletes)
