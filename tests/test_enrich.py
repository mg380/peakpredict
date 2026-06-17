from pathlib import Path

import pandas as pd

from peakpredict.enrich.wikidata import join_to_roster, normalize_name
from peakpredict.scraper.profile import parse_profile

CAREER_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "specs" / "peak-performance-predictor" / "spike" / "athlete_45032.html"
)


def test_parse_profile_on_fixture():
    profile = parse_profile(CAREER_FIXTURE.read_text(encoding="utf-8"))
    assert profile["height_cm"] == 196.0  # Bolt's profile
    assert profile["weight_kg"] == 88.0


def test_parse_profile_handles_missing():
    profile = parse_profile("<html><body><table></table></body></html>")
    assert profile == {"height_cm": None, "weight_kg": None}


def test_normalize_name_strips_accents_and_case():
    assert normalize_name("Andrés  Simón") == "andres simon"
    assert normalize_name("Usain BOLT") == "usain bolt"


def test_join_to_roster_matches_on_name_and_dob():
    roster = pd.DataFrame(
        {
            "pid": [1, 2, 3],
            "name": ["Usain Bolt", "Andrés Simón", "Nobody Here"],
            "dob": ["1986-08-21", "1968-09-01", "1990-01-01"],
        }
    )
    wd = pd.DataFrame(
        {
            "name": ["usain bolt", "Andres Simon", "Someone Else"],
            "dob": ["1986-08-21", "1968-09-01", "1986-08-21"],
            "height_cm": [196.0, 180.0, 170.0],
            "weight_kg": [88.0, None, 60.0],
        }
    )
    matched = join_to_roster(roster, wd).set_index("pid")
    assert matched.loc[1, "height_cm"] == 196.0  # accent/case-insensitive match
    assert matched.loc[2, "height_cm"] == 180.0
    assert pd.isna(matched.loc[3, "height_cm"])  # no name+dob match
