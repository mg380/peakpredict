"""Enrich the roster with physical characteristics from Wikidata (CC0).

Wikidata holds height/weight/DOB for ~15k athletics athletes, queryable in bulk
via SPARQL. We pull them, match to our athletes on normalized name + date of
birth, and write height/weight (tagged ``physical_source='wikidata'``) onto
``raw.athlete`` — without overwriting any value already present (e.g. from the
tilastopaja profile, which takes priority).

Run: python -m peakpredict.enrich.wikidata  [--db data/raw/peakpredict.duckdb]
"""

from __future__ import annotations

import argparse
import re
import unicodedata

import pandas as pd

from ..common.io import RAW_DB_PATH, connect, init_raw_store
from ..common.logging import get_logger

log = get_logger("enrich.wikidata")

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
_USER_AGENT = "peakpredict/0.1 (athletics peak-performance research)"
# athletics athletes (sport = athletics, Q542) with a height; weight optional
_QUERY = """
SELECT ?name ?dob ?height ?mass WHERE {
  ?p wdt:P641 wd:Q542 ; wdt:P2048 ?height ; wdt:P569 ?dob ; rdfs:label ?name .
  FILTER(LANG(?name) = "en")
  OPTIONAL { ?p wdt:P2067 ?mass . }
}
"""
_H_RANGE = (120.0, 230.0)  # cm sanity bounds
_W_RANGE = (30.0, 200.0)   # kg sanity bounds


def normalize_name(s: str) -> str:
    """Lowercase, strip diacritics and collapse whitespace for matching."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s.strip().lower())


def fetch_wikidata_physical(timeout: int = 180) -> pd.DataFrame:
    """Fetch (name, dob, height_cm, weight_kg) for athletics athletes from Wikidata."""
    import requests

    resp = requests.get(
        SPARQL_ENDPOINT,
        params={"query": _QUERY, "format": "json"},
        headers={"User-Agent": _USER_AGENT, "Accept": "application/sparql-results+json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    rows = []
    for b in resp.json()["results"]["bindings"]:
        try:
            h = float(b["height"]["value"])
        except (KeyError, ValueError):
            continue
        h_cm = h * 100 if h < 3 else h  # statements are in metres or centimetres
        if not (_H_RANGE[0] <= h_cm <= _H_RANGE[1]):
            continue
        weight = None
        mass = b.get("mass", {}).get("value")
        if mass is not None:
            try:
                w = float(mass)
                weight = w if _W_RANGE[0] <= w <= _W_RANGE[1] else None
            except ValueError:
                weight = None
        rows.append({
            "name": b["name"]["value"], "dob": b["dob"]["value"][:10],
            "height_cm": round(h_cm, 1), "weight_kg": weight,
        })
    return pd.DataFrame(rows, columns=["name", "dob", "height_cm", "weight_kg"])


def _match_key(names: pd.Series, dobs: pd.Series) -> pd.Series:
    norm = names.map(normalize_name)
    day = pd.to_datetime(dobs, errors="coerce").dt.date.astype("string")
    return norm + "|" + day


def join_to_roster(roster: pd.DataFrame, wd: pd.DataFrame) -> pd.DataFrame:
    """Return (pid, height_cm, weight_kg) for roster athletes matched in Wikidata."""
    r = roster.copy()
    r["_k"] = _match_key(r["name"], r["dob"])
    w = wd.copy()
    w["_k"] = _match_key(w["name"], w["dob"])
    w = w.dropna(subset=["height_cm"]).sort_values("height_cm").drop_duplicates("_k", keep="first")
    merged = r.merge(w[["_k", "height_cm", "weight_kg"]], on="_k", how="left")
    return merged[["pid", "height_cm", "weight_kg"]]


def enrich(db_path=RAW_DB_PATH) -> dict:
    """Fetch Wikidata physical data, match, and write onto raw.athlete."""
    wd = fetch_wikidata_physical()
    log.info("fetched %d Wikidata athletes with height", len(wd))
    con = connect(db_path)
    init_raw_store(con)
    try:
        roster = con.execute("SELECT pid, name, dob FROM raw.athlete WHERE dob IS NOT NULL").df()
        matched = join_to_roster(roster, wd)
        hits = matched.dropna(subset=["height_cm"])
        for r in hits.itertuples(index=False):
            weight = None if pd.isna(r.weight_kg) else float(r.weight_kg)
            con.execute(
                "UPDATE raw.athlete SET height_cm = ?, weight_kg = ?, physical_source = 'wikidata' "
                "WHERE pid = ? AND height_cm IS NULL",
                [float(r.height_cm), weight, int(r.pid)],
            )
        total = con.execute("SELECT count(*) FROM raw.athlete").fetchone()[0]
        with_h = con.execute(
            "SELECT count(*) FROM raw.athlete WHERE height_cm IS NOT NULL"
        ).fetchone()[0]
        with_w = con.execute(
            "SELECT count(*) FROM raw.athlete WHERE weight_kg IS NOT NULL"
        ).fetchone()[0]
    finally:
        con.close()
    summary = {"wikidata_rows": int(len(wd)), "roster": int(len(roster)),
               "matched": int(len(hits)), "athletes_total": int(total),
               "with_height": int(with_h), "with_weight": int(with_w)}
    log.info("enrichment: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="enrich roster with Wikidata physical data")
    parser.add_argument("--db", default=str(RAW_DB_PATH))
    args = parser.parse_args(argv)
    s = enrich(args.db)
    cov_h = s["with_height"] / s["athletes_total"] if s["athletes_total"] else 0
    print("\nWikidata enrichment complete")
    print(f"  Wikidata athletes pulled : {s['wikidata_rows']}")
    print(f"  roster matched           : {s['matched']}")
    print(f"  athletes with height now : {s['with_height']} / {s['athletes_total']} ({cov_h:.0%})")
    print(f"  athletes with weight now : {s['with_weight']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
