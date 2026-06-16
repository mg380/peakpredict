"""Load raw-store tables into DataFrames for the analysis pipeline."""

from __future__ import annotations

import duckdb
import pandas as pd


def load_raw(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (performances, athletes) DataFrames from the raw store."""
    perf = con.execute(
        "SELECT pid, event_id, indoor, perf_date, mark, wind FROM raw.performance"
    ).df()
    ath = con.execute("SELECT pid, sex, dob, name, country FROM raw.athlete").df()
    return perf, ath
