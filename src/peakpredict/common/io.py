"""Storage helpers: DuckDB connections, the raw-store schema, and Parquet IO.

The raw store (DuckDB) is the scraper -> pipeline contract. Processed datasets
and the artifact bundle are Parquet/JSON. These helpers centralize how every
component opens and reads/writes those stores.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from .config import REPO_ROOT

if TYPE_CHECKING:
    import pandas as pd

DATA_DIR = REPO_ROOT / "data"
RAW_DB_PATH = DATA_DIR / "raw" / "peakpredict.duckdb"

# Raw-store schema (scraper -> pipeline contract). Relationships:
# performance.pid and scrape_state.pid reference athlete.pid (enforced in code,
# not as DB FKs, to keep incremental upserts simple).
RAW_DDL: tuple[str, ...] = (
    "CREATE SCHEMA IF NOT EXISTS raw;",
    """
    CREATE TABLE IF NOT EXISTS raw.athlete (
        pid             INTEGER PRIMARY KEY,
        name            TEXT,
        country         TEXT,
        sex             SMALLINT,          -- 1 = men, 2 = women
        dob             DATE,
        url             TEXT,
        height_cm       DOUBLE,            -- static physical profile (source-tagged)
        weight_kg       DOUBLE,
        physical_source TEXT,              -- 'tilastopaja' | 'wikidata' | ...
        scraped_at      TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS raw.event (
        event_id TEXT PRIMARY KEY,
        name     TEXT,
        indoor   BOOLEAN
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS raw.performance (
        perf_id     BIGINT PRIMARY KEY,
        pid         INTEGER,
        event_id    TEXT,
        indoor      BOOLEAN,
        perf_date   DATE,
        mark_raw    TEXT,
        mark        DOUBLE,
        wind        DOUBLE,
        record_flag TEXT,
        round_pos   TEXT,
        competition TEXT,
        location    TEXT,
        scraped_at  TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS raw.scrape_state (
        pid        INTEGER PRIMARY KEY,
        status     TEXT,              -- pending | done | failed
        attempts   INTEGER DEFAULT 0,
        last_error TEXT,
        updated_at TIMESTAMP
    );
    """,
)


def connect(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Open (creating parent dirs) a DuckDB connection with the ``raw`` schema ready."""
    path = Path(db_path) if db_path is not None else RAW_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    return con


def init_raw_store(con: duckdb.DuckDBPyConnection) -> None:
    """Create all raw tables if they do not already exist (idempotent)."""
    for stmt in RAW_DDL:
        con.execute(stmt)
    # migrate older stores that predate the physical-profile columns
    for col, typ in (("height_cm", "DOUBLE"), ("weight_kg", "DOUBLE"), ("physical_source", "TEXT")):
        try:
            con.execute(f"ALTER TABLE raw.athlete ADD COLUMN {col} {typ}")
        except duckdb.Error:
            pass  # column already exists


def write_parquet(df: pd.DataFrame, path: Path | str) -> Path:
    """Write a DataFrame to Parquet, creating parent dirs."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)
    return out


def read_parquet(path: Path | str) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    import pandas as pd

    return pd.read_parquet(path, engine="pyarrow")
