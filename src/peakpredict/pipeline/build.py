"""B-data build: raw store -> processed datasets.

Runs B1-B5 and writes the processed Parquet tables plus the fitted normalizer
and feature schema. The trained model + artifact bundle (B6-B10) are produced
by a later phase and consume these outputs.

Run: ``python -m peakpredict.pipeline.build --db data/raw/peakpredict.duckdb``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.io import DATA_DIR, RAW_DB_PATH, connect, write_parquet
from ..common.logging import get_logger
from .features import build_features, feature_schema
from .labels import build_labels
from .loading import load_raw
from .normalize import add_scores, fit_normalizer
from .season_best import build_season_bests

log = get_logger("pipeline.build")
PROCESSED_DIR = DATA_DIR / "processed"


def build_processed(db_path: str | Path = RAW_DB_PATH, out_dir: Path = PROCESSED_DIR) -> dict:
    """Build season-bests, labels, and features; persist to ``out_dir``."""
    con = connect(db_path)
    try:
        perf, athletes = load_raw(con)
    finally:
        con.close()

    season_bests = build_season_bests(perf, athletes)
    normalizer = fit_normalizer(season_bests)
    scored = add_scores(season_bests, normalizer)
    labels = build_labels(scored)
    physical = athletes[["pid", "height_cm", "weight_kg"]]
    features = build_features(scored, labels, physical=physical)

    athletes_dir = (
        athletes[athletes["pid"].isin(scored["pid"].unique())][["pid", "name", "country", "sex"]]
        .drop_duplicates("pid")
        .reset_index(drop=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(scored, out_dir / "season_bests.parquet")
    write_parquet(labels, out_dir / "labels.parquet")
    write_parquet(features, out_dir / "features.parquet")
    write_parquet(athletes_dir, out_dir / "athletes.parquet")
    (out_dir / "normalization.json").write_text(json.dumps(normalizer.to_dict()))
    (out_dir / "feature_schema.json").write_text(feature_schema().model_dump_json(indent=2))

    summary = {
        "season_bests": int(len(scored)),
        "labels": int(len(labels)),
        "features": int(len(features)),
    }
    log.info("build complete: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="peakpredict pipeline data build (B1-B5)")
    parser.add_argument("--db", default=str(RAW_DB_PATH), help="raw DuckDB store")
    parser.add_argument("--out", default=str(PROCESSED_DIR), help="processed output dir")
    args = parser.parse_args(argv)
    build_processed(args.db, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
