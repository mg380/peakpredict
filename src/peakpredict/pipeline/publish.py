"""B10 — assemble the versioned artifact bundle (the B -> C contract).

Trains the predictor ladder, selects the simplest model that is at least as good
as the baseline on the temporal split, and writes a self-contained, versioned
bundle the dashboard loads: predictor, normalizer, feature schema, population
aggregates, similarity index, indicator report, validation report, manifest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib

from ..common.io import DATA_DIR, read_parquet, write_parquet
from ..common.logging import get_logger
from ..common.schemas import ArtifactManifest
from .aggregates import build_population_aggregates, build_similarity_index
from .evaluate import temporal_evaluate
from .indicators import compute_indicators
from .model import GroupMeanBaseline, PooledRidge

log = get_logger("pipeline.publish")
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = DATA_DIR / "artifacts"
EVENTS = ["40", "50", "70"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _select_primary(report: dict) -> str:
    ridge, base = report.get("ridge"), report.get("baseline")
    if isinstance(ridge, dict) and isinstance(base, dict):
        return "ridge" if ridge["mae"] <= base["mae"] else "baseline"
    return "baseline"


def publish(
    processed_dir: Path = PROCESSED_DIR,
    out_root: Path = ARTIFACT_DIR,
    version: str | None = None,
    created_at: str | None = None,
) -> tuple[Path, str, dict]:
    processed_dir = Path(processed_dir)
    season_bests = read_parquet(processed_dir / "season_bests.parquet")
    features = read_parquet(processed_dir / "features.parquet")

    report = temporal_evaluate(features)
    primary = _select_primary(report)
    model = PooledRidge().fit(features) if primary == "ridge" else GroupMeanBaseline().fit(features)

    indicators = compute_indicators(features)
    aggregates = build_population_aggregates(season_bests)
    similar = build_similarity_index(features)

    created_at = created_at or datetime.now(UTC).isoformat()
    version = version or "v" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out_root) / version
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump({"primary": primary, "model": model}, out / "predictor.pkl")
    # bundle the shared contracts AND the processed per-athlete data Explore needs
    for fname in (
        "normalization.json",
        "feature_schema.json",
        "season_bests.parquet",
        "labels.parquet",
        "athletes.parquet",
    ):
        shutil.copyfile(processed_dir / fname, out / fname)
    write_parquet(aggregates, out / "aggregates.parquet")
    write_parquet(similar, out / "similar_index.parquet")
    (out / "indicators.json").write_text(json.dumps(indicators, indent=2))
    (out / "validation.json").write_text(json.dumps(report, indent=2))

    fs = json.loads((processed_dir / "feature_schema.json").read_text())
    schema_version = fs["schema_version"]

    def _metric(name: str, key: str) -> float:
        node = report.get(name)
        return float(node[key]) if isinstance(node, dict) else float("nan")

    manifest = ArtifactManifest(
        version=version,
        created_at=created_at,
        code_commit=_git_commit(),
        data_snapshot=str(processed_dir),
        schema_version=schema_version,
        event_group="sprints",
        events=EVENTS,
        metrics={
            "primary_mae": _metric(primary, "mae"),
            "baseline_mae": _metric("baseline", "mae"),
            "primary_interval_coverage": _metric(primary, "interval_coverage"),
        },
    )
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    log.info("published %s (primary=%s)", version, primary)
    return out, primary, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="peakpredict artifact publish (B6-B10)")
    parser.add_argument("--processed", default=str(PROCESSED_DIR))
    parser.add_argument("--out", default=str(ARTIFACT_DIR))
    parser.add_argument("--version", default=None)
    args = parser.parse_args(argv)
    publish(Path(args.processed), Path(args.out), args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
