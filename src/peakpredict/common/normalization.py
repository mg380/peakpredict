"""Performance normalization — the single shared scoring contract.

A normalized *score* is monotone and **higher-is-better** for every event, so
trajectories are comparable across athletes regardless of whether the raw mark
is a time (lower better) or a distance/points (higher better).

The same fitted normalizer is published in the artifact bundle by the pipeline
and re-loaded by the dashboard, so the Upload flow scores a manually-entered
mark identically to how the model was trained (no divergence).

v1 uses ``ZScoreNormalizer`` (within event+sex standardization, direction-aware).
``WorldAthleticsNormalizer`` is the interface-compatible slot for the official
World Athletics scoring tables once their coefficients are supplied.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .event_maps import is_lower_better


def _key(event_id: str, sex: int) -> str:
    return f"{event_id}:{int(sex)}"


@runtime_checkable
class Normalizer(Protocol):
    """Maps a raw mark to a higher-is-better score and back."""

    def transform(self, mark: float, event_id: str, sex: int) -> float: ...
    def inverse(self, score: float, event_id: str, sex: int) -> float: ...


class ZScoreNormalizer:
    """Within-(event, sex) standardization, oriented so higher = better.

    ``params`` maps ``"<event_id>:<sex>"`` -> (mean, std) of the raw marks.
    For lower-is-better events the sign is flipped so a faster time yields a
    higher score.
    """

    def __init__(self, params: dict[str, tuple[float, float]] | None = None) -> None:
        self.params: dict[str, tuple[float, float]] = dict(params or {})

    # -- fitting -----------------------------------------------------------
    def fit(self, df) -> ZScoreNormalizer:
        """Fit mean/std per (event_id, sex) from a frame with those columns + 'mark'."""
        import pandas as pd  # local import keeps module load light for the dashboard

        if not isinstance(df, pd.DataFrame):
            raise TypeError("fit expects a pandas DataFrame")
        for col in ("event_id", "sex", "mark"):
            if col not in df.columns:
                raise ValueError(f"missing required column '{col}'")
        params: dict[str, tuple[float, float]] = {}
        for (event_id, sex), grp in df.groupby(["event_id", "sex"]):
            marks = grp["mark"].astype(float)
            std = float(marks.std(ddof=0))
            if len(marks) >= 2 and std > 0:
                params[_key(str(event_id), int(sex))] = (float(marks.mean()), std)
        self.params = params
        return self

    # -- transform ---------------------------------------------------------
    def _lookup(self, event_id: str, sex: int) -> tuple[float, float]:
        try:
            return self.params[_key(event_id, sex)]
        except KeyError:
            raise KeyError(
                f"no normalization params for event '{event_id}', sex {sex}; "
                "the normalizer must be fit on (or loaded with) this event+sex"
            ) from None

    def transform(self, mark: float, event_id: str, sex: int) -> float:
        mean, std = self._lookup(event_id, sex)
        sign = -1.0 if is_lower_better(event_id) else 1.0
        return sign * (float(mark) - mean) / std

    def inverse(self, score: float, event_id: str, sex: int) -> float:
        mean, std = self._lookup(event_id, sex)
        sign = -1.0 if is_lower_better(event_id) else 1.0
        return mean + sign * float(score) * std

    # -- persistence (for the artifact bundle) -----------------------------
    def to_dict(self) -> dict:
        return {
            "kind": "zscore",
            "params": {k: [m, s] for k, (m, s) in self.params.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> ZScoreNormalizer:
        if data.get("kind") != "zscore":
            raise ValueError(f"expected kind 'zscore', got {data.get('kind')!r}")
        params = {k: (float(v[0]), float(v[1])) for k, v in data.get("params", {}).items()}
        return cls(params)


class WorldAthleticsNormalizer:
    """Interface slot for the official World Athletics scoring tables.

    Not implemented in v1: it requires the published per-event coefficient
    tables. Implementing it must not change any caller — it satisfies the same
    ``Normalizer`` protocol as ``ZScoreNormalizer``.
    """

    def transform(self, mark: float, event_id: str, sex: int) -> float:
        raise NotImplementedError(
            "World Athletics scoring tables are not bundled yet; use ZScoreNormalizer in v1"
        )

    def inverse(self, score: float, event_id: str, sex: int) -> float:
        raise NotImplementedError(
            "World Athletics scoring tables are not bundled yet; use ZScoreNormalizer in v1"
        )
