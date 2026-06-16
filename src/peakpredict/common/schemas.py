"""Cross-boundary data models (pydantic v2).

These are the contracts that cross component boundaries. ``UploadedAthlete`` is
the dashboard upload contract; ``FeatureSchema``/``ArtifactManifest`` describe
the pipeline -> dashboard artifact bundle. Using validated models (not loose
dicts) at boundaries is mandated by the SRS.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .event_maps import is_supported_v1, is_valid_sex

Confidence = Literal[
    "ok",
    "low",
    "insufficient",
    "out_of_distribution",
    "unsupported_event",
]


class PerformanceRow(BaseModel):
    """One parsed performance from an athlete's career page (raw store row)."""

    pid: int
    event_id: str
    indoor: bool = False
    perf_date: date
    mark: float
    wind: float | None = None
    record_flag: str | None = None
    round_pos: str | None = None
    competition: str | None = None
    location: str | None = None


class UploadedResult(BaseModel):
    """One manually-entered result row from the dashboard Upload form.

    Age (years) is entered directly, since the model works in age space; this
    avoids requiring a date of birth for an uploaded prospect.
    """

    age: float = Field(gt=0)
    mark: float = Field(gt=0)
    wind: float | None = None
    competition: str | None = None


class UploadedAthlete(BaseModel):
    """A user-uploaded athlete to predict on. Event must be in the v1 set."""

    sex: int
    event_id: str
    results: list[UploadedResult] = Field(min_length=1)

    @field_validator("sex")
    @classmethod
    def _check_sex(cls, v: int) -> int:
        if not is_valid_sex(v):
            raise ValueError("sex must be 1 (men) or 2 (women)")
        return v

    @field_validator("event_id")
    @classmethod
    def _check_event(cls, v: str) -> str:
        if not is_supported_v1(v):
            raise ValueError(f"event_id '{v}' is not supported in v1 (sprints only)")
        return v


class PeakPrediction(BaseModel):
    """The model's output for one athlete, with uncertainty and confidence flag."""

    peak_age: float
    interval_lo: float
    interval_hi: float
    peak_score: float
    window_lo: float
    window_hi: float
    confidence: Confidence


class FieldSpec(BaseModel):
    """One field a user upload must provide (the feature/upload contract)."""

    name: str
    dtype: str
    required: bool = True
    unit: str | None = None
    description: str | None = None


class FeatureSchema(BaseModel):
    """The versioned schema a dashboard upload is validated against."""

    schema_version: str
    fields: list[FieldSpec]


class ArtifactManifest(BaseModel):
    """Metadata stamped on every published artifact bundle (reproducibility)."""

    version: str
    created_at: str
    code_commit: str
    data_snapshot: str
    schema_version: str
    event_group: str
    events: list[str]
    metrics: dict[str, float] = Field(default_factory=dict)
