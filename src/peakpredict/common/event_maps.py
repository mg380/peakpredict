"""Event identifiers, names, and per-event performance direction.

Event ids come from the source database (confirmed in the feasibility spike:
40=100m, 50=200m, 60=300m, 70=400m). The full map is populated from the source
``Events.h5`` as event groups are added; v1 only needs the sprint subset.
"""

from __future__ import annotations

# Confirmed sprint/track ids (spike + source Events list). Extend per group.
EVENT_ID_TO_NAME: dict[str, str] = {
    "40": "100m",
    "50": "200m",
    "60": "300m",
    "70": "400m",
}
NAME_TO_EVENT_ID: dict[str, str] = {v: k for k, v in EVENT_ID_TO_NAME.items()}

# v1 scope: sprints, both sexes.
SUPPORTED_V1_EVENTS: frozenset[str] = frozenset({"40", "50", "70"})

# Sexes as encoded by the source: 1 = men, 2 = women.
SEX_MEN = 1
SEX_WOMEN = 2
VALID_SEXES: frozenset[int] = frozenset({SEX_MEN, SEX_WOMEN})

# Substrings marking events where a HIGHER raw mark is better (field + combined).
# Used as a fallback when an id's direction is not explicitly known. All v1
# events are track times (lower is better), so this only matters on expansion.
_HIGHER_IS_BETTER_NAME_HINTS: tuple[str, ...] = (
    "jump",
    "vault",
    "throw",
    "put",
    "shot",
    "discus",
    "hammer",
    "javelin",
    "athlon",  # decathlon / heptathlon / pentathlon (points: higher better)
)


def event_name(event_id: str) -> str:
    """Human-readable event name, or a stable fallback if unmapped."""
    return EVENT_ID_TO_NAME.get(event_id, f"event_{event_id}")


def is_supported_v1(event_id: str) -> bool:
    return event_id in SUPPORTED_V1_EVENTS


def is_valid_sex(sex: int) -> bool:
    return sex in VALID_SEXES


def is_lower_better(event_id: str) -> bool:
    """True when a smaller raw mark is a better performance (track times).

    Defaults to True (track-time semantics); field/combined events are detected
    by name hint and return False. v1 (sprints) is always True.
    """
    name = EVENT_ID_TO_NAME.get(event_id, "").lower()
    return not any(hint in name for hint in _HIGHER_IS_BETTER_NAME_HINTS)
