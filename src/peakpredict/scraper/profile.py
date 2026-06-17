"""Parse the static physical profile from a tilastopaja athlete page.

The profile is a small table of ``<b>Label:</b> value`` rows (Height, Weight,
Club, ...). Height/weight are static per athlete (not per competition), so this
yields one value each. Pure function — tested against a saved page fixture.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

_H_RANGE = (120.0, 230.0)  # cm
_W_RANGE = (30.0, 200.0)   # kg
_NUM = re.compile(r"(\d+(?:[.,]\d+)?)")


def parse_profile(html: str) -> dict:
    """Return {'height_cm': float|None, 'weight_kg': float|None} from the page."""
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, float | None] = {"height_cm": None, "weight_kg": None}
    for b in soup.find_all("b"):
        label = b.get_text(strip=True).rstrip(":").strip().lower()
        if label not in ("height", "weight"):
            continue
        cell = b.parent.get_text(" ", strip=True) if b.parent else ""
        m = _NUM.search(cell)
        if not m:
            continue
        val = float(m.group(1).replace(",", "."))
        if label == "height":
            cm = val * 100 if val < 3 else val  # statements may be in m or cm
            out["height_cm"] = cm if _H_RANGE[0] <= cm <= _H_RANGE[1] else None
        else:
            out["weight_kg"] = val if _W_RANGE[0] <= val <= _W_RANGE[1] else None
    return out
