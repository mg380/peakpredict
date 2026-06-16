"""A3 — parse an athlete career page into performance rows.

Ports the feasibility-spike parsing of ``db/at.php`` pages. The page groups
results into ``Outdoor``/``Indoor`` sections; within a section each event lives
in a ``<div id="Outdoorx{event_id}">`` whose rows are grouped by single-cell
year headers. A data row's first cell holds ``mark[ wind][ record]`` (separated
by non-breaking spaces) and the last cell holds the day+month.

``parse_career`` is pure (HTML in, rows out) so it is unit-tested against a
saved page fixture with no network.
"""

from __future__ import annotations

import re
from datetime import date

from bs4 import BeautifulSoup

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}
_SECTIONS = ("Outdoor", "Indoor")
_WIND_RE = re.compile(r"^[+-]\d+(?:\.\d+)?$")
# leading time/number: optional minutes (e.g. "1:43.20") then seconds/value
_MARK_RE = re.compile(r"^(?:(\d+):)?(\d+(?:\.\d+)?)")


def parse_mark_cell(cell_text: str) -> tuple[str, float | None, float | None, str | None]:
    """Split a mark cell into (raw, mark_seconds_or_value, wind, record_flag)."""
    raw = cell_text.replace("\xa0", " ").strip()
    tokens = raw.split()
    mark: float | None = None
    wind: float | None = None
    records: list[str] = []
    if tokens:
        m = _MARK_RE.match(tokens[0])
        if m:
            minutes = float(m.group(1)) if m.group(1) else 0.0
            mark = minutes * 60.0 + float(m.group(2))
        for tok in tokens[1:]:
            if _WIND_RE.match(tok):
                wind = float(tok)
            elif tok.strip("=").isalpha():  # WR, AR, =WR, NR, PB, SB ...
                records.append(tok)
    return raw, mark, wind, (" ".join(records) or None)


def _parse_day_month(text: str, year: int | None) -> date | None:
    if year is None:
        return None
    parts = text.replace("\xa0", " ").split()
    if len(parts) < 2 or parts[1] not in _MONTHS:
        return None
    try:
        return date(year, _MONTHS[parts[1]], int(parts[0]))
    except (ValueError, TypeError):
        return None


def parse_career(html: str, pid: int) -> list[dict]:
    """Parse a career page into raw performance dicts (no perf_id/scraped_at)."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    for section in _SECTIONS:
        indoor = section == "Indoor"
        tables = soup.find_all("div", id=lambda v, s=section: bool(v) and v.startswith(s + "x"))
        for table in tables:
            event_id = table["id"].split("x")[-1]
            year: int | None = None
            for tr in table.find_all("tr"):
                cells = [c.text.strip() for c in tr.find_all("td")]
                if len(cells) == 1:
                    token = cells[0].strip()
                    if token.isnumeric():
                        year = int(token)
                    continue
                if not cells or not any(cells):
                    continue
                raw, mark, wind, record = parse_mark_cell(cells[0])
                perf_date = _parse_day_month(cells[-1], year)
                rows.append(
                    {
                        "pid": pid,
                        "event_id": event_id,
                        "indoor": indoor,
                        "perf_date": perf_date,
                        "mark_raw": raw,
                        "mark": mark,
                        "wind": wind,
                        "record_flag": record,
                        "round_pos": cells[1] if len(cells) >= 2 else None,
                        "competition": cells[2] if len(cells) >= 3 else None,
                        "location": cells[3] if len(cells) >= 4 else None,
                    }
                )
    return rows


def scrape_career(session, pid: int, sex: int) -> list[dict]:
    """Fetch and parse one athlete's career page."""
    html = session.get_page(session.athlete_url(pid, sex))
    return parse_career(html, pid)
