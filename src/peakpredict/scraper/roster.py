"""A2 — parse an all-time ranking page (``db/alltfull.php``) into athlete rows.

Each ranked row carries the athlete's name (a link with ``ID=<pid>``), country,
and date of birth. This yields the roster (with DOB, needed for age features)
and the set of athlete ids whose career pages component A3 then fetches.

Column layout (confirmed from the original working notebook):
``[rank, score, wind, record, name(link), country, dob, position, event, location, date]``.
``parse_roster`` is pure for unit testing against a saved page fixture.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from bs4 import BeautifulSoup

from ..common.logging import get_logger

_ID_RE = re.compile(r"ID=(\d+)")
log = get_logger("scraper.roster")


def _parse_dob(text: str) -> date | None:
    """Parse '21 Aug 86' -> date, correcting 2-digit years that land in the future."""
    text = text.replace("\xa0", " ").strip()
    if not text:
        return None
    try:
        dob = datetime.strptime(text, "%d %b %y").date()
    except ValueError:
        return None
    if dob > date.today():  # e.g. '68' parsed as 2068 -> 1968
        dob = dob.replace(year=dob.year - 100)
    return dob


def parse_roster(html: str, sex: int, *, indoor: bool = False) -> list[dict]:
    """Parse ranked athletes from an all-time list page."""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        log.warning("no table found in roster page")
        return []
    table = tables[-1]
    athletes: list[dict] = []
    seen: set[int] = set()
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 7:
            continue
        rank = cells[0].text.strip()
        if not rank.isnumeric():
            continue
        link = cells[4].find("a")
        if not link or "href" not in link.attrs:
            continue
        m = _ID_RE.search(link["href"])
        if not m:
            continue
        pid = int(m.group(1))
        if pid in seen:
            continue
        seen.add(pid)
        athletes.append(
            {
                "pid": pid,
                "name": link.text.strip(),
                "country": cells[5].text.strip() or None,
                "dob": _parse_dob(cells[6].text.strip()),
                "sex": sex,
                "url": f"https://www.tilastopaja.info/db/at.php?Sex={sex}&ID={pid}",
            }
        )
    return athletes


def scrape_roster(session, event_id: str, sex: int, *, indoor: bool = False) -> list[dict]:
    """Fetch and parse one all-time list (event + sex)."""
    html = session.get_page(session.roster_url(event_id, sex, indoor=indoor))
    return parse_roster(html, sex, indoor=indoor)
