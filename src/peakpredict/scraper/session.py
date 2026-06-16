"""A1 — authenticated session to the source site.

Logs in with Selenium (the login form is JS-driven) and serves page HTML.
Credentials come from ``common.config`` (gitignored ``.secrets``) and are never
logged. URLs follow the structure confirmed in the feasibility spike.
"""

from __future__ import annotations

import time
from contextlib import suppress

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from ..common.config import get_secret
from ..common.logging import get_logger

BASE_URL = "https://www.tilastopaja.info"
LOGIN_URL = f"{BASE_URL}/login.php"


class LoginError(RuntimeError):
    """Raised when authentication does not leave the login page."""


class SessionManager:
    """A logged-in Selenium session. Reusable across many page fetches."""

    def __init__(self, *, headless: bool = True, page_timeout: int = 40) -> None:
        self.headless = headless
        self.page_timeout = page_timeout
        self.driver: webdriver.Chrome | None = None
        self._authed = False
        self.log = get_logger("scraper.session")

    def _init_driver(self) -> None:
        if self.driver is not None:
            return
        opts = Options()
        if self.headless:
            for arg in (
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--window-size=1400,1000",
            ):
                opts.add_argument(arg)
        self.driver = webdriver.Chrome(options=opts)
        self.driver.set_page_load_timeout(self.page_timeout)

    def login(self) -> SessionManager:
        """Authenticate. Raises LoginError if the login page is still shown."""
        self._init_driver()
        assert self.driver is not None
        user = get_secret("TILASTOPAJA_USER")
        password = get_secret("TILASTOPAJA_PASS")
        self.driver.get(LOGIN_URL)
        self.driver.find_element(By.NAME, "user").send_keys(user)
        self.driver.find_element(By.NAME, "password").send_keys(password + Keys.RETURN)
        with suppress(Exception):
            self.driver.find_element(By.XPATH, "//input[@type='button' and @value='Login']").click()
        time.sleep(3)
        if "login.php" in self.driver.current_url:
            raise LoginError("authentication failed: still on login page (check credentials)")
        self._authed = True
        self.log.info("authenticated")
        return self

    def get_page(self, url: str, params: dict | None = None) -> str:
        """Return page HTML, authenticating on first use."""
        if not self._authed:
            self.login()
        assert self.driver is not None
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{query}"
        self.driver.get(url)
        return self.driver.page_source

    # -- URL builders (spike-confirmed) ------------------------------------
    def athlete_url(self, pid: int, sex: int) -> str:
        return f"{BASE_URL}/db/at.php?Sex={sex}&ID={pid}"

    def roster_url(self, event_id: str, sex: int, *, indoor: bool = False) -> str:
        ind = 1 if indoor else 0
        return (
            f"{BASE_URL}/db/alltfull.php?Ind={ind}&Event={event_id}"
            f"&Sex={sex}&area=&All=0&Age=99"
        )

    def close(self) -> None:
        if self.driver is not None:
            with suppress(Exception):
                self.driver.quit()
            self.driver = None
            self._authed = False
