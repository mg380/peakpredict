"""Phase-0 feasibility spike: fetch ONE athlete career page and inspect the
longitudinal structure. Respectful: single athlete, single request after login."""
import os, sys, time, json
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

load_dotenv("sports_data_platform/.env")
USER = os.environ.get("SPORTS_DATA_USER"); PW = os.environ.get("SPORTS_DATA_PASS")
assert USER and PW, "missing creds"
ATHLETE = "https://www.tilastopaja.info/db/at.php?Sex=1&ID=45032"  # Usain Bolt

opts = Options()
opts.add_argument("--headless=new"); opts.add_argument("--window-size=1400,1000")
opts.add_argument("--no-sandbox"); opts.add_argument("--disable-gpu")
print("launching chrome..."); sys.stdout.flush()
d = webdriver.Chrome(options=opts)
try:
    d.set_page_load_timeout(40)
    d.get("https://www.tilastopaja.info/login.php")
    d.find_element(By.NAME, "user").send_keys(USER)
    d.find_element(By.NAME, "password").send_keys(PW + Keys.RETURN)
    try: d.find_element(By.XPATH, "//input[@type='button' and @value='Login']").click()
    except Exception: pass
    time.sleep(4)
    print("after-login url:", d.current_url, "| title:", d.title[:60])
    d.get(ATHLETE)
    try: WebDriverWait(d, 15).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
    except Exception: pass
    time.sleep(2)
    html = d.page_source
    open("specs/peak-performance-predictor/spike/athlete_45032.html","w",encoding="utf-8").write(html)
    print("html bytes:", len(html))
    soup = BeautifulSoup(html, "html.parser")
    records = []
    for location in ["Outdoor","Indoor"]:
        tables = soup.find_all("div", id=lambda v: v and v.startswith(location+"x"))
        for t in tables:
            event = t["id"].split("x")[-1]
            year = None
            for row in t.find_all("tr"):
                cells = [c.text.strip() for c in row.find_all("td")]
                if len(cells)==1:
                    if cells[0].isnumeric(): year = int(cells[0])
                    continue
                if cells and any(cells):
                    records.append({"loc":location,"event":event,"year":year,"cells":cells})
    print("total parsed performance rows:", len(records))
    years = [r["year"] for r in records if r["year"]]
    if years: print("year span:", min(years), "->", max(years), "| distinct years:", len(set(years)))
    print("distinct (loc,event):", sorted({(r['loc'],r['event']) for r in records}))
    print("\n-- sample rows --")
    for r in records[:12]: print(r)
    json.dump(records, open("specs/peak-performance-predictor/spike/athlete_45032_parsed.json","w"), indent=1)
finally:
    d.quit()
print("DONE")
