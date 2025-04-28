import os
import csv
import time
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# === Load .env and API Key ===
load_dotenv()
API_KEY = os.getenv("SERPAPI_API_KEY")
if not API_KEY:
    raise ValueError("❌ SERPAPI_API_KEY not set in .env")

# === Params ===
SITES = ["eleconomista.com.mx", "elfinanciero.com.mx"]
START_YEAR = 2015
END_YEAR   = 2024
PAUSE_S    = 2
OUTPUT_CSV = "outputs/news_economics_mexico_2015_2024.csv"
BASE_URL   = "https://serpapi.com/search"

# Economic keywords as before…
ECONOMIC_KEYWORDS = [
  "PIB","inflación","crecimiento","recesión","caída",
  "desaceleración","expansión","prevé","proyecta","estima",
  "pronostica","tasas de interés","Banxico","empleo",
  "desempleo","actividad económica"
]

def build_query(site):
    kws = " OR ".join(f'"{w}"' for w in ECONOMIC_KEYWORDS)
    return f"site:{site} ({kws})"

def fetch_news(site, start_dt, end_dt):
    """Fetch news with a custom date range via tbs=cdr… and sort-by-date."""
    # Google’s tbs wants MM/DD/YYYY
    cd_min = start_dt.strftime("%-m/%-d/%Y")
    cd_max = end_dt.strftime("%-m/%-d/%Y")
    tbs    = f"cdr:1,cd_min:{cd_min},cd_max:{cd_max},sbd:1"

    params = {
      "engine":  "google_news",
      "q":       build_query(site),
      "api_key": API_KEY,
      "gl":      "MX",    # Mexico
      "hl":      "es",    # Spanish
      "tbs":     tbs,     # custom date range + sort by date :contentReference[oaicite:1]{index=1}
      "no_cache":"true",
      "output":  "json"
    }

    resp = requests.get(BASE_URL, params=params)
    js   = resp.json()
    if "error" in js:
        print("⚠️ API error:", js["error"])
        return []
    return js.get("news_results", [])

def scrape_news():
    all_articles = []
    for year in range(START_YEAR, END_YEAR+1):
        for month in range(1,13):
            sd = datetime(year, month, 1)
            ed = (sd.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            for site in SITES:
                print(f"Fetching {site} | {sd:%Y-%m} → {ed:%Y-%m}")
                for art in fetch_news(site, sd, ed):
                    all_articles.append({
                      "site": site,
                      "start": sd.strftime("%Y-%m-%d"),
                      "end":   ed.strftime("%Y-%m-%d"),
                      "title":          art.get("title"),
                      "link":           art.get("link"),
                      "published_date": art.get("date"),
                      "source":         art.get("source"),
                      "snippet":        art.get("snippet")
                    })
                time.sleep(PAUSE_S)
    return all_articles

def save(arts, path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(arts[0].keys()))
        w.writeheader()
        w.writerows(arts)
    print(f"✅ {len(arts)} articles saved to {path}")

if __name__=="__main__":
    print("🚀 Starting …")
    all_arts = scrape_news()
    if not all_arts:
        print("⚠️ No articles—check API key, rate limits or your tbs format.")
    else:
        save(all_arts, OUTPUT_CSV)
    print("🏁 Done.")
