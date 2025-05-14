import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL    = "https://www.banxico.org.mx/publicaciones-y-prensa/informes-trimestrales/informes-trimestrales-precios.html"
SAVE_DIR    = "data/"
START_YEAR  = 2015
END_YEAR    = 2024

os.makedirs(SAVE_DIR, exist_ok=True)

def slugify(text):
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text

def fetch_html():
    resp = requests.get(BASE_URL)
    resp.raise_for_status()
    return resp.text

def extract_pdf_links_and_names(html):
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a", string=lambda s: s and "Texto completo" in s)
    entries = []
    for a in anchors:
        # grab the aria-label or title, e.g. "Texto completo de Informe trimestral, octubre-diciembre 2024"
        label = a.get("aria-label") or a.get("title") or ""
        # extract the quarter-year part, e.g. "octubre-diciembre 2024"
        m = re.search(r"informe trimestral,\s*(.+\s+(\d{4}))", label, re.IGNORECASE)
        if not m:
            continue
        quarter_text, year_str = m.group(1), m.group(2)
        year = int(year_str)
        # filter by year limits
        if year < START_YEAR or year > END_YEAR:
            continue

        href = a.get("href", "")
        if not href.endswith(".pdf"):
            continue

        full_url = urljoin(BASE_URL, href)
        # build a filename: informe-trimestral_oct-dic-2024.pdf
        filename = f"informe-trimestral_{slugify(quarter_text)}.pdf"
        entries.append((full_url, filename))
    return entries

def download_named_pdfs(entries):
    for url, fname in entries:
        path = os.path.join(SAVE_DIR, fname)
        if os.path.exists(path):
            print(f"✓ Skipped (exists): {fname}")
            continue
        print(f"↓ Downloading: {fname}")
        resp = requests.get(url)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)

def run():
    html    = fetch_html()
    entries = extract_pdf_links_and_names(html)
    download_named_pdfs(entries)

if __name__ == "__main__":
    run()