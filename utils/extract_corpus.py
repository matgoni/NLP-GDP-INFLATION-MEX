# pipeline_mexico.py
# ETL for NLP-Driven Macroeconomic Discourse Analysis – Mexico-Focused
# Single-column parsing for 2015–2017; two-column thereafter; enhanced filtering.

import os
import re
import json
import glob
import pdfplumber
import warnings

# Suppress CropBox warnings
warnings.filterwarnings("ignore", message="CropBox missing from /Page.*")

# Directories
raw_dir = 'data/raw'
output_dir = 'data/extracted'

os.makedirs(output_dir, exist_ok=True)

# Keyword lists
gdp_keywords = ['PIB', 'producto interno bruto', 'crecimiento económico', 'desarrollo económico']
inflation_keywords = ['inflación', 'inflación subyacente', 'IPC', 'tasa de inflación', 'precios al consumidor']
mexico_keywords = ['México', 'mexicano', 'nacional', 'doméstico', 'economía nacional', 'mercado interno', 'Gobierno de México', 'Banco de México', 'Banxico']
global_keywords = ['mundo', 'mundial', 'global', 'internacional', 'economía mundial', 'zona del euro', 'China', 'Asia', 'Japón', 'Unión Europea', 'EE.UU.', 'Latinoamérica', 'economías emergentes']

# Filtering thresholds
MIN_SENTENCE_LENGTH = 30
PRIORITY_SCORE_THRESHOLD = 2


def extract_text_columns(path, two_column=True):
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            if two_column:
                w, h = page.width, page.height
                left = page.within_bbox((0, 0, w/2, h)).extract_text() or ''
                right = page.within_bbox((w/2, 0, w, h)).extract_text() or ''
                texts.append(f"{left}\n{right}")
            else:
                texts.append(page.extract_text() or '')
    return '\n'.join(texts)


def tokenize(text):
    return [s.strip() for s in re.split(r'(?<=[\.\!?])\s+', text) if s.strip()]


def score_sentence(sent, keywords):
    score = 0
    for kw in keywords:
        score += len(re.findall(r"\b" + re.escape(kw) + r"\b", sent, re.IGNORECASE))
    return score


def filter_exclude(sentences, exclude_patterns):
    exc = re.compile(r"\b(?:" + "|".join(map(re.escape, exclude_patterns)) + r")\b", re.IGNORECASE)
    return [s for s in sentences if not exc.search(s)]


def process_pdf(path):
    # Determine year from filename
    base = os.path.splitext(os.path.basename(path))[0]
    year_match = re.search(r"(20\d{2})", base)
    year = int(year_match.group(1)) if year_match else None
    # Use single column for 2015–2017, two columns from 2018 onward
    two_col = False if year and 2015 <= year <= 2017 else True

    text = extract_text_columns(path, two_column=two_col)
    sentences = tokenize(text)

    # Exclude global context, short lines, chart captions
    filtered = filter_exclude(sentences, global_keywords)
    filtered = [s for s in filtered if len(s) >= MIN_SENTENCE_LENGTH]
    filtered = [s for s in filtered if not re.match(r"(?i)^\s*grá?fica", s)]

    # Prepare patterns
    gdp_pat = re.compile(r"\b(?:" + "|".join(map(re.escape, gdp_keywords)) + r")\b", re.IGNORECASE)
    inf_pat = re.compile(r"\b(?:" + "|".join(map(re.escape, inflation_keywords)) + r")\b", re.IGNORECASE)
    mex_pat = re.compile(r"\b(?:" + "|".join(map(re.escape, mexico_keywords)) + r")\b", re.IGNORECASE)

    # Buckets
    gdp_prioritized, gdp_other = [], []
    inf_prioritized, inf_other = [], []

    # Classify sentences
    for s in filtered:
        gdp_score = score_sentence(s, gdp_keywords)
        inf_score = score_sentence(s, inflation_keywords)
        mex_score = score_sentence(s, mexico_keywords)
        # GDP
        if gdp_score > 0:
            if mex_score + gdp_score >= PRIORITY_SCORE_THRESHOLD:
                gdp_prioritized.append(s)
            else:
                gdp_other.append(s)
        # Inflation
        if inf_score > 0:
            if mex_score + inf_score >= PRIORITY_SCORE_THRESHOLD:
                inf_prioritized.append(s)
            else:
                inf_other.append(s)

    return {
        'gdp_prioritized': gdp_prioritized,
        'inflation_prioritized': inf_prioritized,
        'gdp_other': gdp_other,
        'inflation_other': inf_other
    }


def main():
    for path in glob.glob(os.path.join(raw_dir, '*.pdf')):
        base = os.path.splitext(os.path.basename(path))[0]
        year = (re.search(r"20\d{2}", base) or ['unknown'])[0]
        year_dir = os.path.join(output_dir, year)
        os.makedirs(year_dir, exist_ok=True)

        data = process_pdf(path)
        out_path = os.path.join(year_dir, f"{base}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Written {out_path}")

if __name__ == '__main__':
    main()
