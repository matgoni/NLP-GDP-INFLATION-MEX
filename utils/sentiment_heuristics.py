# sentiment_heuristics.py
# Estimate sentiment orientation per year and topic using lexical heuristics

import os
import json
import re
from glob import glob
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd

# Basic Spanish positive/negative wordlists (extendable)
POSITIVE_WORDS = set([
    "crecimiento", "mejor", "favorable", "positivo", "sólido", "expansión", "fortaleza",
    "avance", "estabilidad", "reducción", "logro", "recuperación"
])

NEGATIVE_WORDS = set([
    "inflación", "caída", "deterioro", "negativo", "riesgo", "debilidad", "incertidumbre",
    "contracción", "desaceleración", "presión", "volatilidad", "aumento"
])


def extract_year_from_filename(filename):
    match = re.search(r'(\d{4})', filename)
    return match.group(1) if match else None


def load_sentiment_data(preprocessed_dir):
    data = defaultdict(lambda: defaultdict(list))

    for path in tqdm(glob(os.path.join(preprocessed_dir, 'preprocessed_*.json'))):
        year = extract_year_from_filename(path)
        if not year:
            continue

        with open(path, 'r', encoding='utf-8') as f:
            doc = json.load(f)

        for topic in ['gdp_prioritized', 'gdp_other']:
            data['gdp'][year].extend(doc.get(topic, []))

        for topic in ['inflation_prioritized', 'inflation_other']:
            data['inflation'][year].extend(doc.get(topic, []))

    return data


def compute_sentiment_scores(corpus):
    results = []

    for topic in corpus:
        for year in sorted(corpus[topic].keys()):
            tokens = [word for sentence in corpus[topic][year] for word in sentence.split()]
            total = len(tokens)
            counter = Counter(tokens)
            pos = sum(counter[w] for w in POSITIVE_WORDS if w in counter)
            neg = sum(counter[w] for w in NEGATIVE_WORDS if w in counter)
            neutral = total - pos - neg

            polarity = (pos - neg) / total if total > 0 else 0
            sentiment = {
                "topic": topic,
                "year": year,
                "positive": pos,
                "negative": neg,
                "neutral": neutral,
                "total_tokens": total,
                "sentiment_score": polarity
            }
            results.append(sentiment)

    return pd.DataFrame(results)


if __name__ == "__main__":
    PREPROCESSED_DIR = "data/preprocessed"
    OUTPUT_PATH = "data/features/sentiment/sentiment_heuristics.csv"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    corpus = load_sentiment_data(PREPROCESSED_DIR)
    df = compute_sentiment_scores(corpus)
    df.to_csv(OUTPUT_PATH, index=False)
    print("Sentiment heuristic scores saved to sentiment_heuristics.csv")