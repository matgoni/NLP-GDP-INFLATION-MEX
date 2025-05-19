# Estimate clarity metrics (length, tokens per sentence, lexical density) per topic and year

import os
import json
import re
from glob import glob
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np


def extract_year_from_filename(filename):
    match = re.search(r'(\d{4})', filename)
    return match.group(1) if match else None


def load_clarity_data(preprocessed_dir):
    data = defaultdict(lambda: defaultdict(list))  # data[topic][year] = list of sentences

    for path in tqdm(glob(os.path.join(preprocessed_dir, 'preprocessed_*.json'))):
        year = extract_year_from_filename(path)
        if not year:
            continue

        with open(path, 'r', encoding='utf-8') as f:
            doc = json.load(f)

        for topic in ['gdp_prioritized', 'gdp_other']:
            for sentence in doc.get(topic, []):
                data['gdp'][year].append(sentence)

        for topic in ['inflation_prioritized', 'inflation_other']:
            for sentence in doc.get(topic, []):
                data['inflation'][year].append(sentence)

    return data


def compute_clarity_metrics(corpus):
    """Compute average sentence length, lexical density, and total tokens."""
    results = []

    for topic in corpus:
        for year in sorted(corpus[topic].keys()):
            sentences = corpus[topic][year]
            if not sentences:
                continue

            num_sentences = len(sentences)
            tokens_per_sentence = [len(sentence.split()) for sentence in sentences]
            all_tokens = [token for sentence in sentences for token in sentence.split()]
            unique_tokens = set(all_tokens)

            metrics = {
                "topic": topic,
                "year": year,
                "num_sentences": num_sentences,
                "total_tokens": len(all_tokens),
                "avg_tokens_per_sentence": np.mean(tokens_per_sentence),
                "lexical_density": len(unique_tokens) / len(all_tokens) if all_tokens else 0
            }
            results.append(metrics)

    return pd.DataFrame(results)


if __name__ == "__main__":
    PREPROCESSED_DIR = "data/preprocessed"
    OUTPUT_PATH = "data/features/clarity/clarity_metrics.csv"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    corpus = load_clarity_data(PREPROCESSED_DIR)
    df = compute_clarity_metrics(corpus)
    df.to_csv(OUTPUT_PATH, index=False)
    print("Clarity metrics saved to clarity_metrics.csv")
