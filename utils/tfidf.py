# Compute TF-IDF matrices per year and topic (GDP vs Inflation), filtered by POS (NOUN, VERB)

import os
import json
import re
import time
from glob import glob
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import stanza

# Initialize Stanza pipeline
stanza.download('es')
nlp = stanza.Pipeline(lang='es', processors='tokenize,pos,lemma', tokenize_no_ssplit=True)


def extract_year_from_filename(filename):
    match = re.search(r'(\d{4})', filename)
    return match.group(1) if match else None


def extract_filtered_lemmas(texts, allowed_pos={'NOUN', 'VERB'}):
    lemmas = []
    for sentence in texts:
        doc = nlp(sentence)
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos in allowed_pos and word.lemma and len(word.lemma) > 2:
                    lemmas.append(word.lemma.lower())
    return ' '.join(lemmas)


def load_topic_corpus_by_year(preprocessed_dir):
    corpora = defaultdict(lambda: defaultdict(list))  # structure: corpora[topic][year] = [texts]

    for path in tqdm(glob(os.path.join(preprocessed_dir, 'preprocessed_*.json'))):
        year = extract_year_from_filename(path)
        if not year:
            continue

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for topic in ['gdp_prioritized', 'gdp_other']:
            if topic in data:
                corpora['gdp'][year].extend(data[topic])

        for topic in ['inflation_prioritized', 'inflation_other']:
            if topic in data:
                corpora['inflation'][year].extend(data[topic])

    return corpora


def compute_tfidf_matrices(corpora, max_features=1000):
    tfidf_matrices = {}
    vectorizers = {}

    for topic in corpora:
        tfidf_matrices[topic] = {}
        vectorizers[topic] = {}
        for year in sorted(corpora[topic].keys()):
            start_time = time.time()
            print(f"Processing {topic} {year}...")

            doc = extract_filtered_lemmas(corpora[topic][year], allowed_pos={"NOUN", "VERB"})

            if not doc.strip():
                print(f"  Skipping {topic} {year} — no valid verbs/nouns found.")
                continue

            try:
                vectorizer = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
                tfidf = vectorizer.fit_transform([doc])
                if not vectorizer.get_feature_names_out().size:
                    print(f"  Skipping {topic} {year} — TF-IDF vocabulary is empty after filtering.")
                    continue

                df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out(), index=[year])
                tfidf_matrices[topic][year] = df
                vectorizers[topic][year] = vectorizer
                print(f"  Done in {time.time() - start_time:.2f}s → {len(doc.split())} tokens")
            except ValueError as e:
                print(f"  Skipping {topic} {year} due to TF-IDF error: {e}")

    return tfidf_matrices, vectorizers


def save_tfidf_matrices(tfidf_matrices, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for topic in tfidf_matrices:
        if tfidf_matrices[topic]:
            all_years_df = pd.concat(tfidf_matrices[topic].values())
            all_years_df.to_csv(os.path.join(output_dir, f"tfidf_{topic}.csv"))


if __name__ == "__main__":
    PREPROCESSED_DIR = "data/preprocessed"
    OUTPUT_DIR = "data/features/tfidf"

    corpora = load_topic_corpus_by_year(PREPROCESSED_DIR)
    tfidf_matrices, _ = compute_tfidf_matrices(corpora)
    save_tfidf_matrices(tfidf_matrices, OUTPUT_DIR)
    print("TF-IDF matrices (NOUN+VERB only) saved per topic.")