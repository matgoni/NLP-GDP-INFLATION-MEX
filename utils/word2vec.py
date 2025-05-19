#  Train Word2Vec model and compute average embeddings per year and topic

import os
import json
import re
from glob import glob
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def extract_year_from_filename(filename):
    match = re.search(r'(\d{4})', filename)
    return match.group(1) if match else None


def load_tokenized_corpus_by_year(preprocessed_dir):
    corpora = defaultdict(lambda: defaultdict(list))  # corpora[topic][year] = list of token lists

    for path in tqdm(glob(os.path.join(preprocessed_dir, 'preprocessed_*.json'))):
        year = extract_year_from_filename(path)
        if not year:
            continue

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for topic in ['gdp_prioritized', 'gdp_other']:
            for sentence in data.get(topic, []):
                corpora['gdp'][year].append(sentence.split())

        for topic in ['inflation_prioritized', 'inflation_other']:
            for sentence in data.get(topic, []):
                corpora['inflation'][year].append(sentence.split())

    return corpora


def train_word2vec_model(corpus, vector_size=100, window=5, min_count=2, sg=1):
    all_sentences = []
    for yearly_data in corpus.values():
        for sentences in yearly_data.values():
            all_sentences.extend(sentences)

    model = Word2Vec(
        sentences=all_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=sg
    )
    return model


def compute_average_embeddings(corpora, model):
    embeddings = defaultdict(dict)

    for topic in corpora:
        for year, sentences in corpora[topic].items():
            vectors = []
            for tokens in sentences:
                word_vecs = [model.wv[token] for token in tokens if token in model.wv]
                if word_vecs:
                    vectors.append(np.mean(word_vecs, axis=0))
            if vectors:
                embeddings[topic][year] = np.mean(vectors, axis=0)
            else:
                embeddings[topic][year] = np.zeros(model.vector_size)

    return embeddings


def save_embeddings(embeddings, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for topic, yearly_vecs in embeddings.items():
        df = pd.DataFrame.from_dict(yearly_vecs, orient='index')
        df.index.name = 'year'
        df.to_csv(os.path.join(output_dir, f'embeddings_{topic}.csv'))


if __name__ == "__main__":
    PREPROCESSED_DIR = "data/preprocessed"
    OUTPUT_DIR = "data/features/embeddings"

    corpora = load_tokenized_corpus_by_year(PREPROCESSED_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = train_word2vec_model(corpora)
    model.save(os.path.join(OUTPUT_DIR, 'word2vec.model'))

    embeddings = compute_average_embeddings(corpora, model)
    save_embeddings(embeddings, OUTPUT_DIR)
    print("Word2Vec yearly embeddings saved.")
