import os
import json
import re
import stanza
import ftfy
from nltk.corpus import stopwords
from nltk import download
from langdetect import detect
from pathlib import Path
from glob import glob
from tqdm import tqdm

# Download resources
download('stopwords')
stanza.download('es')  # Run only once

# Load Spanish stopwords and initialize Stanza NLP pipeline
stop_words = set(stopwords.words('spanish'))
nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,lemma', tokenize_no_ssplit=True)

# Logging rejected sentences (optional)
REJECTED_SENTENCES = []

def clean_text(text):
    text = ftfy.fix_text(text)  # Fix broken Unicode
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()

    # Merge sequences of single characters (e.g., "h i p ó t e s i s" → "hipótesis")
    def merge_split_words(t):
        return re.sub(r'(?:\b\w\b\s*){3,}', lambda m: m.group(0).replace(' ', ''), t)

    text = merge_split_words(text)
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def is_valid_sentence(sentence):
    tokens = sentence.split()
    if len(tokens) < 3:
        return False
    short_tokens = [t for t in tokens if len(t) <= 2]
    if len(short_tokens) / len(tokens) > 0.5:
        return False
    try:
        if detect(sentence) != 'es':
            return False
    except:
        return False
    return True

def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]

def lemmatize(tokens):
    text = ' '.join(tokens)
    doc = nlp(text)
    return [
        word.lemma.lower()
        for sent in doc.sentences
        for word in sent.words
        if word.lemma and word.lemma.lower() not in stop_words and word.lemma != 'PRON'
    ]

def preprocess_text(text):
    cleaned = clean_text(text)
    sentences = re.split(r'[.!?]', cleaned)
    valid_sentences = []
    for s in sentences:
        s = s.strip()
        if is_valid_sentence(s):
            valid_sentences.append(s)
        else:
            REJECTED_SENTENCES.append(s)
    if not valid_sentences:
        return ''
    tokens = ' '.join(valid_sentences).split()
    tokens_nostop = remove_stopwords(tokens)
    lemmas = lemmatize(tokens_nostop)
    return ' '.join(lemmas)

def preprocess_json_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    processed_data = {}
    for category in ['gdp_prioritized', 'inflation_prioritized', 'gdp_other', 'inflation_other']:
        processed_entries = []
        for entry in data.get(category, []):
            preprocessed_entry = preprocess_text(entry)
            if preprocessed_entry:
                processed_entries.append(preprocessed_entry)
        processed_data[category] = processed_entries

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(processed_data, outfile, ensure_ascii=False, indent=2)

def run_pipeline(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_files = glob(os.path.join(input_dir, '**', '*.json'), recursive=True)

    for input_file in tqdm(input_files, desc='Preprocessing JSON files'):
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f'preprocessed_{filename}')
        preprocess_json_file(input_file, output_file)

    # Optional: log rejected sentences
    if REJECTED_SENTENCES:
        with open(os.path.join(output_dir, 'rejected_sentences.txt'), 'w', encoding='utf-8') as f:
            for sent in REJECTED_SENTENCES:
                f.write(sent + '\n')

if __name__ == "__main__":
    INPUT_DIR = 'data/extracted'
    OUTPUT_DIR = 'data/preprocessed'
    run_pipeline(INPUT_DIR, OUTPUT_DIR)