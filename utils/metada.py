# Adds structured metadata to preprocessed JSON documents in NLP pipeline

import os
import json
import re
from collections import Counter
from datetime import datetime
from glob import glob
from tqdm import tqdm
import stanza

# Initialize Stanza pipeline for POS tagging
stanza.download('es')
nlp = stanza.Pipeline(lang='es', processors='tokenize,pos,lemma', tokenize_no_ssplit=True)

# Define indicator keywords (lowercase for comparison)
GDP_KEYWORDS = ["pib", "producto interno bruto", "crecimiento económico"]
INFLATION_KEYWORDS = ["inflación", "ipc", "precios al consumidor"]

# Custom mapping from month range in filename to quarters
date_quarter_map = {
    "enero-marzo": ("Q1", "01"),
    "abril-junio": ("Q2", "04"),
    "julio-septiembre": ("Q3", "07"),
    "octubre-diciembre": ("Q4", "10")
}

def extract_date_from_filename(filename):
    """Extract date and quarter from more flexible filename patterns."""
    name = filename.lower()
    year_match = re.search(r'(\d{4})', name)
    for label, (quarter, month) in date_quarter_map.items():
        if label in name:
            year = year_match.group(1) if year_match else "0000"
            return f"{year}-{month}-01", quarter
    return None, None


def tag_indicators(text_sections):
    """Tag indicators based on keyword matches across all text sections"""
    tags = set()
    for section in text_sections:
        joined_text = ' '.join(section).lower() if isinstance(section, list) else section.lower()
        if any(kw in joined_text for kw in GDP_KEYWORDS):
            tags.add("GDP")
        if any(kw in joined_text for kw in INFLATION_KEYWORDS):
            tags.add("inflation")
    return list(tags)


def get_top_verbs(text):
    """Extract the top 10 most frequent verbs using Stanza POS tagging."""
    doc = nlp(text)
    verbs = [word.lemma.lower() for sent in doc.sentences for word in sent.words if word.upos == 'VERB']
    top_verbs = [verb for verb, _ in Counter(verbs).most_common(10)]
    return top_verbs


def get_stats(text_data):
    """Calculate token/section stats and top frequent verbs"""
    total_tokens = 0
    total_sentences = 0
    all_tokens = []
    active_sections = []
    full_text = []

    for key, entries in text_data.items():
        if entries:
            active_sections.append(key)
            for entry in entries:
                tokens = entry.split()
                total_tokens += len(tokens)
                total_sentences += 1
                all_tokens.extend(tokens)
                full_text.append(entry)

    joined_text = ' '.join(full_text)
    top_verbs = get_top_verbs(joined_text)

    return {
        "num_tokens": total_tokens,
        "num_sentences": total_sentences,
        "num_sections": len(active_sections),
        "active_sections": active_sections,
        "top_verbs": top_verbs
    }


def enrich_metadata(json_input_path, metadata_output_path, source_name="Banxico"):
    filename = os.path.basename(json_input_path)
    document_id = os.path.splitext(filename)[0].replace("preprocessed_", "")

    with open(json_input_path, 'r', encoding='utf-8') as f:
        text_data = json.load(f)

    date_str, quarter = extract_date_from_filename(filename)
    indicators = tag_indicators(text_data.values())
    stats = get_stats(text_data)

    metadata = {
        "document_id": document_id,
        "filename": filename,
        "date": date_str,
        "quarter": quarter,
        "indicators": indicators,
        "source": source_name,
        "text_file": json_input_path,
        **stats
    }

    with open(metadata_output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def run_metadata_enrichment(preprocessed_dir, metadata_dir):
    os.makedirs(metadata_dir, exist_ok=True)
    preprocessed_files = glob(os.path.join(preprocessed_dir, 'preprocessed_*.json'))

    for file in tqdm(preprocessed_files, desc="Enriching Metadata"):
        filename = os.path.basename(file).replace(".json", "_metadata.json")
        output_path = os.path.join(metadata_dir, filename)
        enrich_metadata(file, output_path)


if __name__ == "__main__":
    PREPROCESSED_DIR = "data/preprocessed"
    METADATA_DIR = "data/metadata"
    run_metadata_enrichment(PREPROCESSED_DIR, METADATA_DIR)