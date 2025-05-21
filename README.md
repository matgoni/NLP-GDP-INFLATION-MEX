# NLP-Based Analysis of GDP & Inflation in Mexico  
**Bank of Mexico Inflation Reports (2015–2024)**

## Project Summary

This project implements a domain-adapted Natural Language Processing (NLP) pipeline to analyze the quarterly inflation reports published by the Bank of Mexico from 2015 to 2024. It aims to uncover thematic emphasis, sentiment polarity, semantic drift, and clarity trends in how macroeconomic topics—specifically GDP and inflation—are communicated over time. The approach is optimized for interpretability and computational efficiency in Spanish-language economic documents.

---

## Objectives

- Automatically retrieve and preprocess economic reports from 2015 to 2024.
- Identify and extract topic-relevant discourse related to GDP and inflation.
- Quantify rhetorical tone, textual complexity, and semantic shifts over time.
- Visualize trends in term salience, clarity, and framing strategies in central bank communication.

---

## Methodology and NLP Tasks

The project is structured as a modular pipeline composed of the following stages:

### 1. Data Acquisition and Parsing
- Automated scraping of quarterly reports from the Banxico website (`scrape_banxico.py`).
- PDF parsing with layout-aware extraction for single and two-column documents (`extract_corpus.py`).

### 2. Text Preprocessing
- Unicode correction, noise removal, and normalization using `ftfy`, `re`, and `nltk`.
- Sentence segmentation and tokenization.
- Lemmatization with the Spanish `Stanza` NLP pipeline.
- Stopword filtering using extended Spanish stopword lists.

### 3. Topic-Aware Filtering
- Rule-based classification of sentences into GDP or inflation categories based on term frequency and context scoring.

### 4. Feature Extraction
- **TF-IDF Matrices** (`tfidf.py`): POS-filtered term weighting by topic and year.
- **Word Embeddings** (`word2vec.py`): Temporal Word2Vec models with PCA-based visualization of semantic drift.
- **Sentiment Analysis** (`sentiment_heuristics.py`): Lexicon-based polarity scoring using economic sentiment dictionaries.
- **Clarity Metrics** (`clarity_metrics.py`): Sentence length, lexical density, and token complexity metrics.
- **Metadata Enrichment** (`metada.py`): Quarterly date inference, top verbs extraction, and indicator tagging.

### 5. Visualization and EDA
- Trend lines, heatmaps, PCA scatterplots, and thematic term evolution graphs (`visualizations.py`, `eda_analysis.py`).

---

## Technical Contributions and Innovations

This project introduces several technical innovations to enable accurate and efficient macroeconomic discourse analysis:

- **Domain-Aware Filtering**: A novel keyword-based priority scoring system isolates GDP and inflation discourse while minimizing global/noisy content.
- **Efficient Preprocessing for Spanish**: Custom cleaning and lemmatization tailored to economic vocabulary enhance term quality and reduce nonsensical outputs.
- **Low-Resource Semantic Drift Detection**: Yearly Word2Vec embeddings capture shifts in concept framing using PCA, without relying on transformer-based models.
- **Robust Lexical Feature Engineering**: Despite its simplicity, the pipeline achieves low noise in TF-IDF outputs (only 4/30 nonsensical terms for GDP, 3/30 for inflation).
- **Clarity and Sentiment Quantification**: Lightweight heuristics measure communicative tone and complexity, offering interpretable insights into institutional communication strategies.

Together, these contributions demonstrate that meaningful analysis of policy narratives can be achieved without expensive infrastructure or large-scale pretrained models.

---

## Outputs

The following visualizations and data products are generated:
- **Sentiment Trends**: Line plots comparing GDP vs. inflation sentiment over time.
- **TF-IDF Heatmaps**: Year-wise importance of terms for each topic.
- **PCA of Word Embeddings**: Semantic drift visualization.
- **Clarity Metrics**: Sentence-level complexity by year and topic.
- **Term Evolution Graphs**: Longitudinal change in top TF-IDF terms.

---

## Usage

### Install dependencies:
```bash
pip install -r requirements.txt
```
### Run pipeline

```bash
python scrape_banxico.py           # Download raw PDFs
python extract_corpus.py           # Extract and filter sentences
python preprocessing.py            # Clean and lemmatize text
python tfidf.py                    # Generate TF-IDF matrices
python sentiment_heuristics.py     # Compute sentiment scores
python clarity_metrics.py          # Measure clarity metrics
python word2vec.py                 # Train and save embeddings
python visualizations.py           # Generate plots
```
