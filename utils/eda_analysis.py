# Exploratory Data Analysis (EDA) on TF-IDF outputs
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def term_frequency_analysis(tfidf_csv_path, top_n=30):
    df = pd.read_csv(tfidf_csv_path, index_col=0)
    term_totals = df.sum(axis=0).sort_values(ascending=False)
    top_terms = term_totals.head(top_n)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_terms.values, y=top_terms.index)
    plt.title(f"Top {top_n} TF-IDF Terms")
    plt.xlabel("TF-IDF Score (Summed Over Years)")
    plt.ylabel("Term")
    plt.tight_layout()
    os.makedirs("figures/eda", exist_ok=True)
    plt.savefig("figures/eda/top_terms_tfidf.png")
    plt.close()


def visualize_topic_evolution(tfidf_csv_path):
    df = pd.read_csv(tfidf_csv_path, index_col=0)
    top_terms = df.sum().sort_values(ascending=False).head(15).index
    df_filtered = df[top_terms]

    plt.figure(figsize=(12, 6))
    for term in top_terms:
        plt.plot(df_filtered.index, df_filtered[term], label=term, marker='o')
    plt.title("Thematic Term Evolution Over Time")
    plt.xlabel("Year")
    plt.ylabel("TF-IDF Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/eda/topic_evolution.png")
    plt.close()


if __name__ == "__main__":
    # Run EDA steps
    term_frequency_analysis("data/features/tfidf/tfidf_gdp.csv")
    visualize_topic_evolution("data/features/tfidf/tfidf_gdp.csv")
