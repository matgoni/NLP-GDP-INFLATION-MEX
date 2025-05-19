# Generate visualizations for TF-IDF, embeddings, clarity, and sentiment

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_tfidf_heatmap(csv_path, title, output_path):
    df = pd.read_csv(csv_path, index_col=0)
    plt.figure(figsize=(14, 6))
    sns.heatmap(df.T, cmap="viridis", annot=False)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Key Terms")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_embeddings_pca(csv_path, topic, output_path):
    df = pd.read_csv(csv_path, index_col=0)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(df.values)

    plt.figure(figsize=(8, 6))
    for i, year in enumerate(df.index):
        plt.scatter(reduced[i, 0], reduced[i, 1], label=str(year))
        plt.text(reduced[i, 0], reduced[i, 1], str(year))
    plt.title(f"PCA of Average Word Embeddings ({topic.upper()})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_metric_line(csv_path, metric, title, output_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="year", y=metric, hue="topic", marker="o")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(metric.replace("_", " ").capitalize())
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    plot_tfidf_heatmap(
        "data/features/tfidf/tfidf_gdp.csv",
        "TF-IDF of Key Terms by Year (GDP)",
        "figures/tfidf_gdp_heatmap.png"
    )
    plot_tfidf_heatmap(
        "data/features/tfidf/tfidf_inflation.csv",
        "TF-IDF of Key Terms by Year (Inflation)",
        "figures/tfidf_inflation_heatmap.png"
    )
    plot_embeddings_pca(
        "data/features/embeddings/embeddings_gdp.csv",
        "gdp",
        "figures/pca_gdp_embeddings.png"
    )
    plot_embeddings_pca(
        "data/features/embeddings/embeddings_inflation.csv",
        "inflation",
        "figures/pca_inflation_embeddings.png"
    )
    plot_metric_line(
        "data/features/clarity/clarity_metrics.csv",
        "avg_tokens_per_sentence",
        "Clarity Over Time (Average Tokens per Sentence)",
        "figures/clarity_avg_tokens.png"
    )
    plot_metric_line(
        "data/features/sentiment/sentiment_heuristics.csv",
        "sentiment_score",
        "Sentiment Score Over Time",
        "figures/sentiment_score.png"
    )

    print("Visualizations saved in 'figures/' folder.")
