"""
descriptive_stats.py
Generate descriptive statistics tables and plots
for the Internet Advertisements dataset.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PROCESSED_DATA_PATH, PLOTS_DIR, METRICS_DIR, LABEL_COLUMN


def generate_descriptive_stats():
    # -------------------------------------------------
    # Load cleaned dataset
    # -------------------------------------------------
    df = pd.read_csv(PROCESSED_DATA_PATH, header=None)

    # Rename last column as class label
    df.rename(columns={df.columns[-1]: LABEL_COLUMN}, inplace=True)

    # Rename first three columns
    df.rename(columns={
        0: "height",
        1: "width",
        2: "aspect_ratio"
    }, inplace=True)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    numeric_cols = ["height", "width", "aspect_ratio"]

    # -------------------------------------------------
    # 1. Summary statistics table
    # -------------------------------------------------
    stats = df[numeric_cols].describe().T
    stats["median"] = df[numeric_cols].median()

    stats.to_csv(METRICS_DIR / "descriptive_stats.csv")

    # -------------------------------------------------
    # 2. Histograms
    # -------------------------------------------------
    for col in numeric_cols:
        plt.figure(figsize=(7,5))
        plt.hist(df[col], bins=30)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"hist_{col}.png")
        plt.close()

    # -------------------------------------------------
    # 3. Boxplots by class
    # -------------------------------------------------
    for col in numeric_cols:
        plt.figure(figsize=(7,5))
        sns.boxplot(data=df, x=LABEL_COLUMN, y=col)
        plt.title(f"{col} by class")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"box_{col}.png")
        plt.close()

    # -------------------------------------------------
    # 4. Scatter plots
    # -------------------------------------------------
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=df, x="width", y="height", hue=LABEL_COLUMN, alpha=0.5)
    plt.title("Width vs Height")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scatter_width_height.png")
    plt.close()

    plt.figure(figsize=(7,5))
    sns.scatterplot(data=df, x="aspect_ratio", y="height", hue=LABEL_COLUMN, alpha=0.5)
    plt.title("Aspect Ratio vs Height")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scatter_ratio_height.png")
    plt.close()

    # -------------------------------------------------
    # 5. Correlation heatmap
    # -------------------------------------------------
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close()

    print("Descriptive statistics generated.")