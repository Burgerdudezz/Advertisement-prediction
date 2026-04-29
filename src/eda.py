import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PROCESSED_DATA_PATH, PLOTS_DIR, LABEL_COLUMN


def generate_eda():
    df = pd.read_csv(PROCESSED_DATA_PATH, header=None)
    df.rename(columns={df.columns[-1]: LABEL_COLUMN}, inplace=True)

    # rename first 3 columns
    df.rename(columns={
        0: "height",
        1: "width",
        2: "aspect_ratio"
    }, inplace=True)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # BOXPLOTS BY CLASS
    # -------------------------
    for col in ["height", "width", "aspect_ratio"]:
        plt.figure(figsize=(7,5))
        sns.boxplot(data=df, x=LABEL_COLUMN, y=col)
        plt.title(f"{col} by class")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"box_{col}_class.png")
        plt.close()

    # -------------------------
    # SCATTER PLOTS
    # -------------------------
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

    # -------------------------
    # CORRELATION HEATMAP
    # -------------------------
    corr = df[["height","width","aspect_ratio"]].corr()

    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close()

    print("EDA figures generated.")