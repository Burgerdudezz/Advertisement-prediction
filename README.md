# Internet Advertisements – Random Forest Classifier

A beginner-friendly machine learning project that predicts whether a web image is an **advertisement** or not, using the [UCI Internet Advertisements dataset](https://www.kaggle.com/datasets/uciml/internet-advertisements-data-set) and a Random Forest classifier.

---

## Objective

Build a binary classifier that labels web images as advertisements (`ad.`) or non-ads (`nonad.`) using the UCI Internet Advertisements dataset.

---

## Dataset

- **Source:** UCI Internet Advertisements dataset (linked above).
- **Task:** Binary classification (`ad.` vs `nonad.`).
- **Input:** High-dimensional numeric features derived from image and URL properties.

---

## Project Structure

```
.
├── data/
│   ├── raw/           # Place the downloaded CSV here (not tracked by git)
│   └── processed/     # Cleaned data is written here automatically
├── outputs/
│   ├── metrics/       # metrics.csv, confusion_matrix.txt, feature_importance.csv
│   ├── plots/         # roc_curve.png
│   └── predictions/   # (reserved for future use)
├── notebooks/
│   └── quick_eda.ipynb
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── descriptive_stats.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## Setup

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd Advertisement-prediction
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**

   Download `add.data` (or rename it to `add.csv`) from Kaggle and place it in `data/raw/`.
   Update `RAW_DATA_PATH` in `src/config.py` if your filename differs.

---

## Running the Pipeline

```bash
python -m src.run_pipeline
```

This single command runs the full pipeline:
1. **Preprocess** – cleans the raw CSV and writes `data/processed/cleaned.csv`
2. **EDA** – generates exploratory plots in `outputs/plots/`
3. **Descriptive stats** – saves summary tables and plots in `outputs/metrics/` and `outputs/plots/`
4. **Train** – trains a Random Forest and saves `outputs/model.joblib`
5. **Evaluate** – scores the model and writes results to `outputs/`

---

## Expected Output Files

| Path | Description |
|------|-------------|
| `data/processed/cleaned.csv` | Cleaned dataset |
| `outputs/model.joblib` | Trained Random Forest model |
| `outputs/metrics/metrics.csv` | Accuracy, Precision, Recall, F1, ROC-AUC |
| `outputs/metrics/confusion_matrix.txt` | Confusion matrix |
| `outputs/metrics/feature_importance.csv` | Per-feature importances |
| `outputs/metrics/threshold_metrics.csv` | Precision/Recall/F1 over thresholds |
| `outputs/metrics/predictions.csv` | Predictions + probabilities |
| `outputs/plots/roc_curve.png` | ROC curve plot |
| `outputs/plots/pr_curve.png` | Precision-recall curve plot |
| `outputs/plots/confusion_matrix.png` | Confusion matrix heatmap |
| `outputs/plots/threshold_tradeoff.png` | Threshold tradeoff curves |
| `outputs/plots/class_distribution.png` | Actual vs predicted class counts |
| `outputs/plots/feature_importance_top20.png` | Top feature importances |

---

## Preprocessing Summary

- Handles accidental header/index rows in the raw CSV.
- Replaces missing values marked as `?` with `NaN` and imputes numeric means.
- Strips whitespace from class labels to keep them consistent.

---

## Model

- **Algorithm:** Random Forest classifier (scikit-learn).
- **Rationale:** Robust baseline for high-dimensional numeric features and imbalanced classes.
- **Tuning:** Small grid over `n_estimators` and `max_features`.

---

## Evaluation

- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
- **Diagnostics:** Confusion matrix, ROC/PR curves, class distribution, threshold tradeoff plots.

---

## Reproducibility

- Random seed, train/test split, and model settings are centralized in `src/config.py`.
- Main entrypoint: `python -m src.run_pipeline`.

---

## Limitations and Future Work

- The model uses a simple hyperparameter search; broader tuning may improve performance.
- Feature engineering is minimal; additional domain features could help.
- Consider calibration or alternative models (e.g., gradient boosting) for better probabilistic outputs.

---

## References

- UCI Internet Advertisements dataset (linked above)
- scikit-learn documentation

---

## Configuration

All tunable settings live in `src/config.py`. Edit that file to change the data path,
random seed, train/test split ratio, or Random Forest hyperparameters.
