# Internet Advertisements – Random Forest Classifier

A beginner-friendly machine learning project that predicts whether a web image is an **advertisement** or not, using the [UCI Internet Advertisements dataset](https://www.kaggle.com/datasets/uciml/internet-advertisements-data-set) and a Random Forest classifier.

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

   Download `ad.data` (or rename it to `ad.csv`) from Kaggle and place it in `data/raw/`.
   Update `RAW_DATA_PATH` in `src/config.py` if your filename differs.

---

## Running the Pipeline

```bash
python -m src.run_pipeline
```

This single command runs the full pipeline:
1. **Preprocess** – cleans the raw CSV and writes `data/processed/cleaned.csv`
2. **Train** – trains a Random Forest and saves `outputs/model.joblib`
3. **Evaluate** – scores the model and writes results to `outputs/`

---

## Expected Output Files

| Path | Description |
|------|-------------|
| `data/processed/cleaned.csv` | Cleaned dataset |
| `outputs/model.joblib` | Trained Random Forest model |
| `outputs/metrics/metrics.csv` | Accuracy, Precision, Recall, F1, ROC-AUC |
| `outputs/metrics/confusion_matrix.txt` | Confusion matrix |
| `outputs/metrics/feature_importance.csv` | Per-feature importances |
| `outputs/plots/roc_curve.png` | ROC curve plot |

---

## Configuration

All tuneable settings live in `src/config.py`. Edit that file to change the data path,
random seed, train/test split ratio, or Random Forest hyperparameters.
