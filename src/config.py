"""
config.py – Central configuration for the Internet Ads pipeline.

Edit the values here to tune paths, the random seed, the train/test
split ratio, and the Random Forest hyperparameters.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # repository root

RAW_DATA_PATH      = BASE_DIR / "data" / "raw"  / "add.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned.csv"

MODEL_PATH              = BASE_DIR / "outputs" / "model.joblib"
METRICS_DIR             = BASE_DIR / "outputs" / "metrics"
PLOTS_DIR               = BASE_DIR / "outputs" / "plots"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2          # 20 % of data goes to the test set

# ---------------------------------------------------------------------------
# Target column name (last column in the UCI dataset)
# ---------------------------------------------------------------------------
TARGET_COLUMN = "class"

# ---------------------------------------------------------------------------
# Random Forest hyperparameters
# ---------------------------------------------------------------------------
RF_PARAMS = {
    "n_estimators":    200,       # number of trees
    "max_depth":       None,      # grow trees to full depth (set an int to limit)
    "min_samples_split": 2,       # minimum samples required to split a node
    "class_weight":    "balanced", # compensate for class imbalance
    "random_state":    RANDOM_SEED,
    "n_jobs":          -1,        # use all CPU cores
}
