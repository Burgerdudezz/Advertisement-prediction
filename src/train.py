"""
train.py – Train a Random Forest classifier on the cleaned dataset.

Steps
-----
1. Load the cleaned CSV produced by preprocess.py.
2. Split into train / test sets (stratified on the target column).
3. Fit a RandomForestClassifier with the hyperparameters from config.py.
4. Save the trained model to outputs/model.joblib.
5. Return X_test, y_test so evaluate.py can score the model.
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_SEED,
    RF_PARAMS,
)


def train(processed_path=PROCESSED_DATA_PATH, model_path=MODEL_PATH):
    """
    Load cleaned data, train a Random Forest, and persist the model.

    Parameters
    ----------
    processed_path : Path or str
        Cleaned CSV produced by preprocess.py.
    model_path : Path or str
        Where to save the trained model (joblib format).

    Returns
    -------
    tuple
        (X_test, y_test) – held-out test data for evaluation.
    """
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(processed_path)

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].str.strip()   # strip whitespace from class labels

    # ------------------------------------------------------------------
    # 2. Train / test split (stratified to preserve class ratios)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"  Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ------------------------------------------------------------------
    # 3. Fit the model
    # ------------------------------------------------------------------
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    print("  Model training complete.")

    # ------------------------------------------------------------------
    # 4. Save the model
    # ------------------------------------------------------------------
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"  Model saved → {model_path}")

    return X_test, y_test
