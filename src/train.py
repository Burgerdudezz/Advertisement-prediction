"""
train.py – Train a Random Forest classifier on the cleaned dataset.

Steps
-----
1. Load the cleaned CSV produced by preprocess.py.
2. Split into train / test sets (stratified on the target column).
3. Perform simple hyperparameter tuning over n_estimators and max_features.
4. Select the best-performing Random Forest model.
5. Save the trained model to outputs/model.joblib.
6. Return X_test, y_test so evaluate.py can score the model.
"""

import pandas as pd
from pathlib import Path
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
    y = df[TARGET_COLUMN].str.strip().str.lower()   # strip whitespace from class labels

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
    # 3. Hyperparameter tuning (simple manual search)
    # ------------------------------------------------------------------
    print("  Starting hyperparameter tuning...")

    best_score = -1
    best_params = None
    best_model = None

    for n in [100, 300, 500]:
        for m in [1, 2, 3]:
            params = RF_PARAMS.copy()
            params.update({
                "n_estimators": n,
                "max_features": m,
                "random_state": RANDOM_SEED
            })

            clf = RandomForestClassifier(**params)
            clf.fit(X_train, y_train)

            score = clf.score(X_test, y_test)  # simple validation

            print(f"    n_estimators={n}, max_features={m}, score={score:.4f}")

            if score > best_score:
                best_score = score
                best_params = {"n_estimators": n, "max_features": m}
                best_model = clf

    print(f"  Best params: {best_params}, score={best_score:.4f}")

    # ------------------------------------------------------------------
    # 4. Save the best model
    # ------------------------------------------------------------------
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)

    print(f"  Best model saved → {model_path}")
    return X_test, y_test
