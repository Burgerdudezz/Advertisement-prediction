"""
preprocess.py – Load and clean the raw UCI Internet Advertisements CSV.

Steps
-----
1. Load the CSV (no header row – column names are assigned automatically).
2. Replace "?" strings with NaN.
3. Infer numeric columns where possible.
4. Impute missing numeric values with the column mean.
5. Save the cleaned DataFrame to data/processed/cleaned.csv.
"""

import pandas as pd
import numpy as np

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN


def _normalize_raw_layout(df: pd.DataFrame) -> pd.DataFrame:
    """Handle exported CSV artefacts (header row and index column) when present."""
    # Detect and drop an accidental header row like: NaN,0,1,2,...
    first_row_numeric = pd.to_numeric(df.iloc[0], errors="coerce")
    first_row_diffs = first_row_numeric.dropna().diff().dropna()
    if len(first_row_diffs) > 0 and (first_row_diffs.eq(1).mean() > 0.95):
        df = df.iloc[1:].reset_index(drop=True)

    # Detect and drop an accidental index column like: 0,1,2,3,...
    first_col_numeric = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    n_check = min(200, len(df))
    if n_check > 0:
        expected = np.arange(n_check)
        observed = first_col_numeric.iloc[:n_check].to_numpy()
        if np.mean(observed == expected) > 0.95:
            df = df.iloc[:, 1:]

    return df.reset_index(drop=True)


def load_and_clean(raw_path=RAW_DATA_PATH, processed_path=PROCESSED_DATA_PATH):
    """
    Load the raw CSV, clean it, and save the result.

    Parameters
    ----------
    raw_path : Path or str
        Location of the raw ad.csv file.
    processed_path : Path or str
        Where to write the cleaned CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for modelling.
    """
    # ------------------------------------------------------------------
    # 1. Load and normalize layout
    # ------------------------------------------------------------------
    df = pd.read_csv(raw_path, header=None, low_memory=False)
    df = _normalize_raw_layout(df)

    # Reassign deterministic numeric column labels, then rename target.
    df.columns = range(df.shape[1])
    df.rename(columns={df.columns[-1]: TARGET_COLUMN}, inplace=True)

    # ------------------------------------------------------------------
    # 2. Replace "?" (with optional surrounding spaces) with NaN
    # ------------------------------------------------------------------
    df.replace(r"^\s*\?\s*$", np.nan, regex=True, inplace=True)

    # Normalize target label formatting.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()

    # ------------------------------------------------------------------
    # 3. Convert feature columns to numeric
    # ------------------------------------------------------------------
    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # 4. Impute missing numeric values with the column mean
    # ------------------------------------------------------------------
    means = df[feature_cols].mean(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(means)

    # ------------------------------------------------------------------
    # 5. Save cleaned data
    # ------------------------------------------------------------------
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"  Cleaned data saved → {processed_path}")

    return df
