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

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, LABEL_COLUMN


def _normalize_raw_layout(df: pd.DataFrame) -> pd.DataFrame:
    """Truncate header rows or index columns accidentally included in the raw CSV."""
    # Detect and drop an accidental header row like: NaN,0,1,2,...
    first_row_numeric = pd.to_numeric(df.iloc[0], errors="coerce") # Locate  the first row string and attempt to convert it to numeric, coercing errors to NaN
    n_check = min(200, len(df.columns))# Check at most the first 200 columns to avoid excessive computation on wide datasets.
    first_row_diffs = first_row_numeric.iloc[:n_check].dropna().diff().dropna()  # Headers often look like a perfect sequence of integers, so check if the first row is mostly a sequence of numbers with a constant step (e.g., 0,1,2,...).
    if len(first_row_diffs) > 0 and (first_row_diffs.eq(1).mean() > 0.95): # If more than 95% of the differences are equal to 1, we can be confident that this row is an accidental header and drop it.
        df = df.iloc[1:].reset_index(drop=True)

    # Detect and drop an accidental index column like: Nan,0,1,2,3,...
    #Same idea as above but applied to the first column instead of the first row. If the first column looks like a perfect sequence of integers, it's likely an accidental index column and we can drop it.
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
    Load the raw CSV, clean all corrupted data and impute missing values, and save the result.

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
    df.rename(columns={df.columns[-1]: LABEL_COLUMN}, inplace=True)

    # ------------------------------------------------------------------
    # 2. Replace "?" (with optional surrounding spaces) with numerical value NaN
    # ------------------------------------------------------------------
    df.replace(r"^\s*\?\s*$", np.nan, regex=True, inplace=True)

    # Cleans label column from whitespace that occured during data loading and could 
    # cause issues during model training and evaluation. 
    # This ensures that all class labels are consistent and free of leading/trailing spaces, which is crucial for accurate classification.
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

    # ------------------------------------------------------------------
    # 3. Convert feature columns to numeric
    # ------------------------------------------------------------------
    feature_cols = [col for col in df.columns if col != LABEL_COLUMN]
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
    df.to_csv(processed_path, index=False, header=False)
    print(f"  Cleaned data saved → {processed_path}")

    return df
