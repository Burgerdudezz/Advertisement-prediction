"""
preprocess.py – Load and clean the raw UCI Internet Advertisements CSV.

Steps
-----
1. Load the CSV (no header row – column names are assigned automatically).
2. Replace "?" strings with NaN.
3. Infer numeric columns where possible.
4. Impute missing numeric values with the column median.
5. Save the cleaned DataFrame to data/processed/cleaned.csv.
"""

import pandas as pd
import numpy as np

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN


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
    # 1. Load – the UCI file has no header, so pandas assigns 0, 1, 2 …
    # ------------------------------------------------------------------
    df = pd.read_csv(raw_path, header=None, low_memory=False)

    # Rename the last column to the target name defined in config
    df.rename(columns={df.columns[-1]: TARGET_COLUMN}, inplace=True)

    # ------------------------------------------------------------------
    # 2. Replace "?" with NaN
    # ------------------------------------------------------------------
    df.replace("?", np.nan, inplace=True)

    # ------------------------------------------------------------------
    # 3. Infer numeric types (columns that look like numbers become float)
    # ------------------------------------------------------------------
    for col in df.columns:
        if col != TARGET_COLUMN:
            converted = pd.to_numeric(df[col], errors="coerce")
            # Only adopt the converted column when no new NaN values are
            # introduced (i.e. the column really is numeric).
            if converted.isna().sum() <= df[col].isna().sum():
                df[col] = converted

    # ------------------------------------------------------------------
    # 4. Impute missing numeric values with the column median
    # ------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # ------------------------------------------------------------------
    # 5. Save cleaned data
    # ------------------------------------------------------------------
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"  Cleaned data saved → {processed_path}")

    return df
