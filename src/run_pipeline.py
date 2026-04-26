"""
run_pipeline.py – Orchestrate the full ML pipeline.

Usage
-----
    python -m src.run_pipeline

Steps executed
--------------
1. Preprocess  – clean raw CSV → data/processed/cleaned.csv
2. Train       – fit Random Forest → outputs/model.joblib
3. Evaluate    – compute metrics and plots → outputs/metrics/ & outputs/plots/
"""

from src.preprocess import load_and_clean
from src.train import train
from src.evaluate import evaluate


def main():
    print("=" * 50)
    print("Internet Ads – Random Forest Pipeline")
    print("=" * 50)

    # ------------------------------------------------------------------
    # Step 1 – Preprocess
    # ------------------------------------------------------------------
    print("\n[1/3] Preprocessing raw data …")
    load_and_clean()

    # ------------------------------------------------------------------
    # Step 2 – Train
    # ------------------------------------------------------------------
    print("\n[2/3] Training the Random Forest …")
    X_test, y_test = train()

    # ------------------------------------------------------------------
    # Step 3 – Evaluate
    # ------------------------------------------------------------------
    print("\n[3/3] Evaluating the model …")
    evaluate(X_test, y_test)

    print("\n" + "=" * 50)
    print("Pipeline complete!  Check the outputs/ directory.")
    print("=" * 50)


if __name__ == "__main__":
    main()
