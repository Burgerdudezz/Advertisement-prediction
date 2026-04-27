"""
evaluate.py – Score the trained model and persist evaluation artefacts.

Artefacts produced
------------------
- outputs/metrics/metrics.csv            – accuracy, precision, recall, F1, ROC-AUC
- outputs/metrics/confusion_matrix.txt   – human-readable confusion matrix
- outputs/metrics/feature_importance.csv – per-feature importances
- outputs/plots/roc_curve.png            – ROC curve for the positive class
- outputs/metrics/predictions.csv        – actual vs predicted labels + probabilities
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import LabelBinarizer

from src.config import MODEL_PATH, METRICS_DIR, PLOTS_DIR


def evaluate(X_test, y_test, model_path=MODEL_PATH):
    """
    Load the saved model, score it on the test set, and write artefacts.

    Parameters
    ----------
    X_test : pd.DataFrame
        Feature matrix for the held-out test set.
    y_test : pd.Series
        True labels for the test set.
    model_path : Path or str
        Location of the saved model (joblib format).
    """
    # Create output directories if they do not exist yet
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model and predict
    # ------------------------------------------------------------------
    clf = joblib.load(model_path)
    y_pred = clf.predict(X_test)

    # ------------------------------------------------------------------
    # 2. Compute classification metrics
    # ------------------------------------------------------------------
    # Determine the positive class label (the one that is NOT "nonad.")
    classes = clf.classes_
    pos_label = next(c for c in classes if "nonad" not in c.lower())

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=pos_label)
    recall    = recall_score(y_test, y_pred, pos_label=pos_label)
    f1        = f1_score(y_test, y_pred, pos_label=pos_label)

    # ROC-AUC: use predicted probabilities for the positive class
    y_prob = clf.predict_proba(X_test)[:, list(classes).index(pos_label)]
    roc_auc = roc_auc_score(
        LabelBinarizer().fit_transform(y_test).ravel(), y_prob
    )

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")

    # ------------------------------------------------------------------
    # 3. Save metrics CSV
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame([{
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }])
    metrics_path = METRICS_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Metrics saved → {metrics_path}")

    # ------------------------------------------------------------------
    # 4. Save confusion matrix as a text file
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_lines = [
        "Confusion Matrix",
        "Rows = True label | Columns = Predicted label",
        f"Classes: {list(classes)}",
        "",
        str(cm),
    ]
    cm_path = METRICS_DIR / "confusion_matrix.txt"
    cm_path.write_text("\n".join(cm_lines))
    print(f"  Confusion matrix saved → {cm_path}")

    # ------------------------------------------------------------------
    # 5. Save feature importances
    # ------------------------------------------------------------------
    feature_names = X_test.columns.tolist()
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    fi_path = METRICS_DIR / "feature_importance.csv"
    importance_df.to_csv(fi_path, index=False)
    print(f"  Feature importances saved → {fi_path}")

    # ------------------------------------------------------------------
    # 6. Save ROC curve plot
    # ------------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=pos_label)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Internet Advertisements")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()

    roc_path = PLOTS_DIR / "roc_curve.png"
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)
    print(f"  ROC curve saved → {roc_path}")

    # ------------------------------------------------------------------
    # 7. Save predictions CSV
    # ------------------------------------------------------------------
    pred_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred,
        "probability": y_prob,
    })

    pred_path = METRICS_DIR / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print(f"  Predictions saved → {pred_path}")
