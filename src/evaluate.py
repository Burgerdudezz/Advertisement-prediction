"""
evaluate.py – Score the trained model and persist evaluation artefacts.

Artefacts produced
------------------
- outputs/metrics/metrics.csv            – accuracy, precision, recall, F1, ROC-AUC
- outputs/metrics/confusion_matrix.txt   – human-readable confusion matrix
- outputs/metrics/threshold_metrics.csv  – precision/recall/F1 over thresholds
- outputs/metrics/feature_importance.csv – per-feature importances
- outputs/plots/roc_curve.png            – ROC curve for the positive class
- outputs/plots/pr_curve.png             – precision-recall curve
- outputs/plots/confusion_matrix.png     – confusion matrix heatmap
- outputs/plots/probability_histogram.png – score distribution by true class
- outputs/plots/threshold_tradeoff.png   – precision/recall/F1 vs threshold
- outputs/plots/class_distribution.png   – actual vs predicted class counts
- outputs/plots/feature_importance_top20.png – top 20 feature importances
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
    precision_recall_curve,
    average_precision_score,
)

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
    neg_label = next(c for c in classes if c != pos_label)

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=pos_label)
    recall    = recall_score(y_test, y_pred, pos_label=pos_label)
    f1        = f1_score(y_test, y_pred, pos_label=pos_label)

    # ROC-AUC: align y_true with the same positive class used for y_prob.
    y_prob = clf.predict_proba(X_test)[:, list(classes).index(pos_label)]
    y_true_bin = (y_test == pos_label).astype(int)
    roc_auc = roc_auc_score(y_true_bin, y_prob)
    avg_precision = average_precision_score(y_true_bin, y_prob)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"  PR-AUC   : {avg_precision:.4f}")

    # ------------------------------------------------------------------
    # 3. Save metrics CSV
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame({
        "metric": ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
        "value": [
            f"{acc:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            f"{roc_auc:.4f}",
            f"{avg_precision:.4f}",
        ],
    })
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
    # 6.1 Save top-20 feature importances as a bar chart
    # ------------------------------------------------------------------
    top_n = 20
    top_importance = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_importance["feature"].astype(str), top_importance["importance"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()

    fi_plot_path = PLOTS_DIR / "feature_importance_top20.png"
    fig.savefig(fi_plot_path, dpi=120)
    plt.close(fig)
    print(f"  Feature importance plot saved → {fi_plot_path}")

    # ------------------------------------------------------------------
    # 6.2 Save ROC curve plot
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
    # 6.3 Save Precision-Recall curve plot
    # ------------------------------------------------------------------
    pr_precision, pr_recall, _ = precision_recall_curve(y_true_bin, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pr_recall, pr_precision, lw=2, label=f"PR curve (AP = {avg_precision:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve – Internet Advertisements")
    ax.legend(loc="lower left")
    ax.grid(True)
    plt.tight_layout()

    pr_path = PLOTS_DIR / "pr_curve.png"
    fig.savefig(pr_path, dpi=120)
    plt.close(fig)
    print(f"  PR curve saved → {pr_path}")

    # ------------------------------------------------------------------
    # 6.4 Save confusion matrix heatmap
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    cm_plot_path = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_plot_path, dpi=120)
    plt.close(fig)
    print(f"  Confusion matrix plot saved → {cm_plot_path}")

    # ------------------------------------------------------------------
    # 6.5 Save probability histogram by true class
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(y_prob[y_test == neg_label], bins=bins, alpha=0.6, label=f"True {neg_label}")
    ax.hist(y_prob[y_test == pos_label], bins=bins, alpha=0.6, label=f"True {pos_label}")
    ax.set_xlabel(f"Predicted probability of {pos_label}")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distribution by True Class")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    hist_path = PLOTS_DIR / "probability_histogram.png"
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)
    print(f"  Probability histogram saved → {hist_path}")

    # ------------------------------------------------------------------
    # 6.6 Save threshold tradeoff table and plot
    # ------------------------------------------------------------------
    threshold_grid = np.linspace(0, 1, 101)
    rows = []
    for thr in threshold_grid:
        y_pred_thr = np.where(y_prob >= thr, pos_label, neg_label)
        rows.append({
            "threshold": thr,
            "precision": precision_score(y_test, y_pred_thr, pos_label=pos_label, zero_division=0),
            "recall": recall_score(y_test, y_pred_thr, pos_label=pos_label, zero_division=0),
            "f1": f1_score(y_test, y_pred_thr, pos_label=pos_label, zero_division=0),
        })

    threshold_df = pd.DataFrame(rows)
    threshold_path = METRICS_DIR / "threshold_metrics.csv"
    threshold_df.to_csv(threshold_path, index=False)
    print(f"  Threshold metrics saved → {threshold_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
    ax.plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
    ax.axvline(0.5, color="k", linestyle="--", linewidth=1, label="Default threshold")
    ax.set_xlabel("Decision threshold for ad")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Tradeoff Curves")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    thr_plot_path = PLOTS_DIR / "threshold_tradeoff.png"
    fig.savefig(thr_plot_path, dpi=120)
    plt.close(fig)
    print(f"  Threshold tradeoff plot saved → {thr_plot_path}")

    # ------------------------------------------------------------------
    # 6.7 Save class distribution comparison plot
    # ------------------------------------------------------------------
    actual_counts = pd.Series(y_test).value_counts().reindex(classes, fill_value=0)
    pred_counts = pd.Series(y_pred).value_counts().reindex(classes, fill_value=0)

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, actual_counts.values, width, label="Actual")
    ax.bar(x + width / 2, pred_counts.values, width, label="Predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution: Actual vs Predicted")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    dist_path = PLOTS_DIR / "class_distribution.png"
    fig.savefig(dist_path, dpi=120)
    plt.close(fig)
    print(f"  Class distribution plot saved → {dist_path}")

    # ------------------------------------------------------------------
    # 7. Save predictions CSV
    # ------------------------------------------------------------------
    pred_df = pd.DataFrame({
        "source_index": y_test.index,
        "actual": y_test.values,
        "predicted": y_pred,
        "probability": y_prob,
    })

    pred_path = METRICS_DIR / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print(f"  Predictions saved → {pred_path}")
