import os
from src.config import FIGURE_DIR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)
import numpy as np


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    auprc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "AUPRC": round(float(auprc), 4),
        "ROC_AUC": round(float(roc_auc), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1": round(float(f1), 4),
    }
    raise NotImplementedError()


def plot_evaluation_results(y_true, y_prob, model_name="Model", threshold=0.5):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Precision-Recall Curve (Chỉ số quan trọng nhất cho dữ liệu mất cân bằng)
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    axes[0].plot(recalls, precisions, color="darkorange", lw=2, label=f"AUPRC = {auprc:.4f}")
    axes[0].set_title(f"{model_name} - Precision-Recall Curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend(loc="lower left")
    axes[0].grid(True, alpha=0.3)

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    axes[1].set_title(f"{model_name} - ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    # 3. Confusion Matrix tại ngưỡng threshold
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Normal (0)", "Fraud (1)"],
        yticklabels=["Normal (0)", "Fraud (1)"],
        ax=axes[2],
    )
    axes[2].set_title(f"{model_name} - Confusion Matrix (th={threshold:.4f})")
    axes[2].set_xlabel("Predicted Label")
    axes[2].set_ylabel("True Label")

    plt.tight_layout()
    save_path = FIGURE_DIR / f"{model_name}_evaluation.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    raise NotImplementedError()