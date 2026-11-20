import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURE_DIR
from sklearn.metrics import precision_recall_curve, recall_score, precision_score, f1_score, average_precision_score, confusion_matrix

def evaluate_predictions(y_true, y_prob, threshold=0.5):
    """
    TODO:
    Choose suitable metric evalution for problem.

    Parameter:
        y_true (array-like): Real-label (0: Normal, 1: Fraud).
        y_prob (array-like): Probability of class 1.
        threshold (float): Thresh hold for classification

    Trả về:
        dict: dictionary contain metric and value corresponding.
              Ex: {"AUPRC": 0.8215, "F1": 0.7901, "Recall": 0.8100}
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "AUPRC": round(float(average_precision_score(y_true, y_prob)), 3),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }
    # raise NotImplementedError()


def plot_evaluation_results(y_true, y_prob, model_name="Model", threshold=0.5):
    """
    TODO:
    Draw and save diagram to outputs/figures/.
    You can draw: Precision-Recall Curve, ROC Curve, Confusion Matrix.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    auprc = average_precision_score(y_true, y_prob)
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Precision-Recall Curve
    axes[0].plot(recalls, precisions, lw=2, label=f"AUPRC = {auprc:.4f}")
    axes[0].set_title(f"{model_name} - Precision-Recall Curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    # 2. Confusion Matrix
    sns.heatmap(cm.T, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Fraud (1)", "Normal (0)"],
                yticklabels=["Fraud (1)", "Normal (0)"], ax=axes[1])
    axes[1].set_title(f"{model_name} - Confusion Matrix")
    axes[1].set_xlabel("True")
    axes[1].set_ylabel("Predicted")

    plt.tight_layout()
    fig_path = FIGURE_DIR / f"{model_name.lower()}_evaluation.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evaluation figure to: {fig_path}")
    # raise NotImplementedError()