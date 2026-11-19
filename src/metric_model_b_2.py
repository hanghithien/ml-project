import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURE_DIR


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    """
    Tính các metric đánh giá riêng cho Model B2 - SVM (độc lập với src/metrics.py),
    phù hợp cho bài toán phân loại nhị phân mất cân bằng cực độ.

    Parameters:
        y_true (array-like): Nhãn thực tế (0: Normal, 1: Fraud).
        y_prob (array-like): Xác suất dự đoán thuộc lớp 1.
        threshold (float): Ngưỡng phân loại nhị phân.

    Returns:
        dict: Từ điển chứa các chỉ số đánh giá và ma trận nhầm lẫn.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    y_pred = (y_prob_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred).ravel()

    return {
        "AUPRC": round(float(average_precision_score(y_true_arr, y_prob_arr)), 4),
        "ROC_AUC": round(float(roc_auc_score(y_true_arr, y_prob_arr)), 4),
        "Precision": round(float(precision_score(y_true_arr, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true_arr, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true_arr, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_true_arr, y_pred)), 4),
        "Balanced_Accuracy": round(float(balanced_accuracy_score(y_true_arr, y_pred)), 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def plot_evaluation_results(y_true, y_prob, model_name="Model_B_2", threshold=0.5):
    """
    Vẽ và lưu biểu đồ đánh giá riêng cho Model B2 - SVM (Precision-Recall Curve,
    ROC Curve, Confusion Matrix) vào outputs/figures/.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    y_pred = (y_prob_arr >= threshold).astype(int)

    auprc = average_precision_score(y_true_arr, y_prob_arr)
    roc_auc = roc_auc_score(y_true_arr, y_prob_arr)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
    fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
    cm = confusion_matrix(y_true_arr, y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Precision-Recall Curve
    axes[0].plot(recalls, precisions, color="teal", lw=2, label=f"AUPRC = {auprc:.4f}")
    if len(pr_thresholds) > 0:
        idx = int(np.argmin(np.abs(pr_thresholds - threshold)))
        axes[0].scatter(
            recalls[idx],
            precisions[idx],
            color="crimson",
            s=60,
            zorder=5,
            label=f"Threshold = {threshold:.2f}",
        )
    baseline_rate = float(np.mean(y_true_arr))
    axes[0].axhline(
        baseline_rate,
        color="gray",
        linestyle="--",
        lw=1,
        label=f"Baseline = {baseline_rate:.4f}",
    )
    axes[0].set_title(f"{model_name} - Precision-Recall Curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    # 2. ROC Curve
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    axes[1].set_title(f"{model_name} - ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right")

    # 3. Confusion Matrix
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
    axes[2].set_title(f"{model_name} - Confusion Matrix (t={threshold:.2f})")
    axes[2].set_xlabel("Predicted Label")
    axes[2].set_ylabel("True Label")

    plt.tight_layout()
    fig_path = FIGURE_DIR / f"{model_name.lower()}_evaluation.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evaluation figure to: {fig_path}")
