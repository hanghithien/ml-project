import os
import numpy as np
from src.config import FIGURE_DIR
from sklearn.metrics import recall_score, precision_score, f1_score

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
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }
    # raise NotImplementedError()


def plot_evaluation_results(y_true, y_prob, model_name="Model", threshold=0.5):
    """
    TODO:
    Draw and save diagram to outputs/figures/.
    You can draw: Precision-Recall Curve, ROC Curve, Confusion Matrix.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    print("Hello")
    # raise NotImplementedError()