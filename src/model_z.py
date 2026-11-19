import numpy as np
from sklearn.preprocessing import RobustScaler

from src.config import RAW_DATA_PATH, RANDOM_STATE, TEST_SIZE, COLS_TO_SCALE
from src.preprocess import load_and_clean_data, split_train_test, get_cv_splitter
from src.metrics import evaluate_predictions, plot_evaluation_results

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve

def handle_imbalance(X_train_fold, y_train_fold):
    """
    TODO (Optional): Choose technique to handle imbalance data
    if use class_weight='balanced' directly in model, return default.
    """
    return X_train_fold, y_train_fold

def build_model(contamination='auto'):
    return IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    raise NotImplementedError("Implement not found")
def get_anomaly_scores(model, X):
    return model.decision_function(X)

def find_best_threshold(y_true, scores):
    # Tìm ngưỡng tối ưu F1-score trên tập validation
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.0

def run_model_b():
    # Load data preprocessing 
    df_clean = load_and_clean_data(RAW_DATA_PATH)
    X_train_full, X_test, y_train_full, y_test = split_train_test(
        df_clean, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    skf = get_cv_splitter()
    cv_metrics_list = []
    fold_thresholds = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
        X_train = X_train_full.iloc[train_idx].copy()
        y_train = y_train_full.iloc[train_idx].copy()
        X_val = X_train_full.iloc[val_idx].copy()
        y_val = y_train_full.iloc[val_idx].copy()


        scaler = RobustScaler()
        X_train[COLS_TO_SCALE] = scaler.fit_transform(X_train[COLS_TO_SCALE])
        X_val[COLS_TO_SCALE] = scaler.transform(X_val[COLS_TO_SCALE])

        # =====================================================================
        # Todo AnyThing
        # =====================================================================
        X_train_res, y_train_res = handle_imbalance(X_train, y_train)
        model = build_model(contamination=float(y_train.mean()))
        model.fit(X_train_res)

        y_val_prob = model.predict_proba(X_val)[:, 1]
        # =====================================================================
        y_val_scores = get_anomaly_scores(model, X_val)
        fold_thresh = find_best_threshold(y_val, y_val_scores)
        fold_thresholds.append(fold_thresh)
        fold_metrics = evaluate_predictions(y_val, y_val_scores, threshold=fold_thresh)
        cv_metrics_list.append(fold_metrics)
        print(f"Fold {fold} (thresh={fold_thresh:.4f}) -> {fold_metrics}")

    # =========================================================================
    # Evaluation on test
    # =========================================================================
    final_scaler = RobustScaler()
    X_train_scaled = X_train_full.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[COLS_TO_SCALE] = final_scaler.fit_transform(X_train_scaled[COLS_TO_SCALE])
    X_test_scaled[COLS_TO_SCALE] = final_scaler.transform(X_test_scaled[COLS_TO_SCALE])

    X_train_res, y_train_res = handle_imbalance(X_train_scaled, y_train_full)
    final_model = build_model()
    final_model.fit(X_train_res)

    # Adjust Threshold CV (Cross validation) if needed
    best_threshold = float(np.mean(fold_thresholds))
    y_test_scores = get_anomaly_scores(final_model, X_test_scaled)
    test_metrics = evaluate_predictions(y_test, y_test_scores, threshold=best_threshold)

    print(f"TEST RESULT (MODEL B - Isolation Forest): {test_metrics}")
    plot_evaluation_results(y_test, y_test_scores, model_name="Model_B_IsolationForest")
    return final_model, test_metrics


if __name__ == "__main__":
    run_model_b()
