import numpy as np
from sklearn.preprocessing import RobustScaler

from src.config import RAW_DATA_PATH, RANDOM_STATE, TEST_SIZE, COLS_TO_SCALE
from src.preprocess import load_and_clean_data, split_train_test, get_cv_splitter
from src.metrics import evaluate_predictions, plot_evaluation_results

from xgboost import XGBClassifier

def handle_imbalance(X_train_fold, y_train_fold):
    """
    TODO (Optional): Choose technique to handle imbalance data
    if use class_weight='balanced' directly in model, return default.
    """
    return X_train_fold, y_train_fold

def build_model():
    """
    TODO: Build Your Model.
    """
    try:
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        return model
    except:
        raise NotImplementedError("Implement not found")


def run_model_ensemble(model = build_model(), data = load_and_clean_data(RAW_DATA_PATH)):
    # Load data preprocessing 
    # df_clean = load_and_clean_data(RAW_DATA_PATH)
    X_train_full, X_test, y_train_full, y_test = split_train_test(
        data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    skf = get_cv_splitter()
    cv_metrics_list = []

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
        model.fit(X_train_res, y_train_res)

        y_val_prob = model.predict_proba(X_val)[:, 1]
        # =====================================================================

        fold_metrics = evaluate_predictions(y_val, y_val_prob)
        cv_metrics_list.append(fold_metrics)
        print(f"Fold {fold} -> {fold_metrics}")

    # =========================================================================
    # Evaluation on test
    # =========================================================================
    final_scaler = RobustScaler()
    X_train_scaled = X_train_full.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[COLS_TO_SCALE] = final_scaler.fit_transform(X_train_scaled[COLS_TO_SCALE])
    X_test_scaled[COLS_TO_SCALE] = final_scaler.transform(X_test_scaled[COLS_TO_SCALE])

    X_train_res, y_train_res = handle_imbalance(X_train_scaled, y_train_full)
    final_model = model
    final_model.fit(X_train_res, y_train_res)

    # Adjust Threshold CV (Cross validation) if needed
    best_threshold = 0.5
    y_test_prob = final_model.predict_proba(X_test_scaled)[:, 1]

    test_metrics = evaluate_predictions(y_test, y_test_prob, threshold=best_threshold)
    print(f"TEST RESULT (MODEL ENSEMBLE): {test_metrics}")
    plot_evaluation_results(y_test, y_test_prob, model_name="Model_ENSEMBLE")
    return final_model, test_metrics


if __name__ == "__main__":
    run_model_ensemble()
