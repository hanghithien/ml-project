from preprocessing import load_and_clean_data, get_train_val_data
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

df_clean = load_and_clean_data('./Data/creditcard.csv')
X, y = get_train_val_data(df_clean)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
    
    scaler = RobustScaler()
    X_train['Log_Amount'] = scaler.fit_transform(X_train[['Log_Amount']])
    X_val['Log_Amount'] = scaler.transform(X_val[['Log_Amount']])
    
    print(f"--- Đang chạy Fold {fold} ---")
    print(f"Kích thước X_train: {X_train.shape}, Tỷ lệ gian lận: {y_train.mean():.4%}")
    ## To do: Implement Model

    
