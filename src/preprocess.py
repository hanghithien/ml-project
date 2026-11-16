import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def load_and_clean_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    df = df.drop_duplicates(keep='first').copy()
    df['Hour'] = (df['Time'] // 3600) % 24
    df['Log_Amount'] = np.log1p(df['Amount'])
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

    return df

def get_train_val_data(cleaned_df):
    X = cleaned_df.drop(columns=['Class'])
    y = cleaned_df['Class']
    return X, y

if __name__ == "__main__":
    RAW_DATA_PATH = './data/creditcard.csv'
    OUTPUT_PATH = './data/creditcard_cleaned.csv'
    
    df_clean = load_and_clean_data(RAW_DATA_PATH)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"Data Preprocess Successful: {OUTPUT_PATH}")