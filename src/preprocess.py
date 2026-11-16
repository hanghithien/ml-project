import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import gdown

from src.config import (
    GDRIVE_FILE_ID,
    RAW_DATA_PATH,
    CLEANED_DATA_PATH,
    TEST_SIZE,
    N_SPLITS,
    RANDOM_STATE,
)

def download_data(file_path, file_id=GDRIVE_FILE_ID):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(file_path), quiet=False)


def load_and_clean_data(file_path):
    download_data(file_path)
    df = pd.read_csv(file_path)
    df = df.drop_duplicates(keep='first').copy()
    df['Hour'] = (df['Time'] // 3600) % 24
    df['Log_Amount'] = np.log1p(df['Amount'])
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df = df.drop(columns=['Time', 'Hour', 'Amount'])

    return df

def split_train_test(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

def get_cv_splitter():
    return StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

if __name__ == "__main__":
    df_clean = load_and_clean_data(RAW_DATA_PATH)
    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
    df_clean.to_csv(CLEANED_DATA_PATH, index=False)
    