from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
CLEANED_DATA_PATH = DATA_DIR / "processed" / "creditcard_cleaned.csv"


GDRIVE_FILE_ID = "19v_wQ_zvMFigjdoOUbljVpcd6DWyKxKo"

OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
COLS_TO_SCALE = ["Log_Amount"]