
## Hướng dẫn cài đặt & chạy code

### Môi trường

- **Python 3.13**
- Thư viện chính: `scikit-learn`, `xgboost`, `imbalanced-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `gdown` (danh sách đầy đủ trong `requirements.txt`).
- Cần Internet ở lần chạy đầu tiên để tải dữ liệu từ Google Drive.

**Tạo môi trường ảo (chọn một):**

```bash
# Cách 1: venv
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Cách 2: Conda
conda create -n btl-ml python=3.13
conda activate btl-ml
```

**Cài thư viện:**

```bash
pip install -r requirements.txt
```

### Dữ liệu

Ở lần chạy đầu, code tự tải `creditcard.csv` vào `data/raw/`.

Dữ liệu nằm tại thư mục
```
data/raw/creditcard.csv
```

### Chạy

> **Luôn chạy từ thư mục gốc của project** và dùng cờ `-m` để import `src` đúng.

```bash
# (Tuỳ chọn) Tiền xử lý và lưu dữ liệu đã làm sạch vào data/processed/creditcard_cleaned.csv
python -m src.preprocess

# Chạy từng mô hình (mỗi lệnh tự tiền xử lý, CV 5-fold, đánh giá trên test, lưu hình)
python -m src.model_logistic     # Logistic Regression
python -m src.model_svm          # SVM (RBF)
python -m src.model_ANN          # ANN (MLP)
python -m src.model_ensemble     # XGBoost
```

Mỗi lệnh in ra kết quả từng fold (`Fold k -> {...}`), kết quả test (`TEST RESULT (...)`) và lưu hình vào `outputs/figures/`.

## Cấu trúc thư mục

```
Final/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py            # Đường dẫn, seed, tham số chia dữ liệu, Google Drive ID
│   ├── preprocess.py        # Tải dữ liệu, làm sạch, tạo đặc trưng, chia train/test, CV splitter
│   ├── metrics.py           # Hàm tính metric và vẽ PR curve + confusion matrix
│   ├── model_logistic.py    # Logistic Regression
│   ├── model_svm.py         # SVM RBF + under-sampling + Platt scaling
│   ├── model_ANN.py         # MLP (baseline deep learning)
│   └── model_ensemble.py    # XGBoost (ensemble)
├── data/                    # raw/ và processed/
└── outputs/                 # figures/
```

**Notebook**:

| Notebook | Nội dung |
|---|---|
| `notebook/EDA.ipynb` | Phân tích khám phá dữ liệu. Cần có `data/raw/creditcard.csv` (chạy một lần `python -m src.preprocess` để tự tải). |
| `notebook/test_metric.ipynb` | Chạy cả 4 mô hình, tạo bảng so sánh `outputs/model_comparison.csv` và các hình `compare_*.png` (mục 4). |
