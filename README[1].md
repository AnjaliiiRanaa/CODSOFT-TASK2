# 💳 Credit Card Fraud Detection

A machine learning model that classifies a credit card transaction as **fraudulent** or **legitimate**, comparing **Logistic Regression**, **Decision Tree**, and **Random Forest**.

## 📌 Problem Statement

Build a model to detect fraudulent credit card transactions. Use a dataset containing information about credit card transactions, and experiment with algorithms like Logistic Regression, Decision Trees, or Random Forests to classify transactions as fraudulent or legitimate.

## 📂 Dataset

Based on the **Credit Card Fraud Detection** dataset (Kaggle: `mlg-ulb/creditcardfraud`) — real, anonymized transactions made by European cardholders in September 2013.

- `Time` — seconds elapsed since the first transaction in the dataset
- `V1`–`V28` — anonymized features (already PCA-transformed by the original dataset creators for confidentiality)
- `Amount` — transaction amount
- `Class` — **target**: `0` = legitimate, `1` = fraud

**Note on `creditcard.csv` in this repo:** the full Kaggle dataset is 284,807 rows (~98 MB), too large for a normal GitHub upload. This repo includes a **50,492-row sample** — all 492 real fraud cases plus 50,000 randomly sampled legitimate transactions — so it's realistic, still highly imbalanced (~1% fraud), and fits GitHub's upload limit. If you want to train on the full dataset, download it directly from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and replace `creditcard.csv`.

## ⚙️ How It Works

1. **Load data** and check the fraud/legit split (fraud is <1% — this is the core challenge of this task).
2. **Scale** `Amount` and `Time` with `StandardScaler` (V1–V28 are already scaled via PCA).
3. **Train/test split** with `stratify=y` so both sets keep the same fraud ratio.
4. **Train 3 models** — Logistic Regression, Decision Tree, Random Forest — all with `class_weight="balanced"` to compensate for the imbalance.
5. **Evaluate with ROC-AUC, precision, recall, and F1** — not plain accuracy (see note below).
6. **Save** the best model + scalers as `.pkl` files.

## 🚀 How to Run

```bash
pip install -r requirements.txt

# Step 1: train the models (also evaluates and saves the best one)
python fraud_detection.py

# Step 2: predict on a new/sample transaction
python predict.py
```

## 📊 Results

| Model | ROC-AUC | Fraud Recall | Fraud Precision |
|---|---|---|---|
| Logistic Regression | 0.985 | 0.94 | 0.30 |
| Decision Tree | 0.929 | 0.87 | 0.36 |
| **Random Forest (best)** | **0.987** | **0.86** | **0.97** |

**Why ROC-AUC/precision/recall instead of plain accuracy:** fraud is less than 1% of transactions, so a model that predicts "legit" every single time would already score ~99% accuracy while catching zero fraud — accuracy alone is meaningless here. Random Forest was picked because it catches 86% of frauds while only flagging genuine transactions as fraud 3% of the time (precision 0.97) — the practical trade-off a real fraud system needs: high recall (catch fraud) without drowning the bank in false alarms.

## 🔮 Future Improvements

- Train on the full 284,807-row dataset for more signal
- Try SMOTE (oversampling) or XGBoost/LightGBM for better recall–precision balance
- Add a real-time prediction API (Flask/FastAPI)
- Track model drift over time (fraud patterns change)

## 🛠️ Tech Stack

- Python
- pandas
- scikit-learn (Logistic Regression, Decision Tree, Random Forest)
- joblib (saving/loading the model)

## 📁 Project Structure

```
├── creditcard.csv          # sampled dataset (see note above)
├── fraud_detection.py      # training + evaluation
├── predict.py               # predict on a new transaction
├── requirements.txt
└── README.md
```

---
*Built as part of a machine learning internship project (CodSoft — Task 2: Credit Card Fraud Detection).*
