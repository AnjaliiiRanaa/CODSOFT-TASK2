"""
Credit Card Fraud Detection
-----------------------------
Classifies a credit card transaction as fraudulent (1) or legitimate (0),
using Logistic Regression, Decision Tree, and Random Forest — then picks
the best model based on ROC-AUC (NOT plain accuracy — see README for why).

Dataset: creditcard.csv (same one used for the CodSoft ML internship
"Credit Card Fraud Detection" task, originally from Kaggle mlg-ulb/creditcardfraud).

Columns:
    Time    -> seconds since the first transaction in the dataset
    V1-V28  -> anonymized features (already PCA-transformed by the dataset creator)
    Amount  -> transaction amount
    Class   -> 0 = legitimate, 1 = fraud   <-- this is what we predict

Run:
    python fraud_detection.py
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def main():
    print("Loading data...")
    df = pd.read_csv("creditcard.csv")
    print(f"Total transactions: {len(df)}")
    print(f"Fraud transactions: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

    # ---------------------------------------------------------
    # 1. Scale 'Amount' and 'Time' (V1-V28 are already scaled via PCA)
    #    Two separate scalers because they're different columns
    # ---------------------------------------------------------
    amount_scaler = StandardScaler()
    time_scaler = StandardScaler()
    df["Amount_scaled"] = amount_scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"] = time_scaler.fit_transform(df[["Time"]])
    df = df.drop(columns=["Amount", "Time"])

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # ---------------------------------------------------------
    # 2. Train/test split (stratify keeps the same fraud ratio in both sets)
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # ---------------------------------------------------------
    # 3. Train 3 models — all with class_weight='balanced' because
    #    fraud is <1% of the data, and a model that always predicts
    #    "legit" would already score ~99% plain accuracy (but be useless)
    # ---------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)
        results[name] = (model, auc)

        print(f"ROC-AUC: {auc:.4f}")
        print("Confusion matrix (rows=actual, cols=predicted) [legit, fraud]:")
        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds, target_names=["Legit", "Fraud"], zero_division=0))

    # ---------------------------------------------------------
    # 4. Pick the best model by ROC-AUC (best at ranking fraud vs legit)
    # ---------------------------------------------------------
    best_name = max(results, key=lambda n: results[n][1])
    best_model = results[best_name][0]
    print(f"\nBest model: {best_name} (ROC-AUC = {results[best_name][1]:.4f})")

    joblib.dump(best_model, "fraud_model.pkl")
    joblib.dump(amount_scaler, "amount_scaler.pkl")
    joblib.dump(time_scaler, "time_scaler.pkl")
    print("Saved model as 'fraud_model.pkl' and scalers as 'amount_scaler.pkl' / 'time_scaler.pkl'")


if __name__ == "__main__":
    main()
