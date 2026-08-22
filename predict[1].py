"""
Use the trained model to predict whether a NEW transaction is fraud or legit.

Run this only AFTER fraud_detection.py has created
'fraud_model.pkl' and 'scaler.pkl'.

Usage:
    python predict.py
"""

import pandas as pd
import joblib

model = joblib.load("fraud_model.pkl")
amount_scaler = joblib.load("amount_scaler.pkl")
time_scaler = joblib.load("time_scaler.pkl")


def predict_transaction(row: dict):
    """
    row must contain: Time, V1...V28, Amount
    (exactly like one row of creditcard.csv, minus the 'Class' column)
    """
    df = pd.DataFrame([row])
    df["Amount_scaled"] = amount_scaler.transform(df[["Amount"]])
    df["Time_scaled"] = time_scaler.transform(df[["Time"]])
    df = df.drop(columns=["Amount", "Time"])

    # make sure column order matches what the model was trained on
    df = df[model.feature_names_in_]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]  # probability of fraud
    return prediction, probability


if __name__ == "__main__":
    # Example: grab a real row from the dataset to test with
    sample = pd.read_csv("creditcard.csv").drop(columns=["Class"]).iloc[0].to_dict()

    pred, prob = predict_transaction(sample)
    label = "FRAUD" if pred == 1 else "LEGIT"
    print(f"Prediction: {label}  (fraud probability: {prob:.2%})")
