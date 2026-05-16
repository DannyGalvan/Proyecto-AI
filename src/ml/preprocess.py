import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.settings import DATASET_PATH, PROCESSED_DIR, ML_SAMPLE_SIZE

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def load_raw(sample_size: int | None = None, random_state: int = 42) -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}. "
            "Place the CSV at data/financial_data.csv"
        )

    print(f"Loading data from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"Full dataset shape: {df.shape}")
    if sample_size is not None:
        # Stratified sample to preserve fraud ratio
        fraud = df[df["isFraud"] == 1]
        normal = df[df["isFraud"] == 0].sample(
            n=sample_size - len(fraud), random_state=random_state
        )
        df = pd.concat([fraud, normal]).sample(frac=1, random_state=random_state)
        print(f"Sampled dataset shape: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Error balance: difference between expected and actual balance after transaction.
    # A fraud signature is draining an account to 0 while the balance math doesn't add up.
    df["errorBalanceOrig"] = (
        df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    )
    df["errorBalanceDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )

    # Flag: origin account emptied to zero after transaction
    df["origBalanceZero"] = (df["newbalanceOrig"] == 0).astype(int)

    # Ratio of amount relative to origin balance (inf when balance was 0)
    df["amountToOrigRatio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # Only TRANSFER and CASH_OUT transactions are flagged as fraud in this dataset
    df["isHighRiskType"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)

    return df


def encode_type(df: pd.DataFrame) -> pd.DataFrame:
    # One-hot encode transaction type; drop_first avoids perfect multicollinearity
    dummies = pd.get_dummies(df["type"], prefix="type", drop_first=False)
    # Ensure all expected columns exist even if a type is missing in a sample
    for t in TRANSACTION_TYPES:
        col = f"type_{t}"
        if col not in dummies.columns:
            dummies[col] = 0
    return pd.concat([df.drop(columns=["type"]), dummies], axis=1)


def preprocess(
    sample_size: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    save: bool = True,
) -> tuple:
    df = load_raw(sample_size=sample_size, random_state=random_state)

    print("\n--- Class distribution (raw) ---")
    fraud_count = df["isFraud"].sum()
    print(f"Fraud:  {fraud_count:,}  ({fraud_count/len(df)*100:.4f}%)")
    print(f"Normal: {len(df)-fraud_count:,}  ({(len(df)-fraud_count)/len(df)*100:.4f}%)")

    # Drop columns that are identifiers or derived targets (data leakage risk)
    df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])

    df = engineer_features(df)
    df = encode_type(df)

    # Handle missing values (dataset is clean but guard for robustness)
    df = df.fillna(0)

    feature_cols = [c for c in df.columns if c != "isFraud"]
    X = df[feature_cols]
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ---- Outlier handling (winsorization at P99.5) ----
    # `amount` and balance columns are extremely right-skewed (up to ~92M).
    # We clip extreme values at the 99.5th percentile computed ONLY on train
    # to prevent outliers from distorting StandardScaler statistics and LR.
    # Rationale: removing outliers would drop real fraud (frauds tend to be large);
    # winsorization preserves the "this is a large transaction" signal without
    # letting a single extreme value move the feature distribution.
    outlier_cols = [
        "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
    ]
    clip_upper = X_train[outlier_cols].quantile(0.995)
    print("\n--- Outlier clipping (P99.5 thresholds, computed on train) ---")
    for col, upper in clip_upper.items():
        print(f"  {col:<18}: clipped to <= {upper:,.0f}")
    X_train[outlier_cols] = X_train[outlier_cols].clip(upper=clip_upper, axis=1)
    X_test[outlier_cols]  = X_test[outlier_cols].clip(upper=clip_upper, axis=1)

    # Scale numeric features; fit only on training data to prevent leakage
    numeric_cols = [
        "step", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "errorBalanceOrig", "errorBalanceDest", "amountToOrigRatio",
    ]
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print(f"\nTrain size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")
    print(f"Features: {X_train.shape[1]}")

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        X_train.to_parquet(PROCESSED_DIR / "X_train.parquet", index=False)
        X_test.to_parquet(PROCESSED_DIR / "X_test.parquet", index=False)
        y_train.to_frame().to_parquet(PROCESSED_DIR / "y_train.parquet", index=False)
        y_test.to_frame().to_parquet(PROCESSED_DIR / "y_test.parquet", index=False)
        joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")
        print(f"\nProcessed data saved to {PROCESSED_DIR}")

    return X_train, X_test, y_train, y_test, scaler


def load_processed() -> tuple:
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").squeeze()
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").squeeze()
    scaler = joblib.load(PROCESSED_DIR / "scaler.pkl")
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    preprocess(sample_size=ML_SAMPLE_SIZE)
