import os
import ipaddress
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib


RAW_DATA_PATH = "data/raw/retail_customers_COMPLETE_CATEGORICAL.csv"
PROCESSED_DATA_PATH = "data/processed/retail_customers_cleaned.csv"
TRAIN_TEST_DIR = "data/train_test"
MODEL_DIR = "models"


def load_data(path=RAW_DATA_PATH):
    """
    Load raw dataset.
    """
    df = pd.read_csv(path)
    return df


def clean_outliers(df):
    """
    Replace abnormal values with NaN.
    """
    df = df.copy()

    # SupportTicketsCount should not be -1 or 999
    if "SupportTicketsCount" in df.columns:
        df["SupportTicketsCount"] = df["SupportTicketsCount"].replace([-1, 999], np.nan)

    # SatisfactionScore should normally be between 1 and 5
    if "SatisfactionScore" in df.columns:
        df["SatisfactionScore"] = df["SatisfactionScore"].replace([-1, 0, 99], np.nan)

    return df


def parse_registration_date(df):
    """
    Parse RegistrationDate and extract useful date features.
    """
    df = df.copy()

    if "RegistrationDate" in df.columns:
        df["RegistrationDate"] = pd.to_datetime(
            df["RegistrationDate"],
            dayfirst=True,
            errors="coerce"
        )

        df["RegYear"] = df["RegistrationDate"].dt.year
        df["RegMonth"] = df["RegistrationDate"].dt.month
        df["RegDay"] = df["RegistrationDate"].dt.day
        df["RegWeekday"] = df["RegistrationDate"].dt.weekday

        df = df.drop(columns=["RegistrationDate"])

    return df


def is_private_ip(ip):
    """
    Check if an IP address is private.
    """
    try:
        return int(ipaddress.ip_address(ip).is_private)
    except ValueError:
        return np.nan


def get_first_octet(ip):
    """
    Extract first octet from IP address.
    Example: 192.168.1.1 -> 192
    """
    try:
        return int(str(ip).split(".")[0])
    except Exception:
        return np.nan


def transform_ip(df):
    """
    Extract features from LastLoginIP.
    """
    df = df.copy()

    if "LastLoginIP" in df.columns:
        df["IP_FirstOctet"] = df["LastLoginIP"].apply(get_first_octet)
        df["IP_IsPrivate"] = df["LastLoginIP"].apply(is_private_ip)

        df = df.drop(columns=["LastLoginIP"])

    return df


def drop_unnecessary_columns(df):
    """
    Drop useless or risky columns.
    """
    df = df.copy()

    columns_to_drop = [
        "CustomerID",
        "NewsletterSubscribed",
        "ChurnRiskCategory"
    ]

    existing_columns_to_drop = [
        col for col in columns_to_drop if col in df.columns
    ]

    df = df.drop(columns=existing_columns_to_drop)

    return df


def feature_engineering(df):
    """
    Create additional useful features.
    """
    df = df.copy()

    if "MonetaryTotal" in df.columns and "Recency" in df.columns:
        df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)

    if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
        df["AvgBasketValue"] = df["MonetaryTotal"] / (df["Frequency"] + 1)

    if "Recency" in df.columns and "CustomerTenureDays" in df.columns:
        df["TenureRatio"] = df["Recency"] / (df["CustomerTenureDays"] + 1)

    return df


def clean_data(df):
    """
    Full cleaning function before ML preprocessing.
    """
    df = df.copy()

    df = clean_outliers(df)
    df = parse_registration_date(df)
    df = transform_ip(df)
    df = feature_engineering(df)
    df = drop_unnecessary_columns(df)

    return df


def split_features_target(df, target_col="Churn"):
    """
    Split data into features X and target y.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def build_preprocessor(X):
    """
    Build preprocessing pipeline:
    - numerical: imputation + standardization
    - categorical: imputation + one-hot encoding
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, numeric_features, categorical_features


def save_train_test_data(X_train_processed, X_test_processed, y_train, y_test):
    """
    Save processed train/test datasets.
    """
    os.makedirs(TRAIN_TEST_DIR, exist_ok=True)

    pd.DataFrame(X_train_processed).to_csv(
        os.path.join(TRAIN_TEST_DIR, "X_train_processed.csv"),
        index=False
    )

    pd.DataFrame(X_test_processed).to_csv(
        os.path.join(TRAIN_TEST_DIR, "X_test_processed.csv"),
        index=False
    )

    y_train.to_csv(
        os.path.join(TRAIN_TEST_DIR, "y_train.csv"),
        index=False
    )

    y_test.to_csv(
        os.path.join(TRAIN_TEST_DIR, "y_test.csv"),
        index=False
    )


def main():
    """
    Main preprocessing workflow.
    """
    print("Loading raw data...")
    df = load_data()

    print("Raw data shape:", df.shape)

    print("Cleaning data...")
    df_cleaned = clean_data(df)

    os.makedirs("data/processed", exist_ok=True)
    df_cleaned.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Cleaned data saved to:", PROCESSED_DATA_PATH)
    print("Cleaned data shape:", df_cleaned.shape)

    print("Splitting features and target...")
    X, y = split_features_target(df_cleaned, target_col="Churn")

    print("Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Building preprocessing pipeline...")
    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)

    print("Numerical features:", len(numeric_features))
    print("Categorical features:", len(categorical_features))

    print("Fitting preprocessing pipeline on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)

    print("Transforming test data...")
    X_test_processed = preprocessor.transform(X_test)

    print("Saving processed train/test data...")
    save_train_test_data(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)

    print("Preprocessor saved to:", preprocessor_path)

    print("Preprocessing completed successfully.")
    print("X_train_processed shape:", X_train_processed.shape)
    print("X_test_processed shape:", X_test_processed.shape)


if __name__ == "__main__":
    main()