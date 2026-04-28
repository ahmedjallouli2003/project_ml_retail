import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def load_data(path="data/raw/retail.csv"):
    df = pd.read_csv(path)
    return df
def split_target(df, target="Churn"):
    y = df[target]
    X = df.drop(columns=[target])
    return X, y
def parse_dates(df):
    df["RegistrationDate"] = pd.to_datetime(
        df["RegistrationDate"], dayfirst=True, errors="coerce"
    )
    df["RegYear"] = df["RegistrationDate"].dt.year
    df["RegMonth"] = df["RegistrationDate"].dt.month
    df["RegDay"] = df["RegistrationDate"].dt.day
    df["RegWeekday"] = df["RegistrationDate"].dt.weekday
    df.drop(columns=["RegistrationDate"], inplace=True)
    return df
def create_features(df):
    df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)
    df["AvgBasketValue"] = df["MonetaryTotal"] / (df["Frequency"] + 1)
    df["TenureRatio"] = df["Recency"] / (df["CustomerTenure"] + 1)
    return df
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = np.clip(df[column], lower, upper)
    return df

def remove_useless_features(df):
    # ID & constantes
    df.drop(columns=["CustomerID", "Newsletter", "LastLoginIP"], inplace=True, errors='ignore')

    # Variance nulle
    selector = VarianceThreshold(threshold=0)
    selector.fit(df.select_dtypes(include=["int64","float64"]))
    cols = df.select_dtypes(include=["int64","float64"]).columns
    kept = cols[selector.get_support()]
    df = df[kept.tolist() + df.select_dtypes(include=["object"]).columns.tolist()]

    # Colonnes trop de NaN
    threshold = 0.5
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Corrélations >0.8
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    return df