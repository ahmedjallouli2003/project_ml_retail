import os
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


TRAIN_TEST_DIR = "data/train_test"
PROCESSED_DATA_PATH = "data/processed/retail_customers_cleaned.csv"
MODEL_DIR = "models"
REPORTS_DIR = "reports"


def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# 1. CLUSTERING
# ============================================================

def train_clustering():
    print("\n========== CLUSTERING ==========")

    X_train = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "X_train_processed.csv"))
    y_train = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "y_train.csv")).values.ravel()

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_train)
    silhouette = silhouette_score(X_train, clusters)

    cluster_df = pd.DataFrame({
        "Cluster": clusters,
        "Churn": y_train
    })

    cluster_summary = cluster_df.groupby("Cluster").agg(
        Number_of_Customers=("Cluster", "count"),
        Churn_Rate=("Churn", "mean")
    )

    cluster_summary["Churn_Rate"] *= 100

    cluster_df.to_csv(
        os.path.join(REPORTS_DIR, "customer_clusters.csv"),
        index=False
    )

    cluster_summary.to_csv(
        os.path.join(REPORTS_DIR, "cluster_summary.csv")
    )

    joblib.dump(
        kmeans,
        os.path.join(MODEL_DIR, "kmeans_customer_segments.joblib")
    )

    print("Clustering terminé.")
    print("Silhouette Score:", silhouette)
    print(cluster_summary)


# ============================================================
# 2. CLASSIFICATION
# ============================================================

def load_classification_data(use_pca=False):
    if use_pca:
        X_train_path = os.path.join(TRAIN_TEST_DIR, "X_train_pca_80.csv")
        X_test_path = os.path.join(TRAIN_TEST_DIR, "X_test_pca_80.csv")
    else:
        X_train_path = os.path.join(TRAIN_TEST_DIR, "X_train_processed.csv")
        X_test_path = os.path.join(TRAIN_TEST_DIR, "X_test_processed.csv")

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    y_train = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "y_test.csv")).values.ravel()

    return X_train, X_test, y_train, y_test


def evaluate_classification_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n===== {model_name} =====")
    print(classification_report(y_test, y_pred))

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }


def train_classification():
    print("\n========== CLASSIFICATION ==========")

    experiments = [
        {
            "name": "Logistic Regression sans PCA",
            "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "use_pca": False
        },
        {
            "name": "Random Forest sans PCA",
            "model": RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            ),
            "use_pca": False
        },
        {
            "name": "Logistic Regression avec PCA",
            "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "use_pca": True
        },
        {
            "name": "Random Forest avec PCA",
            "model": RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            ),
            "use_pca": True
        }
    ]

    all_results = []

    best_model = None
    best_name = None
    best_f1 = -1
    best_use_pca = False

    for exp in experiments:
        X_train, X_test, y_train, y_test = load_classification_data(
            use_pca=exp["use_pca"]
        )

        model = exp["model"]
        model.fit(X_train, y_train)

        results = evaluate_classification_model(
            model,
            X_test,
            y_test,
            exp["name"]
        )

        all_results.append(results)

        if results["F1-score"] > best_f1:
            best_f1 = results["F1-score"]
            best_model = model
            best_name = exp["name"]
            best_use_pca = exp["use_pca"]

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        os.path.join(REPORTS_DIR, "classification_results.csv"),
        index=False
    )

    joblib.dump(
        best_model,
        os.path.join(MODEL_DIR, "best_churn_model.joblib")
    )

    metadata = {
        "best_model_name": best_name,
        "best_f1_score": best_f1,
        "use_pca": best_use_pca
    }

    joblib.dump(
        metadata,
        os.path.join(MODEL_DIR, "best_churn_model_metadata.joblib")
    )

    print("\nMeilleur modèle de classification :", best_name)
    print("Meilleur F1-score :", best_f1)
    print("Utilise PCA :", best_use_pca)


# ============================================================
# 3. REGRESSION
# ============================================================

def evaluate_regression_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def train_regression():
    print("\n========== REGRESSION ==========")

    df = pd.read_csv(PROCESSED_DATA_PATH)

    target = "MonetaryTotal"

    X = df.drop(columns=[target])
    y = df[target]

    X = X.drop(columns=["Churn"], errors="ignore")

    X = X.drop(
        columns=[
            "MonetaryAvg",
            "MonetaryStd",
            "MonetaryMin",
            "MonetaryMax",
            "MonetaryPerDay",
            "AvgBasketValue"
        ],
        errors="ignore"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

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

    models = [
        {
            "name": "Linear Regression",
            "model": LinearRegression()
        },
        {
            "name": "Random Forest Regressor",
            "model": RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        }
    ]

    all_results = []

    best_model = None
    best_name = None
    best_rmse = float("inf")

    for item in models:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", item["model"])
            ]
        )

        pipeline.fit(X_train, y_train)

        results = evaluate_regression_model(
            pipeline,
            X_test,
            y_test,
            item["name"]
        )

        all_results.append(results)

        if results["RMSE"] < best_rmse:
            best_rmse = results["RMSE"]
            best_model = pipeline
            best_name = item["name"]

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        os.path.join(REPORTS_DIR, "regression_results.csv"),
        index=False
    )

    joblib.dump(
        best_model,
        os.path.join(MODEL_DIR, "best_regression_model.joblib")
    )

    print("\nMeilleur modèle de régression :", best_name)
    print("Meilleur RMSE :", best_rmse)
    print(results_df)


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dirs()

    train_clustering()
    train_classification()
    train_regression()

    print("\nTous les modèles ont été entraînés avec succès.")


if __name__ == "__main__":
    main()