import os
import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


TRAIN_TEST_DIR = "data/train_test"
MODEL_DIR = "models"


def load_processed_data():
    X_train = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "X_train_processed.csv"))
    X_test = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "X_test_processed.csv"))
    y_train = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(TRAIN_TEST_DIR, "y_test.csv"))

    return X_train, X_test, y_train, y_test


def apply_pca(X_train, X_test, variance_threshold=0.80):
    pca = PCA(n_components=variance_threshold)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return pca, X_train_pca, X_test_pca


def save_pca_outputs(pca, X_train_pca, X_test_pca, variance_threshold=0.80):
    os.makedirs(TRAIN_TEST_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    threshold_label = str(int(variance_threshold * 100))

    pd.DataFrame(X_train_pca).to_csv(
        os.path.join(TRAIN_TEST_DIR, f"X_train_pca_{threshold_label}.csv"),
        index=False
    )

    pd.DataFrame(X_test_pca).to_csv(
        os.path.join(TRAIN_TEST_DIR, f"X_test_pca_{threshold_label}.csv"),
        index=False
    )

    joblib.dump(
        pca,
        os.path.join(MODEL_DIR, f"pca_{threshold_label}.joblib")
    )


def main():
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data()

    print("Original X_train shape:", X_train.shape)
    print("Original X_test shape:", X_test.shape)

    variance_threshold = 0.80

    print(f"Applying PCA with {int(variance_threshold * 100)}% variance threshold...")
    pca, X_train_pca, X_test_pca = apply_pca(
        X_train,
        X_test,
        variance_threshold=variance_threshold
    )

    print("PCA X_train shape:", X_train_pca.shape)
    print("PCA X_test shape:", X_test_pca.shape)
    print("Explained variance retained:", pca.explained_variance_ratio_.sum())

    save_pca_outputs(
        pca,
        X_train_pca,
        X_test_pca,
        variance_threshold=variance_threshold
    )

    print("PCA outputs saved successfully.")


if __name__ == "__main__":
    main()