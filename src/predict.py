import joblib
import pandas as pd
import numpy as np

# ===============================
# CHARGER LES MODELES
# ===============================

classification_model = joblib.load("models/best_churn_model.joblib")
classification_metadata = joblib.load("models/best_churn_model_metadata.joblib")

regression_model = joblib.load("models/best_regression_model.joblib")
kmeans_model = joblib.load("models/kmeans_customer_segments.joblib")

preprocessor = joblib.load("models/preprocessor.joblib")

# Si le modèle utilise PCA
use_pca = classification_metadata["use_pca"]

if use_pca:
    pca = joblib.load("models/pca_80.joblib")
else:
    pca = None


# ===============================
# FONCTION DE PREDICTION
# ===============================

def predict_customer(customer_dict):
    """
    customer_dict : dictionnaire représentant un client
    """

    # Convertir en DataFrame
    df = pd.DataFrame([customer_dict])

    # ===============================
    # PREPROCESSING
    # ===============================

    X_processed = preprocessor.transform(df)

    if pca is not None:
        X_processed = pca.transform(X_processed)

    # ===============================
    # CLASSIFICATION (CHURN)
    # ===============================

    churn_pred = classification_model.predict(X_processed)[0]
    churn_proba = classification_model.predict_proba(X_processed)[0][1]

    # ===============================
    # CLUSTERING
    # ===============================

    cluster = kmeans_model.predict(X_processed)[0]

    # ===============================
    # REGRESSION
    # ===============================

    monetary_pred = regression_model.predict(df)[0]

    # ===============================
    # RESULTAT
    # ===============================

    result = {
        "Churn Prediction": int(churn_pred),
        "Churn Probability": float(churn_proba),
        "Cluster": int(cluster),
        "Predicted MonetaryTotal": float(monetary_pred)
    }

    return result


# ===============================
# TEST LOCAL
# ===============================

if __name__ == "__main__":

    # Exemple client (adapter selon tes colonnes)
    sample_customer = {
        "Age": 35,
        "Gender": "Male",
        "AnnualIncome": 50000,
        "TotalPurchases": 10,
        "Recency": 20,
        "Frequency": 5,
        "CustomerTenureDays": 300,
        "LastLoginIP": "192.168.1.1",
        "SupportTicketsCount": 1,
        "SatisfactionScore": 4,
        "NewsletterSubscribed": "No",
        "CustomerType": "Standard",
        "RFMSegment": "Regular",
        "AccountStatus": "Active"
    }

    prediction = predict_customer(sample_customer)

    print("\n=== PREDICTION RESULT ===")
    for key, value in prediction.items():
        print(f"{key}: {value}")