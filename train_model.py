import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def generate_risk_label(row):
    """
    Rule-based risk label generation from engineered features.
    """
    score = 0

    # Low engagement trend score is a strong risk signal
    if row["engagement_trend_score"] < 0.4:
        score += 3
    elif row["engagement_trend_score"] < 0.6:
        score += 1

    # High inactivity is a risk signal
    if row["inactivity_norm"] > 0.6:
        score += 2
    elif row["inactivity_norm"] > 0.3:
        score += 1

    # Poor submission timeliness
    if row["submission_timeliness_score"] < 0.4:
        score += 2
    elif row["submission_timeliness_score"] < 0.7:
        score += 1

    # Contextual risk factors
    score += row["language_risk_weight"]
    score += row["financial_risk_weight"]

    # Convert score to risk level
    if score >= 5:
        return "High"
    elif score >= 3:
        return "Medium"
    else:
        return "Low"


def train_model():
    # Load feature-engineered data
    df = pd.read_csv("data/students_features.csv")

    # Generate labels
    df["risk_level"] = df.apply(generate_risk_label, axis=1)

    # Save labelled dataset
    df.to_csv("data/students_labelled.csv", index=False)
    print("Risk label distribution:")
    print(df["risk_level"].value_counts())
    print()

    # --- Define features and target ---
    FEATURES = [
        "engagement_trend_score",
        "submission_timeliness_score",
        "inactivity_norm",
        "language_risk_weight",
        "financial_risk_weight"
    ]

    X = df[FEATURES]
    y = df["risk_level"]

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Train Random Forest ---
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))

    # --- Save model ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/risk_model.pkl")
    joblib.dump(FEATURES, "models/features.pkl")
    print("Model saved to models/risk_model.pkl")


if __name__ == "__main__":
    train_model()