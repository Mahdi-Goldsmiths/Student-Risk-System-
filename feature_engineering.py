import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Takes raw student data and calculates engineered features used for risk prediction.
    """

    # --- Engagement Trend Score (ETS) ---
    attendance_norm = df["attendance_rate"]  # already 0-1
    vle_norm = df["vle_activity_score"] / 100
    assessment_norm = df["avg_assessment_score"] / 100

    # Weighted average (attendance weighted highest as most reliable signal)
    df["engagement_trend_score"] = (
        (attendance_norm * 0.4) +
        (vle_norm * 0.35) +
        (assessment_norm * 0.25)
    ).round(3)

    # --- Submission Timeliness Score ---
    df["submission_timeliness_score"] = (
        1 - (df["missed_submissions"] / 7)
    ).clip(0, 1).round(3)

    # --- Inactivity Streak Normalised ---
    df["inactivity_norm"] = (df["inactivity_streak"] / 20).clip(0, 1).round(3)

    # --- Contextual Risk Flags ---
    language_risk = {"native": 0.0, "non-native": 0.3, "international": 0.5}
    df["language_risk_weight"] = df["language_status"].map(language_risk)

    
    financial_risk = {"scholarship": 0.0, "loan": 0.2, "self-funded": 0.4}
    df["financial_risk_weight"] = df["financial_support"].map(financial_risk)

    return df


if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv("data/students.csv")

    # Apply feature engineering
    df = engineer_features(df)

    # Save enriched dataset
    df.to_csv("data/students_features.csv", index=False)
    print("Feature engineering complete. Saved to data/students_features.csv")
    print(df[[
        "student_id",
        "engagement_trend_score",
        "submission_timeliness_score",
        "inactivity_norm",
        "language_risk_weight",
        "financial_risk_weight"
    ]].head())