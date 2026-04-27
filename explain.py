import pandas as pd
import joblib

def generate_explanation(row):
    """
    Generates a plain-English explanation of why a student
    has been assigned their risk level.
    """
    reasons = []

    # Engagement
    if row["engagement_trend_score"] < 0.4:
        reasons.append("very low overall engagement (attendance, VLE activity and assessment scores are all poor)")
    elif row["engagement_trend_score"] < 0.6:
        reasons.append("below average engagement across attendance and online activity")

    # Inactivity
    if row["inactivity_norm"] > 0.6:
        reasons.append("a long period of inactivity detected (12+ days without recorded activity)")
    elif row["inactivity_norm"] > 0.3:
        reasons.append("some periods of inactivity detected")

    # Submissions
    if row["submission_timeliness_score"] < 0.4:
        reasons.append("a high number of missed or late submissions")
    elif row["submission_timeliness_score"] < 0.7:
        reasons.append("some missed submissions on record")

    # Contextual
    if row["language_risk_weight"] >= 0.5:
        reasons.append("international student status (may benefit from additional academic support)")
    elif row["language_risk_weight"] == 0.3:
        reasons.append("non-native English speaker (language support may be beneficial)")

    if row["financial_risk_weight"] >= 0.4:
        reasons.append("self-funded financial status (financial pressures may impact engagement)")
    elif row["financial_risk_weight"] == 0.2:
        reasons.append("student loan funded (financial pressures worth monitoring)")

    # Build explanation text
    if not reasons:
        explanation = "No significant risk factors identified. Student appears to be engaging well."
    else:
        explanation = "This student has been flagged due to: " + "; ".join(reasons) + "."

    return explanation


def generate_all_predictions():
    # Load labelled data and model
    df = pd.read_csv("data/students_labelled.csv")
    model = joblib.load("models/risk_model.pkl")
    features = joblib.load("models/features.pkl")

    # Generate model predictions
    df["predicted_risk"] = model.predict(df[features])

    # Generate explanations
    df["explanation"] = df.apply(generate_explanation, axis=1)

    # Save final predictions
    df.to_csv("data/predictions.csv", index=False)
    print("Predictions and explanations generated.")
    print()

    # Preview a few examples
    preview = df[["student_id", "predicted_risk", "explanation"]].head(5)
    for _, row in preview.iterrows():
        print(f"{row['student_id']} — {row['predicted_risk']}")
        print(f"  {row['explanation']}")
        print()


if __name__ == "__main__":
    generate_all_predictions()