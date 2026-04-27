from flask import Flask, render_template, request
import pandas as pd
import joblib
from explain import generate_explanation
from feature_engineering import engineer_features

app = Flask(__name__)

# Load predictions on startup
def load_predictions():
    df = pd.read_csv("data/predictions.csv")
    return df

@app.route("/")
def index():
    df = load_predictions()

    # Get filter from URL if present
    risk_filter = request.args.get("risk", "All")

    if risk_filter != "All":
        filtered = df[df["predicted_risk"] == risk_filter]
    else:
        filtered = df

    # Sort by risk level
    risk_order = {"High": 0, "Medium": 1, "Low": 2}
    filtered = filtered.copy()
    filtered["risk_order"] = filtered["predicted_risk"].map(risk_order)
    filtered = filtered.sort_values("risk_order")

    # Summary stats
    stats = {
        "total": len(df),
        "high": len(df[df["predicted_risk"] == "High"]),
        "medium": len(df[df["predicted_risk"] == "Medium"]),
        "low": len(df[df["predicted_risk"] == "Low"]),
    }

    students = filtered[[
        "student_id", "predicted_risk", "explanation",
        "engagement_trend_score", "inactivity_streak",
        "missed_submissions", "language_status", "financial_support"
    ]].to_dict(orient="records")

    return render_template("index.html", students=students, stats=stats, risk_filter=risk_filter)


@app.route("/student/<student_id>")
def student_profile(student_id):
    df = load_predictions()
    student = df[df["student_id"] == student_id].iloc[0].to_dict()
    return render_template("profile.html", student=student)


if __name__ == "__main__":
    app.run(debug=True)