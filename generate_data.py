import pandas as pd
import numpy as np
import os

# Set seed so data is the same every time we run it
np.random.seed(42)

NUM_STUDENTS = 200

def generate_synthetic_data():
    student_ids = [f"STU{str(i).zfill(4)}" for i in range(1, NUM_STUDENTS + 1)]

    # --- Contextual Features ---
    language_status = np.random.choice(
        ["native", "non-native", "international"],
        size=NUM_STUDENTS,
        p=[0.6, 0.25, 0.15]
    )

    financial_support = np.random.choice(
        ["loan", "scholarship", "self-funded"],
        size=NUM_STUDENTS,
        p=[0.5, 0.3, 0.2]
    )

    year_of_study = np.random.choice([1, 2, 3], size=NUM_STUDENTS)

    # --- Behavioural Features ---
    attendance_rate = np.clip(np.random.normal(0.75, 0.15, NUM_STUDENTS), 0, 1)
    vle_activity_score = np.clip(np.random.normal(60, 20, NUM_STUDENTS), 0, 100)
    avg_assessment_score = np.clip(np.random.normal(58, 15, NUM_STUDENTS), 0, 100)
    missed_submissions = np.random.randint(0, 8, NUM_STUDENTS)
    inactivity_streak = np.random.randint(0, 21, NUM_STUDENTS)

    # --- DataFrame ---
    df = pd.DataFrame({
        "student_id": student_ids,
        "year_of_study": year_of_study,
        "language_status": language_status,
        "financial_support": financial_support,
        "attendance_rate": attendance_rate.round(2),
        "vle_activity_score": vle_activity_score.round(1),
        "avg_assessment_score": avg_assessment_score.round(1),
        "missed_submissions": missed_submissions,
        "inactivity_streak": inactivity_streak
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/students.csv", index=False)
    print(f"Dataset generated: {NUM_STUDENTS} students saved to data/students.csv")
    print(df.head())

if __name__ == "__main__":
    generate_synthetic_data()