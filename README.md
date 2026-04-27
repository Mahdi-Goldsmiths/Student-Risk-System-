# Student Dropout Risk Prediction System

A web-based decision support tool that helps academic tutors identify students 
at risk of disengagement or withdrawal using machine learning and learning analytics.

---

## What It Does

The system analyses synthetic student engagement and contextual data, applies 
feature engineering, trains a machine learning model, and displays predicted 
risk levels in an interactive dashboard. Tutors can view individual student 
profiles with plain-English explanations of why a student was flagged and 
suggested actions to take.

---

## Core Features

- Generates a synthetic dataset of 200 students with behavioural and contextual attributes
- Applies feature engineering including:
  - Engagement Trend Score (ETS)
  - Submission Timeliness Score
  - Inactivity Normalisation
  - Language and Financial Risk Weights
- Trains a Random Forest classifier to predict risk level (High / Medium / Low)
- Displays results in a web dashboard sorted by risk level
- Individual student profile pages showing:
  - Predicted risk level
  - Engagement metrics and behavioural signals
  - Contextual information (language status, financial support, year of study)
  - Plain-English explanation of why the student was flagged
  - Suggested tutor actions based on risk level
- Risk level filtering on the main dashboard

## Setup Instructions

### 1. Install Python
Make sure Python 3.10 or above is installed.  
Download from: https://www.python.org/downloads/  
**Important:** Tick "Add Python to PATH" during installation.

### 2. Install Dependencies
Open a terminal in the project folder and run:

### 3. Run the System
In the terminal, run: py run.py

This will automatically:
1. Generate the synthetic student dataset
2. Run feature engineering
3. Train the machine learning model
4. Generate risk predictions and explanations
5. Launch the web dashboard

### 4. Open the Dashboard
Once the server is running, open your browser and go to:
http://127.0.0.1:5000/

Press `Ctrl + C` in the terminal to stop the server.

---

## Using the Dashboard

- The **main page** shows all 200 students sorted by risk level (High → Medium → Low)
- Use the **filter buttons** to view only High, Medium or Low risk students
- Click **View Profile** on any student to see their full profile including:
  - Risk level and explanation
  - Engagement and behavioural metrics
  - Contextual factors
  - Suggested tutor actions

---

## Technical Decisions

- **Synthetic data** was used instead of a real dataset to allow full control over 
  feature design and avoid data privacy concerns (GDPR compliance)
- **Random Forest** was chosen for its robustness, interpretability of feature 
  importance, and strong performance on small tabular datasets
- **Rule-based label generation** simulates ground truth labels in the absence of 
  real historical dropout data
- All predictions are **advisory only** — the system does not make automated 
  decisions about students

---

## Known Limitations

- Dataset is synthetic — results do not reflect real student populations
- High risk class is small (approx. 10 students) due to synthetic data distribution, 
  which affects model recall for that class
- No user authentication — in a real deployment, access would be restricted to 
  authorised staff only
- No persistent storage — predictions are regenerated each time `run.py` is run
- The system has been tested on Windows 11 with Python 3.14

---

## GDPR Note

All data used in this system is fully synthetic and anonymised. No real student 
data has been used at any stage. In a real deployment, the system would require 
strict access controls, data minimisation, and compliance with institutional 
data governance policies.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas | Data manipulation and CSV handling |
| numpy | Numerical operations |
| scikit-learn | Machine learning model |
| flask | Web framework for dashboard |
| joblib | Model serialisation |

---

