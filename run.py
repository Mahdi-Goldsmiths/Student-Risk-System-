"""
run.py - One-click setup and launch script
Generates data, engineers features, trains model and launches the dashboard.
"""

import subprocess
import sys
import os

def run_step(script, description):
    print(f"\n{'='*50}")
    print(f"  {description}")
    print(f"{'='*50}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f" Error running {script}. Please check the output above.")
        sys.exit(1)
    print("complete.")

if __name__ == "__main__":
    print(" Student Dropout Risk System — Setup & Launch")

    # Step 1 - Generate synthetic data
    run_step("generate_data.py", "Step 1/4: Generating synthetic student data")

    # Step 2 - Feature engineering
    run_step("feature_engineering.py", "Step 2/4: Running feature engineering")

    # Step 3 - Train model and generate predictions
    run_step("train_model.py", "Step 3/4: Training ML model")

    # Step 4 - Generate explanations
    run_step("explain.py", "Step 4/4: Generating risk explanations")

    # Launch dashboard
    print(f"\n{'='*50}")
    print("  Launching Dashboard")
    print(f"{'='*50}")
    print(" All steps complete. Launching dashboard...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server.")

    os.system(f"{sys.executable} app.py")