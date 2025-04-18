import os
import subprocess

def run(script):
    print(f"🚀 Running: {script}")
    result = subprocess.run(["python", script])
    if result.returncode != 0:
        print(f"❌ Error in {script}. Exiting.")
        exit(1)

# === Step 1: Scrape and update latest game scores ===
run("get_scores.py")

# === Step 2: Scrape and update odds (including tomorrow) and merge ===
run("odds_scraper_with_fallback.py")

# === Step 3: Prepare training data for modeling ===
run("prepare_training_data.py")

# === Step 4: Train Models ===
run("train_ats_model.py")
run("train_total_model.py")
run("train_model_home_score.py")
run("train_model_away_score.py")

# === Step 5: Run predictions and generate outputs ===
run("make_predictions.py")

print("\n✅ Pipeline complete. You can now run the dashboard:")
print("👉  streamlit run mlb_predictions_dashboard.py")

