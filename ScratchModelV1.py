import os
import subprocess
from datetime import datetime

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

# === Step 6: GitHub Commit + Push ===
print("\n📦 Committing and pushing to GitHub...")
try:
    subprocess.run(["git", "add", "."], check=True)
    commit_msg = f"🔄 Auto-update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    subprocess.run(["git", "commit", "-m", commit_msg], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print("✅ GitHub push complete.")
except subprocess.CalledProcessError as e:
    print(f"❌ Git push failed: {e}")

print("\n✅ Pipeline complete. You can now run the dashboard:")
print("👉  streamlit run mlb_predictions_dashboard.py")
