# ⚾ MLB Scratch Model V1 – Project Manifest

## 📂 Project Overview

This project forecasts MLB game outcomes using machine learning models trained on daily game data and betting odds. It scrapes real-time scores and odds, preprocesses features, trains models, makes predictions, and visualizes results through a Streamlit dashboard.

---

## 🧩 Pipeline Workflow Summary

Each step is run via the `ScratchModelV1.py` controller script:

### Step 1️⃣ – Scrape Game Scores  
**Script:** `get_scores.py`  
- Scrapes actual MLB game scores from ESPN for the previous day  
- Appends new results to `mlb_boxscores_cleaned.csv`  
- Ensures no duplicates via merge keys

### Step 2️⃣ – Scrape & Merge Odds  
**Script:** `odds_scraper_with_fallback.py`  
- Scrapes daily betting odds using The Odds API (including today's and tomorrow's games)  
- Cleans and deduplicates `mlb_odds_mybookie.csv`  
- Merges scores and odds into `mlb_model_and_odds.csv`  
- Merge keys: `Game Date`, `Home Team`, `Away Team`

### Step 3️⃣ – Prepare Model Features  
**Script:** `prepare_training_data.py`  
- Engineers features such as:
  - Spread Diff, ML Diff, Log Odds Diff  
  - Score-based targets for regression models  
- Saves training set as `mlb_training_ats.csv`

### Step 4️⃣ – Train Machine Learning Models  
**Scripts:**  
- `train_ats_model.py` (classification)  
- `train_total_model.py` (classification)  
- `train_model_home_score.py` (regression)  
- `train_model_away_score.py` (regression)  
- All models are saved as `.pkl` in the `/models` folder

### Step 5️⃣ – Make Predictions  
**Script:** `make_predictions.py`  
- Loads `mlb_model_and_odds.csv`  
- Applies models to generate predictions:
  - ATS Pick and Confidence  
  - Total Pick and Confidence  
  - Predicted Home and Away Scores  
- Computes Fireball ratings 🔥  
- Outputs to `mlb_predictions_all.csv` and `mlb_predictions_today.csv`

---

## 🖥️ Visualization Dashboard

**Script:** `mlb_predictions_dashboard.py`  
**Tool:** [Streamlit](https://streamlit.io)  

### Dashboard Sections:
- **Top 10 ATS Confidence Picks**
- **Top 10 Total (O/U) Confidence Picks**
- **Full Prediction Table**
- **SHAP Feature Importance**
- **Accuracy Summaries:**
  - Model vs Actual (daily + global)
  - Model vs Vegas (daily + global)

---

## 📁 Key Data Files

| File | Description |
|------|-------------|
| `data/mlb_boxscores_cleaned.csv` | Actual game outcomes from ESPN |
| `data/mlb_odds_mybookie.csv` | Betting odds data scraped from The Odds API |
| `data/mlb_model_and_odds.csv` | Merged dataset of scores + odds |
| `data/mlb_training_ats.csv` | Processed feature matrix for model training |
| `data/mlb_predictions_all.csv` | Master file of all predictions |
| `data/mlb_predictions_today.csv` | Filtered output for today's games |

---

## 🧠 Models Trained

| Model | Type | Target |
|-------|------|--------|
| `model_ats.pkl` | Classifier | ATS winner (Home/Away) |
| `model_total.pkl` | Classifier | Total (Over/Under) |
| `model_home_score.pkl` | Regressor | Home team score |
| `model_away_score.pkl` | Regressor | Away team score |

---

## 🔁 Main Pipeline File

**Script:** `ScratchModelV1.py`  
- Runs the full modeling pipeline sequentially  
- Optionally pushes results to GitHub at the end  

---

## 📦 Deployment

- Deployed to Streamlit via:
  - **Repo:** `https://github.com/radgator13/mlb-scratch-model-v1`
  - **App:** `https://mlb-scratch-model-davesbestpicks.streamlit.app`
- Requires `requirements.txt` in the root directory

---

## ✅ Setup Instructions

1. Clone the repo  
2. Run `ScratchModelV1.py` to generate predictions  
3. Launch dashboard:  
   ```bash
   streamlit run mlb_predictions_dashboard.py
