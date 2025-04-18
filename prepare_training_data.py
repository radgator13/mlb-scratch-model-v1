import pandas as pd
import numpy as np
import os

# === Load merged historical data ===
input_file = "data/mlb_model_and_odds.csv"
if not os.path.exists(input_file):
    raise FileNotFoundError(f"❌ Missing data file: {input_file}")

df = pd.read_csv(input_file)
df["Game Date"] = pd.to_datetime(df["Game Date"], errors="coerce")

# === Ensure necessary columns exist ===
required_cols = [
    "Home Team", "Away Team", "Home Score", "Away Score",
    "Spread Home", "Total", "ML Home", "ML Away"
]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing required column: {col}")

# === Create ATS target ===
df["ATS Winner"] = np.where(
    df["Home Score"] + df["Spread Home"] > df["Away Score"], "Home",
    np.where(df["Home Score"] + df["Spread Home"] < df["Away Score"], "Away", "Push")
)
df = df[df["ATS Winner"] != "Push"]

# === Feature Engineering ===
df["Spread_Diff"] = df["Spread Home"]
df["ML_Diff"] = np.log(df["ML Home"]) - np.log(df["ML Away"])
df["Log_Odds_Diff"] = np.log(df["Over Odds"]) - np.log(df["Under Odds"])

# === Drop rows with missing or infinite values ===
features = ["Spread_Diff", "ML_Diff", "Log_Odds_Diff", "Total"]
df = df[features + ["ATS Winner"]].replace([np.inf, -np.inf], np.nan).dropna()

# === Save cleaned training set ===
df.to_csv("data/mlb_training_ats.csv", index=False)
print(f"✅ Training data saved to data/mlb_training_ats.csv — {len(df)} rows")
