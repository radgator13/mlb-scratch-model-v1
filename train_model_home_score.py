# train_model_home_score.py

import pandas as pd
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split
import numpy as np

# === Load training data ===
df = pd.read_csv("data/mlb_model_and_odds.csv")

# === Feature Engineering ===
def safe_div(a, b):
    return np.where(b != 0, a / b, np.nan)

df["ML_Diff"] = df["ML Home"] - df["ML Away"]
df["Log_Odds_Diff"] = np.log(safe_div(df["ML Home"], df["ML Away"]))
df["Spread_Diff"] = df["Spread Home"] - df["Spread Away"]

# === Features and target ===
features = ["Spread_Diff", "ML_Diff", "Log_Odds_Diff", "Total"]
target = "Home Score"

# === Drop rows with missing values ===
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
score = model.score(X_test, y_test)
print(f"✅ Home score model R²: {score:.2%}")

# === Save ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model_home_score.pkl")
joblib.dump(features, "models/trained_features_home_score.pkl")
print("💾 Saved model_home_score.pkl and feature list")
