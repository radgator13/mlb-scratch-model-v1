import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import os

# === Load cleaned training data ===
file_path = "data/mlb_training_ats.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("❌ Training data file not found.")

df = pd.read_csv(file_path)

# === Feature and target selection ===
features_ats = ["Spread_Diff", "ML_Diff", "Log_Odds_Diff", "Total"]
target_ats = "ATS Winner"

X = df[features_ats]
y = df[target_ats]

# === Encode target labels ===
label_encoder_ats = LabelEncoder()
y_encoded = label_encoder_ats.fit_transform(y)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === Train LightGBM model ===
model_ats = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
model_ats.fit(X_train, y_train)

# === Evaluate ===
acc = model_ats.score(X_test, y_test)
print(f"✅ Model accuracy on test set: {acc:.2%}")

# === Save artifacts ===
os.makedirs("models", exist_ok=True)
joblib.dump(model_ats, "models/model_ats.pkl")
joblib.dump(label_encoder_ats, "models/label_encoder_ats.pkl")
joblib.dump(features_ats, "models/trained_features_ats.pkl")
print("💾 ATS model, label encoder, and features saved to /models")
