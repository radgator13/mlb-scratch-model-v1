import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# === Load merged data ===
df = pd.read_csv("data/mlb_model_and_odds.csv")

# === Drop rows without score or total line ===
df = df.dropna(subset=["Home Score", "Away Score", "Total"])

# === Derive Total Result ===
df["Total Line"] = df["Total"]
df["Total Points"] = df["Home Score"] + df["Away Score"]

df["Total Result"] = np.where(
    df["Total Points"] > df["Total Line"], "Over",
    np.where(df["Total Points"] < df["Total Line"], "Under", "Push")
)

df = df[df["Total Result"] != "Push"]

# === Encode target ===
lb_total = LabelEncoder()
df["Total Code"] = lb_total.fit_transform(df["Total Result"])

# === Feature selection ===
features = [
    "ML Home", "ML Away",
    "Spread Home", "Spread Home Odds", "Spread Away", "Spread Away Odds",
    "Total", "Over Odds", "Under Odds"
]

df = df.dropna(subset=features + ["Total Code"])

X = df[features]
y = df["Total Code"]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train LightGBM ===
clf = lgb.LGBMClassifier(random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Total model accuracy: {acc:.2%}")

# === Save Model + Encoder ===
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/model_total.pkl")
joblib.dump(lb_total, "models/label_encoder_total.pkl")
joblib.dump(features, "models/trained_features_total.pkl")

print("💾 Total model, encoder, and features saved to /models")
