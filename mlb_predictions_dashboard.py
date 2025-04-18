import streamlit as st
import pandas as pd
import shap
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

st.set_page_config("MLB Prediction Viewer", page_icon="📊", layout="wide")
st.title("📊 MLB Model Predictions Dashboard")
st.markdown("Explore predictions, confidence levels, and fireball ratings.")

# === Load Data ===
df = pd.read_csv("data/mlb_predictions_all.csv")
df = df.drop_duplicates(subset=["Game Date", "Away Team", "Home Team"], keep="first")
df["Game Date"] = pd.to_datetime(df["Game Date"], errors="coerce")

# === Date Picker ===
all_dates = sorted(df["Game Date"].dropna().dt.date.unique())
default_date = max(all_dates) if all_dates else datetime.today().date()
st.markdown("### 📅 Select Game Date")
selected_date = st.date_input("Select date", default_date, min_value=min(all_dates), max_value=max(all_dates))
filtered_df = df[df["Game Date"].dt.date == selected_date]

# === Calculated Fields ===
filtered_df["Actual Total"] = filtered_df["Home Score"] + filtered_df["Away Score"]
filtered_df["Actual Spread"] = filtered_df["Home Score"] - filtered_df["Away Score"]
filtered_df["Model ATS Team"] = filtered_df.apply(
    lambda row: row["Home Team"] if row["Model ATS Pick"] == "Home" else row["Away Team"], axis=1
)
filtered_df["Model Spread Margin"] = (filtered_df["Predicted Home Score"] - filtered_df["Predicted Away Score"]).round(1)
filtered_df["Model Spread Description"] = filtered_df.apply(
    lambda row: f"{row['Model ATS Team']} to win by {abs(row['Model Spread Margin']):.0f} pts", axis=1
)
filtered_df["Vegas Spread"] = filtered_df.apply(
    lambda row: f"{row['Home Team']} {row['Spread Home']} / {row['Away Team']} {row['Spread Away']}", axis=1
)
filtered_df["Model Total (Est)"] = (filtered_df["Predicted Home Score"] + filtered_df["Predicted Away Score"]).round(1)

# === Page Header ===
st.subheader(f"🧠 Predictions for {selected_date.strftime('%B %d, %Y')}")
st.write(f"Total games: {len(filtered_df)}")

# === Top ATS Picks ===
st.markdown("🔥 **Top 10 ATS Confidence Picks**")
top_ats = filtered_df.sort_values(by="ATS Confidence", ascending=False).head(10)
top_ats["Predicted Score"] = top_ats["Home Team"] + " " + top_ats["Predicted Home Score"].fillna(0).round(0).astype(int).astype(str) + " / " + \
                             top_ats["Away Team"] + " " + top_ats["Predicted Away Score"].fillna(0).round(0).astype(int).astype(str)
top_ats["Actual Score"] = top_ats["Home Team"] + " " + top_ats["Home Score"].fillna(0).round(0).astype(int).astype(str) + " / " + \
                          top_ats["Away Team"] + " " + top_ats["Away Score"].fillna(0).round(0).astype(int).astype(str)

st.dataframe(top_ats[[ 
    "Game Date", "Away Team", "Home Team",
    "Model Spread Description", "Vegas Spread",
    "Predicted Score", "Actual Score",
    "ATS Confidence", "ATS Fireballs"
]].style.format({"ATS Confidence": "{:.2%}"}))

# === Top Total Picks ===
st.markdown("🔥 **Top 10 Over/Under Confidence Picks**")
top_total = filtered_df.sort_values(by="Total Confidence", ascending=False).head(10)
st.dataframe(top_total[[ 
    "Game Date", "Away Team", "Home Team",
    "Model Total (Est)", "Total", "Actual Total",
    "Predicted Home Score", "Predicted Away Score",
    "Home Score", "Away Score",
    "Total Confidence", "Total Fireballs"
]].style.format({
    "Model Total (Est)": "{:.1f}",
    "Total": "{:.1f}",
    "Actual Total": "{:.0f}",
    "Predicted Home Score": "{:.0f}",
    "Predicted Away Score": "{:.0f}",
    "Home Score": "{:.0f}",
    "Away Score": "{:.0f}",
    "Total Confidence": "{:.2%}"
}))

# === Full Table ===
st.markdown("📋 **Full Predictions**")
st.dataframe(filtered_df.style.format({
    "Spread Home": "{:.1f}",
    "Spread Away": "{:.1f}",
    "Spread_Diff": "{:.1f}",
    "Total": "{:.1f}",
    "Model Total (Est)": "{:.1f}",
    "Predicted Home Score": "{:.0f}",
    "Predicted Away Score": "{:.0f}",
    "Home Score": "{:.0f}",
    "Away Score": "{:.0f}",
    "ATS Confidence": "{:.2%}",
    "Total Confidence": "{:.2%}"
}))

# === Model Accuracy Summary ===
st.markdown("## ✅ Prediction Accuracy Summary")

def compute_accuracy(data):
    data = data.dropna(subset=[
        "Home Score", "Away Score", "Predicted Home Score",
        "Predicted Away Score", "Total", "Model Total (Est)"
    ])
    data["Model Predicted Winner"] = np.where(data["Predicted Home Score"] > data["Predicted Away Score"], "Home", "Away")
    data["Actual Winner"] = np.where(data["Home Score"] > data["Away Score"], "Home", "Away")
    data["Correct ATS"] = data["Model Predicted Winner"] == data["Actual Winner"]
    data["Correct Total"] = data.apply(lambda row: (
        (row["Model Total (Est)"] > row["Total"] and row["Actual Total"] > row["Total"]) or
        (row["Model Total (Est)"] < row["Total"] and row["Actual Total"] < row["Total"])
    ), axis=1)
    return data

# Daily accuracy
filtered_eval = compute_accuracy(filtered_df.copy())
daily_ats_acc = filtered_eval["Correct ATS"].mean()
daily_total_acc = filtered_eval["Correct Total"].mean()

# Overall accuracy
df["Actual Total"] = df["Home Score"] + df["Away Score"]
df["Model Total (Est)"] = df["Predicted Home Score"] + df["Predicted Away Score"]
overall_eval = compute_accuracy(df.copy())
overall_ats_acc = overall_eval["Correct ATS"].mean()
overall_total_acc = overall_eval["Correct Total"].mean()

st.markdown(f"### 📅 {selected_date.strftime('%B %d, %Y')} Accuracy")
st.write(f"- ATS Accuracy: {daily_ats_acc:.2%}")
st.write(f"- Total Accuracy: {daily_total_acc:.2%}")

st.markdown("### 🌐 Overall Model Accuracy")
st.write(f"- ATS Accuracy: {overall_ats_acc:.2%}")
st.write(f"- Total Accuracy: {overall_total_acc:.2%}")

# === Vegas Accuracy Comparison ===
st.markdown("## 🧮 Model vs Vegas Accuracy Summary")

def compare_vegas(data):
    data = data.dropna(subset=["Spread Home", "Spread Away", "Total", "Home Score", "Away Score"])
    data["Vegas ATS Winner"] = np.where(
        data["Home Score"] + data["Spread Home"] > data["Away Score"] + data["Spread Away"], "Home",
        np.where(data["Home Score"] + data["Spread Home"] < data["Away Score"] + data["Spread Away"], "Away", "Push")
    )
    data["Actual ATS Winner"] = np.where(
        data["Home Score"] > data["Away Score"], "Home",
        np.where(data["Home Score"] < data["Away Score"], "Away", "Push")
    )
    data["Vegas ATS Correct"] = data["Vegas ATS Winner"] == data["Actual ATS Winner"]
    data["Vegas Total Correct"] = data.apply(lambda row: (
        (row["Actual Total"] > row["Total"] and row["Total"] > 0) or
        (row["Actual Total"] < row["Total"] and row["Total"] > 0)
    ), axis=1)
    return data

vegas_today = compare_vegas(filtered_df.copy())
vegas_all = compare_vegas(df.copy())

st.markdown(f"### 📅 Vegas Accuracy on {selected_date.strftime('%B %d, %Y')}")
st.write(f"- ATS Accuracy: {vegas_today['Vegas ATS Correct'].mean():.2%}")
st.write(f"- Total Accuracy: {vegas_today['Vegas Total Correct'].mean():.2%}")

st.markdown("### 📊 Overall Vegas Accuracy")
st.write(f"- ATS Accuracy: {vegas_all['Vegas ATS Correct'].mean():.2%}")
st.write(f"- Total Accuracy: {vegas_all['Vegas Total Correct'].mean():.2%}")

# === SHAP Breakdown ===
st.markdown("🧠 **SHAP Feature Importance**")
try:
    model = joblib.load("models/model_ats.pkl")
    features = joblib.load("models/trained_features.pkl")
    X = filtered_df[features].copy()
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig)
except Exception as e:
    st.warning(f"⚠️ Unable to show SHAP plot: {e}")

# === Download Predictions ===
st.download_button(
    "⬇️ Download Predictions",
    filtered_df.to_csv(index=False),
    file_name=f"mlb_predictions_{selected_date}.csv",
    mime="text/csv"
)
