import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os, sys
sys.path.insert(0, os.path.abspath("."))

def train_structured_models():
    print("Training structured models comparison...")

    df = pd.read_csv("data/processed/patient_features.csv")
    df["RACE"]      = df["RACE"].fillna("unknown").astype("category").cat.codes
    df["ETHNICITY"] = df["ETHNICITY"].fillna("unknown").astype("category").cat.codes

    X = df.drop(columns=["Id", "DECEASED"])
    y = df["DECEASED"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # 1. Logistic Regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_sc, y_train)
    results["Logistic Regression"] = roc_auc_score(y_test, lr.predict_proba(X_test_sc)[:,1])

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["Random Forest"] = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])

    # 3. XGBoost (already trained, reload)
    xgb_model = joblib.load("models/mortality_xgb.pkl")
    results["XGBoost"] = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1])

    os.makedirs("models", exist_ok=True)
    joblib.dump(lr,     "models/logistic_regression.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(rf,     "models/random_forest.pkl")

    results_df = pd.DataFrame(list(results.items()), columns=["Model","AUC"]).sort_values("AUC", ascending=False)
    results_df.to_csv("output/model_comparison.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["gold" if m == "XGBoost" else "steelblue" for m in results_df["Model"]]
    ax.barh(results_df["Model"], results_df["AUC"], color=colors)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Test AUC")
    ax.set_title("Model Comparison — Structured Data")
    for i, (v, m) in enumerate(zip(results_df["AUC"], results_df["Model"])):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center")
    plt.tight_layout()
    plt.savefig("output/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n✅ Structured models trained and saved!")
    print(results_df.to_string(index=False))
    return results_df

if __name__ == "__main__":
    train_structured_models()
