import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.abspath("."))

def run_fairness():
    model = joblib.load("models/mortality_xgb.pkl")
    df = pd.read_csv("data/processed/patient_features.csv")

    race_labels = df["RACE"].copy()
    ethnicity_labels = df["ETHNICITY"].copy()

    df["RACE"] = df["RACE"].fillna("unknown").astype("category").cat.codes
    df["ETHNICITY"] = df["ETHNICITY"].fillna("unknown").astype("category").cat.codes

    X = df.drop(columns=["Id", "DECEASED"])
    df["PRED_PROB"] = model.predict_proba(X)[:, 1]
    df["PRED"] = model.predict(X)
    df["RACE_LABEL"] = race_labels
    df["ETHNICITY_LABEL"] = ethnicity_labels

    os.makedirs("output", exist_ok=True)

    # Mortality rate by race
    race_stats = df.groupby("RACE_LABEL").agg(
        actual_mortality=("DECEASED", "mean"),
        predicted_mortality=("PRED_PROB", "mean"),
        count=("DECEASED", "count")
    ).reset_index()
    race_stats.to_csv("output/fairness_race.csv", index=False)

    # Mortality rate by ethnicity
    eth_stats = df.groupby("ETHNICITY_LABEL").agg(
        actual_mortality=("DECEASED", "mean"),
        predicted_mortality=("PRED_PROB", "mean"),
        count=("DECEASED", "count")
    ).reset_index()
    eth_stats.to_csv("output/fairness_ethnicity.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(race_stats))
    axes[0].bar([i - 0.2 for i in x], race_stats["actual_mortality"], 0.4, label="Actual", color="steelblue")
    axes[0].bar([i + 0.2 for i in x], race_stats["predicted_mortality"], 0.4, label="Predicted", color="salmon")
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(race_stats["RACE_LABEL"], rotation=30)
    axes[0].set_title("Mortality Rate by Race"); axes[0].legend()

    x2 = range(len(eth_stats))
    axes[1].bar([i - 0.2 for i in x2], eth_stats["actual_mortality"], 0.4, label="Actual", color="steelblue")
    axes[1].bar([i + 0.2 for i in x2], eth_stats["predicted_mortality"], 0.4, label="Predicted", color="salmon")
    axes[1].set_xticks(list(x2)); axes[1].set_xticklabels(eth_stats["ETHNICITY_LABEL"], rotation=30)
    axes[1].set_title("Mortality Rate by Ethnicity"); axes[1].legend()

    plt.tight_layout()
    plt.savefig("output/fairness_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅ Fairness done → output/fairness_race.csv, fairness_ethnicity.csv, fairness_plot.png")
    print(race_stats.to_string(index=False))
    return race_stats, eth_stats

if __name__ == "__main__":
    run_fairness()
