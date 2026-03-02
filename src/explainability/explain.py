import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath("."))

def run_explainability():
    print("Running SHAP explainability...")

    model = joblib.load("models/mortality_xgb.pkl")
    df    = pd.read_csv("data/processed/patient_features.csv")

    df["RACE"]      = df["RACE"].fillna("unknown").astype("category").cat.codes
    df["ETHNICITY"] = df["ETHNICITY"].fillna("unknown").astype("category").cat.codes

    X = df.drop(columns=["Id", "DECEASED"])

    # SHAP values
    explainer  = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    os.makedirs("output", exist_ok=True)

    # Summary bar plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("output/shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved → output/shap_summary_bar.png")

    # Summary beeswarm plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("output/shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved → output/shap_summary_beeswarm.png")

    # Save mean SHAP values
    shap_df = pd.DataFrame({
        "feature":    X.columns,
        "mean_shap":  np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_shap", ascending=False)
    shap_df.to_csv("output/shap_values.csv", index=False)
    print("✅ Saved → output/shap_values.csv")
    print(shap_df.head(10).to_string(index=False))

if __name__ == "__main__":
    run_explainability()
