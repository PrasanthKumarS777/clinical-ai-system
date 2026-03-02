import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os, sys
sys.path.insert(0, os.path.abspath("."))

def train_fusion_model():
    print("Training Fusion Model (XGBoost + NLP)...")

    struct = pd.read_csv("data/processed/patient_features.csv")
    notes  = pd.read_csv("data/processed/clinical_notes_clean.csv")

    # DO NOT lowercase structured columns – must match XGBoost training
    # struct.columns remain: 'Id','AGE','GENDER','RACE',...

    # For notes only, normalize to lowercase
    notes.columns = [c.lower() for c in notes.columns]

    struct["RACE"]      = struct["RACE"].fillna("unknown").astype("category").cat.codes
    struct["ETHNICITY"] = struct["ETHNICITY"].fillna("unknown").astype("category").cat.codes

    # Load models
    xgb_model  = joblib.load("models/mortality_xgb.pkl")
    nlp_model  = joblib.load("models/nlp_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    # Use exact same feature set as during training
    X_struct = struct.drop(columns=["Id", "DECEASED"])
    y        = struct["DECEASED"]

    # XGBoost probabilities for all patients
    xgb_probs = xgb_model.predict_proba(X_struct)[:, 1]

    # Match notes to patients by any reasonable id column
    id_col = None
    for c in ["patient", "id", "patient_id"]:
        if c in notes.columns:
            id_col = c
            break

    if id_col:
        merged = struct[["Id", "DECEASED"]].merge(
            notes[[id_col, "transcription"]],
            left_on="Id", right_on=id_col, how="left"
        )
    else:
        merged = struct[["Id", "DECEASED"]].copy()
        merged["transcription"] = ""

    merged["transcription"] = merged["transcription"].fillna("")

    # NLP probabilities
    text_tfidf = vectorizer.transform(merged["transcription"])
    nlp_probs  = nlp_model.predict_proba(text_tfidf)[:, 1]

    # Simple weighted fusion
    fusion_probs = 0.7 * xgb_probs + 0.3 * nlp_probs

    # Meta-learner on top of both probabilities
    X_meta = np.column_stack([xgb_probs, nlp_probs])
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_meta, y, test_size=0.2, random_state=42, stratify=y
    )

    meta = LogisticRegression(class_weight="balanced", random_state=42)
    meta.fit(X_train_m, y_train_m)

    meta_auc   = roc_auc_score(y_test_m, meta.predict_proba(X_test_m)[:,1])
    xgb_auc    = roc_auc_score(y, xgb_probs)
    nlp_auc    = roc_auc_score(y, nlp_probs)
    fusion_auc = roc_auc_score(y, fusion_probs)

    print("\nAUC Comparison:")
    print(f"  XGBoost only      : {xgb_auc:.3f}")
    print(f"  NLP only          : {nlp_auc:.3f}")
    print(f"  Fusion (0.7/0.3)  : {fusion_auc:.3f}")
    print(f"  Meta-learner      : {meta_auc:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(meta, "models/fusion_model.pkl")

    results = pd.DataFrame({
        "Model": ["XGBoost Only", "NLP Only", "Fusion (Weighted)", "Meta-Learner"],
        "AUC":   [xgb_auc, nlp_auc, fusion_auc, meta_auc]
    })
    results.to_csv("output/fusion_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["steelblue", "salmon", "gold", "green"]
    ax.barh(results["Model"], results["AUC"], color=colors)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUC Score")
    ax.set_title("Fusion Model Comparison")
    for i, v in enumerate(results["AUC"]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center")
    plt.tight_layout()
    plt.savefig("output/fusion_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅ Fusion model saved → models/fusion_model.pkl")
    return results

if __name__ == "__main__":
    train_fusion_model()
