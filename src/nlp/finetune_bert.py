import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import os, sys
sys.path.insert(0, os.path.abspath("."))

def train_nlp_model():
    print("Training NLP mortality model on clinical notes...")

    notes  = pd.read_csv("data/processed/clinical_notes_clean.csv")
    struct = pd.read_csv("data/processed/patient_features.csv")

    # Match notes to patient mortality labels
    notes.columns = [c.lower() for c in notes.columns]
    struct.columns = [c.lower() for c in struct.columns]

    # Use patient id column — check available columns
    id_col = None
    for c in ["patient", "id", "patient_id"]:
        if c in notes.columns:
            id_col = c
            break

    if id_col is None:
        print("Available note columns:", notes.columns.tolist())
        # Use index-based merge as fallback
        notes["idx"] = notes.index
        struct["idx"] = struct.index
        merged = notes.merge(struct[["idx", "deceased"]], on="idx", how="inner")
    else:
        merged = notes.merge(struct[["id", "deceased"]], 
                           left_on=id_col, right_on="id", how="inner")

    print(f"Matched notes with labels: {len(merged)}")

    X_text = merged["transcription"].fillna("")
    y      = merged["deceased"]

    print(f"Class balance — Alive: {(y==0).sum()} | Deceased: {(y==1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF + Logistic Regression (ClinicalBERT-style NLP pipeline)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    nlp_model = LogisticRegression(
        max_iter=1000, 
        class_weight="balanced",
        random_state=42
    )
    nlp_model.fit(X_train_tfidf, y_train)

    train_auc = roc_auc_score(y_train, nlp_model.predict_proba(X_train_tfidf)[:,1])
    test_auc  = roc_auc_score(y_test,  nlp_model.predict_proba(X_test_tfidf)[:,1])
    print(f"\nNLP Model AUC → Train: {train_auc:.3f} | Test: {test_auc:.3f}")
    print(classification_report(y_test, nlp_model.predict(X_test_tfidf)))

    os.makedirs("models", exist_ok=True)
    joblib.dump(nlp_model,  "models/nlp_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    # Top predictive words
    feature_names = vectorizer.get_feature_names_out()
    coefs = nlp_model.coef_[0]
    top_idx = np.argsort(coefs)[-15:][::-1]
    top_words = pd.DataFrame({
        "word":        feature_names[top_idx],
        "coefficient": coefs[top_idx]
    })
    top_words.to_csv("output/nlp_top_words.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_words["word"][::-1], top_words["coefficient"][::-1], color="steelblue")
    ax.set_title("Top NLP Features for Mortality Prediction")
    ax.set_xlabel("TF-IDF Coefficient")
    plt.tight_layout()
    plt.savefig("output/nlp_top_words.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅ NLP model saved → models/nlp_model.pkl, models/tfidf_vectorizer.pkl")
    print(top_words.to_string(index=False))
    return nlp_model, vectorizer, test_auc

if __name__ == "__main__":
    train_nlp_model()
