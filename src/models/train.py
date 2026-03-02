import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, classification_report, 
                              confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.abspath("."))

# ── Load data ──────────────────────────────────────────────
print("Loading validated data...")
df = pd.read_csv("data/processed/patient_features.csv")
print(f"Loaded {df.shape[0]} patients, {df.shape[1]} features")

# ── Encode categoricals ────────────────────────────────────
df["RACE"]      = df["RACE"].fillna("unknown").astype("category").cat.codes
df["ETHNICITY"] = df["ETHNICITY"].fillna("unknown").astype("category").cat.codes

# ── Features / target ──────────────────────────────────────
X = df.drop(columns=["Id", "DECEASED"])
y = df["DECEASED"]

print(f"Class balance — Alive: {(y==0).sum()} | Deceased: {(y==1).sum()}")

# ── Train / test split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── XGBoost model ──────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train==0).sum() / (y_train==1).sum(),  # handle imbalance
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=20
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# ── Metrics ────────────────────────────────────────────────
train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:,1])
test_auc  = roc_auc_score(y_test,  model.predict_proba(X_test)[:,1])
print(f"\nAUC  →  Train: {train_auc:.3f}  |  Test: {test_auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test)))

# ── Save model ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/mortality_xgb.pkl")
print("✅ Model saved → models/mortality_xgb.pkl")

# ── Feature importance ─────────────────────────────────────
importance = pd.DataFrame({
    "feature":    X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv("models/feature_importance.csv", index=False)
print("\nTop 10 features:")
print(importance.head(10).to_string(index=False))

# ── ROC curve plot ─────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {test_auc:.3f}")
plt.plot([0,1],[0,1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Mortality Prediction")
plt.legend(loc="lower right")
plt.tight_layout()
os.makedirs("output", exist_ok=True)
plt.savefig("output/roc_curve.png", dpi=150)
plt.close()
print("✅ ROC curve saved → output/roc_curve.png")
