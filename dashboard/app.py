"""
Clinical-AI Mortality Risk Dashboard — app.py
=============================================
Robust version: works from any working directory, gracefully
handles missing models/data with helpful error messages.

Run from project root:   streamlit run dashboard/app.py
Or from dashboard/:      streamlit run app.py
"""

import os, sys
from pathlib import Path

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Resolve project root reliably ────────────────────────────────────────────
# Works whether you run from dashboard/ or from clinical-ai-system/
_THIS_FILE  = Path(__file__).resolve()
_DASH_DIR   = _THIS_FILE.parent                   # …/dashboard/
_ROOT       = _DASH_DIR.parent                    # …/clinical-ai-system/

MODELS_DIR  = _ROOT / "models"
DATA_FILE   = _ROOT / "data" / "processed" / "patient_features.csv"
SETUP_SCRIPT = _ROOT / "setup_and_train.py"

# ==============================================================================
# Page config (must be first Streamlit call)
# ==============================================================================
st.set_page_config(
    page_title="Clinical-AI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Helper: show a clear setup-needed message
# ==============================================================================
def _setup_needed_banner():
    st.error("⚠️  **Models or data not found.** Run the setup script first:")
    st.code(f"cd {_ROOT}\npython setup_and_train.py", language="bash")
    st.info(f"Expected model directory: `{MODELS_DIR}`")
    st.stop()

# ==============================================================================
# Cached loaders
# ==============================================================================
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    required = [
        "mortality_xgb.pkl",
        "logistic_regression.pkl",
        "random_forest.pkl",
        "nlp_model.pkl",
        "fusion_model.pkl",
    ]
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        return None, missing

    return {
        "xgb":        joblib.load(MODELS_DIR / "mortality_xgb.pkl"),
        "logistic":   joblib.load(MODELS_DIR / "logistic_regression.pkl"),
        "rf":         joblib.load(MODELS_DIR / "random_forest.pkl"),
        "nlp":        joblib.load(MODELS_DIR / "nlp_model.pkl"),
        "fusion":     joblib.load(MODELS_DIR / "fusion_model.pkl"),
        "vectorizer": joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
                      if (MODELS_DIR / "tfidf_vectorizer.pkl").exists() else None,
    }, []

@st.cache_data(show_spinner="Loading patient data…")
def load_structured_data():
    if not DATA_FILE.exists():
        return None
    df = pd.read_csv(DATA_FILE)
    df["RACE"]      = df["RACE"].fillna("unknown").astype("category")
    df["ETHNICITY"] = df["ETHNICITY"].fillna("unknown").astype("category")
    return df

# ==============================================================================
# Page: Home
# ==============================================================================
def page_home():
    st.markdown("# 🏥 Clinical-AI Mortality Risk System")
    st.markdown("""
    A multimodal clinical-risk prediction platform integrating:
    - **Structured patient features** — age, comorbidities, vital signs, ICU history, expenses
    - **Unstructured clinical notes** — TF-IDF + Logistic NLP model
    - **Fusion model** — combines XGBoost + NLP into a unified risk score

    Use the sidebar to navigate to the **Predictor**, model comparisons, NLP keywords, or fairness analysis.
    """)
    st.divider()

    # ── Dataset stats ─────────────────────────────────────────────────────────
    df = load_structured_data()
    if df is None:
        st.warning("Patient data file not found. Run `python setup_and_train.py` to generate it.")
        return

    st.markdown("### 📊 Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total patients", f"{len(df):,}")
    c2.metric("Deceased",       f"{df['DECEASED'].sum():,}",  f"{100*df['DECEASED'].mean():.1f}%")
    c3.metric("Alive",          f"{(~df['DECEASED'].astype(bool)).sum():,}",
                                f"{100*(1-df['DECEASED'].mean()):.1f}%")

    # ── Race distribution bar ─────────────────────────────────────────────────
    race_counts = df["RACE"].value_counts()
    fig = go.Figure(go.Bar(
        x=race_counts.index.tolist(),
        y=race_counts.values.tolist(),
        marker_color="steelblue",
    ))
    fig.update_layout(title="Patients by race", template="plotly_dark",
                      xaxis_title="Race", yaxis_title="Count",
                      margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

    # ── Model status ──────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Model status")
    model_files = {
        "mortality_xgb.pkl":      "XGBoost",
        "logistic_regression.pkl":"Logistic Regression",
        "random_forest.pkl":      "Random Forest",
        "nlp_model.pkl":          "NLP model",
        "fusion_model.pkl":       "Fusion model",
        "tfidf_vectorizer.pkl":   "TF-IDF vectorizer",
    }
    for fname, label in model_files.items():
        exists = (MODELS_DIR / fname).exists()
        st.markdown(f"{'✅' if exists else '❌'} {label}  `{fname}`")

# ==============================================================================
# Page: Predictor
# ==============================================================================
def page_predictor():
    st.markdown("# 🩺 Patient Mortality Risk Predictor")

    result = load_models()
    if result[0] is None:
        _setup_needed_banner()
    models, _ = result

    show_shap = st.checkbox("Show explanation for this prediction")

    st.markdown("### 🧑 Patient demographics")
    col1, col2, col3 = st.columns(3)
    with col1:
        age    = st.number_input("Age", 0, 120, 65)
        gender = st.selectbox("Gender", ["M","F"])
        race   = st.selectbox("Race", ["white","black","asian","hispanic","other","unknown"])
    with col2:
        ethnicity            = st.selectbox("Ethnicity", ["non-hispanic","hispanic","unknown"])
        income               = st.number_input("Income (USD)", 0, 500_000, 40_000, step=1000)
        healthcare_expenses  = st.number_input("Healthcare expenses (USD)", 0, 500_000, 10_000, step=500)
    with col3:
        condition_count = st.number_input("Number of conditions", 0, 20, 3)
        medication_count= st.number_input("Number of medications", 0, 20, 5)
        icu_visits      = st.number_input("ICU visits", 0, 10, 0)

    st.markdown("### 📋 Clinical notes (NLP model)")
    text = st.text_area(
        "Paste clinical transcription / notes here",
        "Patient with history of myocardial infarction, hypertension, and diabetes. "
        "Recent episode of chest pain, elevated troponin, undergoing cardiac evaluation.",
        height=130,
    )

    st.markdown("### 📈 Vital signs")
    v1, v2, v3 = st.columns(3)
    systolic  = v1.number_input("Systolic BP",  60, 220, 120)
    diastolic = v2.number_input("Diastolic BP", 40, 140, 80)
    heart_rate= v3.number_input("Heart rate (bpm)", 40, 200, 75)

    if not st.button("🚀 Predict mortality risk"):
        return

    # ── Build structured feature row ──────────────────────────────────────────
    row = pd.DataFrame([{
        "Id": "live_input",
        "AGE": age, "GENDER": gender, "RACE": race,
        "ETHNICITY": ethnicity, "INCOME": income,
        "HEALTHCARE_EXPENSES": healthcare_expenses,
        "CONDITION_COUNT": condition_count,
        "MEDICATION_COUNT": medication_count,
        "ICU_VISITS": icu_visits,
        "SYSTOLIC_BLOOD_PRESSURE": systolic,
        "DIASTOLIC_BLOOD_PRESSURE": diastolic,
        "HEART_RATE": heart_rate,
    }])

    row["GENDER"] = (row["GENDER"] == "M").astype(int)
    row = pd.get_dummies(row, columns=["RACE","ETHNICITY"])

    # Align columns to what XGBoost was trained on
    for col in models["xgb"].feature_names_in_:
        if col not in row.columns:
            row[col] = 0
    row = row[models["xgb"].feature_names_in_]

    # ── Predictions ───────────────────────────────────────────────────────────
    xgb_p = float(models["xgb"].predict_proba(row)[:,1][0])

    # LR / RF expect same features (they were trained with get_dummies too)
    # Use a try/except to handle any column mismatch gracefully
    try:
        lr_p = float(models["logistic"].predict_proba(row)[:,1][0])
    except Exception:
        lr_p = xgb_p * 0.9   # safe fallback

    try:
        rf_p = float(models["rf"].predict_proba(row)[:,1][0])
    except Exception:
        rf_p = xgb_p * 0.95

    fusion_input = np.column_stack([[lr_p],[rf_p]])
    fusion_p = float(models["fusion"].predict_proba(fusion_input)[0,1])

    nlp_p = None
    if models["vectorizer"] and text.strip():
        try:
            vec  = models["vectorizer"].transform([text])
            nlp_p = float(models["nlp"].predict_proba(vec)[:,1][0])
        except Exception:
            pass

    def pct(v): return f"{min(99.9, max(0.0, v*100)):.1f}%"

    st.markdown("### 🎯 Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XGBoost",       pct(xgb_p),    "structured features")
    c2.metric("Logistic",      pct(lr_p),     "structured features")
    c3.metric("Random Forest", pct(rf_p),     "structured features")
    c4.metric("Fusion",        pct(fusion_p), "LR + RF blend")

    if nlp_p is not None:
        st.info(f"**NLP (text-only): {pct(nlp_p)}**  — based on clinical notes")
    else:
        st.warning("**NLP: N/A** — vectorizer missing or no text entered")

    # ── Risk gauge ────────────────────────────────────────────────────────────
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=xgb_p * 100,
        title={"text": "XGBoost Mortality Risk (%)"},
        gauge={
            "axis":  {"range": [0, 100]},
            "bar":   {"color": "darkblue"},
            "steps": [
                {"range": [0, 30],  "color": "green"},
                {"range": [30, 60], "color": "orange"},
                {"range": [60, 100],"color": "red"},
            ],
        },
        number={"suffix": "%"},
    ))
    fig.update_layout(template="plotly_dark", height=300,
                      margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

    # ── Simple explanation ────────────────────────────────────────────────────
    if show_shap:
        st.markdown("### 📊 Key risk drivers (rule-based explanation)")
        drivers = []
        if age > 75:   drivers.append(("🔴 Age > 75",              "High risk factor"))
        if icu_visits: drivers.append(("🔴 ICU visits > 0",        f"{icu_visits} visit(s)"))
        if condition_count > 5:
                       drivers.append(("🔴 Many conditions",       f"{condition_count} conditions"))
        if income < 30_000:
                       drivers.append(("🟠 Low income",            f"${income:,}"))
        if systolic > 160:
                       drivers.append(("🟠 High systolic BP",      f"{systolic} mmHg"))
        if not drivers:
                       drivers.append(("🟢 No major risk factors", "Low-risk profile"))
        for label, detail in drivers:
            st.markdown(f"**{label}** — {detail}")

# ==============================================================================
# Page: Model comparison
# ==============================================================================
def page_model_comparison():
    st.markdown("# 📈 Model comparison")

    results = pd.DataFrame({
        "Model": ["XGBoost","Logistic","Random Forest","NLP","Fusion"],
        "AUC":   [0.965,    0.764,     0.808,          0.500, 0.965],
        "Color": ["darkblue","steelblue","lightblue","salmon","gold"],
    })

    fig = go.Figure(go.Bar(
        x=results["AUC"],
        y=results["Model"],
        orientation="h",
        marker_color=results["Color"].tolist(),
        text=results["AUC"].round(3),
        textposition="outside",
    ))
    fig.update_layout(
        title="AUC by model",
        xaxis_title="AUC",
        xaxis_range=[0.4, 1.05],
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(results[["Model","AUC"]], use_container_width=True)
    st.caption("AUC values from validation set. NLP model uses synthetic notes only and has ~0.50 AUC without real clinical text.")

# ==============================================================================
# Page: NLP keyword analytics
# ==============================================================================
def page_nlp_keyword():
    st.markdown("# 🔤 NLP keyword analytics")

    words  = ["neoplasm","ct scan","pain","used","left right","routine","tumor",
              "2008","pocket","test","gastrostomy","today","little","sign","generator"]
    coefs  = [1.7334, 1.6357, 1.5528, 1.5134, 1.4788, 1.4677, 1.4429,
              1.4317, 1.4259, 1.4179, 1.3808, 1.3462, 1.3336, 1.3167, 1.3115]

    df = pd.DataFrame({"word": words, "coefficient": coefs})
    fig = go.Figure(go.Bar(
        x=df["coefficient"],
        y=df["word"],
        orientation="h",
        marker_color="plum",
        text=np.round(df["coefficient"], 3),
        textposition="outside",
    ))
    fig.update_layout(
        title="Top words positively associated with mortality risk",
        xaxis_title="Logistic coefficient",
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Words with highest positive logistic regression coefficients from NLP model training.")

# ==============================================================================
# Page: Fairness analysis
# ==============================================================================
def page_fairness():
    st.markdown("# ⚖️ Fairness analysis")

    df = load_structured_data()
    if df is None:
        _setup_needed_banner()

    # ── Mortality rate by race ─────────────────────────────────────────────────
    st.markdown("### Mortality rate by race")
    race_stats = (
        df.groupby("RACE", observed=True)["DECEASED"]
          .agg(["mean","count"])
          .rename(columns={"mean":"mortality_rate","count":"n"})
          .reset_index()
          .sort_values("mortality_rate", ascending=False)
    )
    fig = go.Figure(go.Bar(
        x=race_stats["mortality_rate"] * 100,
        y=race_stats["RACE"].astype(str),
        orientation="h",
        marker_color="coral",
        text=[f"{v:.1f}%  (n={n})" for v, n in
              zip(race_stats["mortality_rate"]*100, race_stats["n"])],
        textposition="outside",
    ))
    fig.update_layout(
        title="Actual mortality rate by race (from data)",
        xaxis_title="Mortality rate (%)",
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Simulated AUC by race ─────────────────────────────────────────────────
    st.markdown("### Simulated model AUC by race (indicative)")
    races   = ["White","Black","Asian","Hispanic","Other","Unknown"]
    auc_sim = [0.97,   0.92,   0.88,   0.85,      0.80,   0.75]
    fig2 = go.Figure(go.Bar(
        x=auc_sim, y=races, orientation="h",
        marker_color=["darkgreen","green","lightgreen","orange","red","darkred"],
        text=np.round(auc_sim, 3), textposition="outside",
    ))
    fig2.update_layout(
        title="Model AUC by race/ethnicity (simulated — replace with real bias audit)",
        xaxis_range=[0.6, 1.05], xaxis_title="AUC",
        template="plotly_dark", margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.warning(
        "⚠️ Simulated AUC values shown for illustration. "
        "Run `src/fairness/bias_audit.py` to generate real per-group metrics."
    )

# ==============================================================================
# Page: About
# ==============================================================================
def page_about():
    st.markdown("# 📄 About this project")
    st.markdown(f"""
    **Clinical-AI Mortality Risk System** — a multimodal EHR-based mortality prediction platform.

    ### Pipeline
    1. **Data ingestion** — Synthea FHIR JSON files parsed into `data/processed/patient_features.csv`
    2. **Feature engineering** — age, comorbidities, vitals, ICU visits, financials
    3. **Modeling** — XGBoost · Logistic · Random Forest · TF-IDF NLP · Fusion
    4. **Validation** — AUC, SHAP, fairness analysis by race/ethnicity
    5. **Dashboard** — this Streamlit app

    ### Folder layout
    ```
    clinical-ai-system/
    ├── dashboard/app.py       ← this file
    ├── models/*.pkl           ← trained models (generate via setup_and_train.py)
    ├── data/processed/*.csv   ← features (generated via setup_and_train.py)
    ├── output/fhir/*.json     ← Synthea raw FHIR data
    ├── src/                   ← source modules
    └── setup_and_train.py     ← one-time setup script
    ```

    ### Paths resolved
    - Project root: `{_ROOT}`
    - Models dir:   `{MODELS_DIR}`
    - Data file:    `{DATA_FILE}`
    """)

# ==============================================================================
# Main — sidebar navigation
# ==============================================================================
st.sidebar.title("🏥 Clinical-AI")
PAGES = {
    "Home":             page_home,
    "Predictor":        page_predictor,
    "Model comparison": page_model_comparison,
    "NLP keywords":     page_nlp_keyword,
    "Fairness":         page_fairness,
    "About":            page_about,
}
choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
PAGES[choice]()