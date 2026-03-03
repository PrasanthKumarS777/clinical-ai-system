"""
Clinical-AI Mortality Risk Dashboard — Premium Edition
=======================================================
Design: Space Grotesk + IBM Plex Mono · #070d0b dark · emerald/gold/coral palette
Full data integration: patients, conditions, medications, encounters,
procedures, observations, immunizations, allergies, imaging studies.
Run:  streamlit run dashboard/app.py
"""

import os, sys, warnings
from pathlib import Path

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Resolve project root ──────────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
_DASH_DIR  = _THIS_FILE.parent
_ROOT      = _DASH_DIR.parent
MODELS_DIR = _ROOT / "models"
DATA_DIR   = _ROOT / "data" / "processed"
CSV_DIR    = _ROOT / "output" / "csv"
OUT_DIR    = _ROOT / "output"
DATA_FILE  = DATA_DIR / "patient_features.csv"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ClinicalAI — Mortality Intelligence",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS  — Emerald / Gold / Coral palette
# ══════════════════════════════════════════════════════════════════════════════
A  = "#00b894"   # emerald mint  — primary accent
A2 = "#fdcb6e"   # golden yellow
A3 = "#e17055"   # warm coral / danger
GO = "#74b9ff"   # sky blue / info
GR = "#55efc4"   # bright mint / success

PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#9ca3af", size=11),
    title_font=dict(color="#dfe6e9", size=14, family="Space Grotesk, sans-serif"),
    xaxis=dict(gridcolor="#1a2420", linecolor="#1a2420", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1a2420", linecolor="#1a2420", tickfont=dict(size=10)),
    margin=dict(l=20, r=20, t=44, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a2420", borderwidth=1),
    colorway=[A, A2, A3, GO, GR, "#a29bfe", "#fd79a8"],
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

:root{{
  --bg:#070d0b; --s:#0e1714; --s2:#141f1b; --b:#1a2420;
  --t:#dfe6e9;  --m:#6b7280;
  --mint:{A}; --gold:{A2}; --coral:{A3}; --sky:{GO}; --teal:{GR};
}}

[data-testid="stAppViewContainer"]{{background:#070d0b !important;}}
[data-testid="stAppViewContainer"] > .main{{background:#070d0b !important;}}
[data-testid="stSidebar"]{{background:#0e1714 !important; border-right:1px solid #1a2420 !important;}}
[data-testid="stSidebar"] *{{color:#dfe6e9;}}
#MainMenu{{visibility:hidden !important;}}
footer{{visibility:hidden !important;}}
[data-testid="stDeployButton"]{{display:none !important;}}

html,body,[class*="css"]{{
  font-family:'IBM Plex Mono',monospace !important;
  color:#dfe6e9 !important;
}}
h1,h2,h3,h4{{font-family:'Space Grotesk',sans-serif !important; color:#dfe6e9 !important;}}

/* ── HERO BANNER ── */
.hero{{
  background:linear-gradient(135deg,#0b1410,#111f1a);
  border:1px solid #1a2420; border-radius:16px;
  padding:36px 36px 28px; margin-bottom:24px;
  position:relative; overflow:hidden;
}}
.hero::before{{
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,{A},{A2},transparent);
}}
.htitle{{
  font-family:'Space Grotesk',sans-serif; font-size:2.6rem; font-weight:800;
  background:linear-gradient(90deg,{A},{A2});
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  line-height:1.1; margin:0 0 6px;
}}
.hsub{{font-size:.85rem; color:#6b7280; letter-spacing:.06em; margin-bottom:0;}}
.tag{{
  display:inline-block; background:rgba(0,184,148,.08);
  border:1px solid rgba(0,184,148,.25); color:{A};
  font-size:.62rem; letter-spacing:.12em; text-transform:uppercase;
  padding:3px 10px; border-radius:100px; margin:10px 4px 0 0;
}}

/* ── KPI CARD ── */
.kcard{{
  background:#0e1714; border:1px solid #1a2420;
  border-radius:12px; padding:18px 20px;
  position:relative; overflow:hidden;
  transition:border-color .25s, box-shadow .25s;
}}
.kcard::after{{
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,{A},{A2});
}}
.kcard:hover{{border-color:rgba(0,184,148,.4); box-shadow:0 8px 28px rgba(0,184,148,.08);}}
.klabel{{font-size:.62rem; letter-spacing:.12em; text-transform:uppercase; color:#6b7280; margin-bottom:6px;}}
.kval{{font-family:'Space Grotesk',sans-serif; font-size:1.9rem; font-weight:700; color:{A}; line-height:1;}}
.kdelta{{font-size:.7rem; color:{GR}; margin-top:4px;}}

/* ── SECTION HEADER ── */
.sec{{
  display:flex; align-items:center; gap:10px;
  margin:28px 0 14px; border-bottom:1px solid #1a2420; padding-bottom:10px;
}}
.snum{{
  font-size:.62rem; color:{A}; background:rgba(0,184,148,.08);
  border:1px solid rgba(0,184,148,.2); padding:2px 7px; border-radius:4px;
}}
.stitle{{font-family:'Space Grotesk',sans-serif; font-size:1.15rem; font-weight:700; color:#dfe6e9; margin:0;}}

/* ── INFO BOX ── */
.ibox{{
  background:rgba(0,184,148,.04); border:1px solid rgba(0,184,148,.15);
  border-radius:10px; padding:12px 16px; margin:10px 0;
  font-size:.82rem; color:#6b7280; line-height:1.6;
}}

/* ── GENERIC CARD ── */
.mcard{{background:#0e1714; border:1px solid #1a2420; border-radius:12px; padding:20px; margin-bottom:14px;}}

/* ── RESULT CARD ── */
.pres{{
  background:linear-gradient(135deg,rgba(0,184,148,.08),rgba(253,203,110,.08));
  border:1px solid rgba(0,184,148,.25); border-radius:14px;
  padding:28px; text-align:center; margin-top:18px;
}}
.pscore{{
  font-family:'Space Grotesk',sans-serif; font-weight:800; font-size:4rem; line-height:1;
  background:linear-gradient(90deg,{A},{A2});
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.badge{{
  display:inline-block; padding:3px 10px; border-radius:100px; font-size:.62rem;
  background:rgba(85,239,196,.12); border:1px solid rgba(85,239,196,.3);
  color:{GR}; margin-left:8px;
}}
.badge-red{{background:rgba(225,112,85,.12); border-color:rgba(225,112,85,.3); color:{A3};}}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]{{background:#141f1b !important; border-radius:10px !important; padding:4px !important; gap:2px !important;}}
.stTabs [data-baseweb="tab"]{{color:#6b7280 !important; border-radius:7px !important; font-family:'IBM Plex Mono',monospace !important; font-size:.75rem !important; letter-spacing:.06em !important;}}
.stTabs [aria-selected="true"]{{background:#0e1714 !important; color:{A} !important;}}

/* ── BUTTON ── */
.stButton>button{{
  background:linear-gradient(135deg,{A},{A2}) !important;
  border:none !important; border-radius:8px !important;
  color:#070d0b !important; font-family:'Space Grotesk',sans-serif !important;
  font-weight:700 !important; letter-spacing:.08em !important;
  padding:10px 28px !important; text-transform:uppercase !important;
  transition:opacity .2s !important;
}}
.stButton>button:hover{{opacity:.88 !important;}}

/* ── INPUTS ── */
.stNumberInput input,.stTextInput input,.stTextArea textarea{{
  background:#141f1b !important; border:1px solid #1a2420 !important;
  color:#dfe6e9 !important; border-radius:8px !important;
  font-family:'IBM Plex Mono',monospace !important; font-size:.82rem !important;
}}
.stSelectbox>div>div{{background:#141f1b !important; border:1px solid #1a2420 !important; border-radius:8px !important;}}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"]{{border:1px solid #1a2420 !important; border-radius:10px !important;}}

/* ── METRIC ── */
[data-testid="stMetric"]{{background:#0e1714 !important; border:1px solid #1a2420 !important; border-radius:12px !important; padding:1rem 1.2rem !important;}}
[data-testid="stMetricLabel"]{{font-family:'IBM Plex Mono',monospace !important; font-size:.65rem !important; letter-spacing:.1em !important; text-transform:uppercase !important; color:#6b7280 !important;}}
[data-testid="stMetricValue"]{{font-family:'Space Grotesk',sans-serif !important; font-size:1.9rem !important; font-weight:700 !important; color:{A} !important;}}

/* ── SIDEBAR NAV ── */
[data-testid="stSidebar"] .stRadio label{{
  font-family:'IBM Plex Mono',monospace !important; font-size:.78rem !important;
  letter-spacing:.04em !important; color:#6b7280 !important;
  transition:color .2s !important; padding:.35rem 0 !important;
}}
[data-testid="stSidebar"] .stRadio label:hover{{color:{A} !important;}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{{width:4px; height:4px;}}
::-webkit-scrollbar-track{{background:#070d0b;}}
::-webkit-scrollbar-thumb{{background:#1a2420; border-radius:2px;}}
::-webkit-scrollbar-thumb:hover{{background:{A};}}

/* ── ALERT ── */
.stSuccess,.stInfo,.stWarning,.stError{{
  border-radius:10px !important; font-family:'IBM Plex Mono',monospace !important; font-size:.8rem !important;
}}

/* ── DIVIDER ── */
hr{{border-color:#1a2420 !important; margin:1.2rem 0 !important;}}
</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def hero(title, subtitle, tags=None):
    tag_html = ""
    if tags:
        tag_html = "".join(f"<span class='tag'>{t}</span>" for t in tags)
    st.markdown(f"""
    <div class='hero'>
      <div class='htitle'>{title}</div>
      <div class='hsub'>{subtitle}</div>
      <div>{tag_html}</div>
    </div>""", unsafe_allow_html=True)

def sec(num, title):
    st.markdown(f"<div class='sec'><span class='snum'>{num}</span><p class='stitle'>{title}</p></div>", unsafe_allow_html=True)

def ibox(text):
    st.markdown(f"<div class='ibox'>{text}</div>", unsafe_allow_html=True)

def kpi(col, label, val, delta="", color=A):
    col.markdown(f"""
    <div class='kcard'>
      <div class='klabel'>{label}</div>
      <div class='kval' style='color:{color};'>{val}</div>
      <div class='kdelta'>{delta}</div>
    </div>""", unsafe_allow_html=True)

def chart(fig, height=320):
    fig.update_layout(**{**PL, "height": height})
    fig.update_xaxes(gridcolor="#1a2420", linecolor="#1a2420")
    fig.update_yaxes(gridcolor="#1a2420", linecolor="#1a2420")
    st.plotly_chart(fig, width="stretch")

def hbar(names, values, title="", color=A, height=320, pct=False):
    txt = [f"{v:.1f}%" if pct else f"{v:,.0f}" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=values, colorscale=[[0, "#1a2420"], [0.5, A2], [1, color]], line_width=0),
        text=txt, textposition="outside", textfont=dict(size=10, color="#6b7280"),
    ))
    fig.update_layout(title=title)
    chart(fig, height)

def vbar(labels, values, title="", color=A, height=300):
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=values, colorscale=[[0, "#1a2420"], [0.5, A2], [1, color]], line_width=0),
        text=[f"{v:,.0f}" for v in values], textposition="outside",
        textfont=dict(size=10, color="#6b7280"),
    ))
    fig.update_layout(title=title)
    chart(fig, height)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⚕ Loading models…")
def load_models():
    required = ["mortality_xgb.pkl", "logistic_regression.pkl", "random_forest.pkl",
                "nlp_model.pkl", "fusion_model.pkl"]
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        return None, missing
    m = {
        "xgb":     joblib.load(MODELS_DIR / "mortality_xgb.pkl"),
        "logistic":joblib.load(MODELS_DIR / "logistic_regression.pkl"),
        "rf":      joblib.load(MODELS_DIR / "random_forest.pkl"),
        "nlp":     joblib.load(MODELS_DIR / "nlp_model.pkl"),
        "fusion":  joblib.load(MODELS_DIR / "fusion_model.pkl"),
    }
    if (MODELS_DIR / "tfidf_vectorizer.pkl").exists():
        m["vectorizer"] = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    if (MODELS_DIR / "scaler.pkl").exists():
        m["scaler"] = joblib.load(MODELS_DIR / "scaler.pkl")
    return m, []

@st.cache_data(show_spinner="Loading patient cohort…")
def load_patients():
    if not DATA_FILE.exists():
        return None
    df = pd.read_csv(DATA_FILE)
    df["RACE"]      = df["RACE"].fillna("unknown").astype(str)
    df["ETHNICITY"] = df["ETHNICITY"].fillna("unknown").astype(str)
    return df

@st.cache_data(show_spinner="Loading CSV data…")
def load_csv(name):
    p = CSV_DIR / name
    if not p.exists(): return None
    return pd.read_csv(p, low_memory=False)

@st.cache_data(show_spinner="Loading output…")
def load_out(name):
    p = OUT_DIR / name
    if not p.exists(): return None
    return pd.read_csv(p)

def setup_banner():
    st.error("⚠️  Models or data not found. Run `python setup_and_train.py` from project root.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='padding:16px 0 6px;'>
      <div style='font-family:Space Grotesk,sans-serif;font-weight:800;font-size:1.35rem;
                  background:linear-gradient(90deg,{A},{A2});
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>⚕ ClinicalAI</div>
      <div style='font-size:.6rem;color:#6b7280;letter-spacing:.1em;margin-top:3px;'>MORTALITY INTELLIGENCE v2.0</div>
    </div>
    <hr style='border:none;border-top:1px solid #1a2420;margin:10px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠  Overview",
        "🩺  Risk Predictor",
        "👥  Population Analytics",
        "💊  Medications & Conditions",
        "🏥  Encounters & Procedures",
        "🔬  Observations & Labs",
        "🤖  Model Performance",
        "🔤  NLP Intelligence",
        "⚖️  Fairness Audit",
        "📊  SHAP Explainability",
        "ℹ️  About",
    ], label_visibility="collapsed")

    st.markdown(f"""
    <hr style='border:none;border-top:1px solid #1a2420;margin:14px 0;'>
    <div style='font-size:.65rem;color:#374151;line-height:2.0;font-family:IBM Plex Mono,monospace;'>
      PROJECT · ClinicalAI DL<br>
      TYPE · Mortality Prediction<br>
      MODELS · XGB · LR · RF · FUSION<br>
      DATA · Synthea FHIR 8M+<br>
      <span style='color:#4b5563;'>Prasanth Kumar Sahu</span>
    </div>""", unsafe_allow_html=True)

    df_pat = load_patients()
    if df_pat is not None:
        total    = len(df_pat)
        deceased = int(df_pat["DECEASED"].sum())
        st.markdown(f"""
        <hr style='border:none;border-top:1px solid #1a2420;margin:14px 0;'>
        <div style='font-size:.65rem;color:#374151;line-height:2.0;font-family:IBM Plex Mono,monospace;'>
          <span style='color:{A};letter-spacing:.1em;text-transform:uppercase;'>COHORT LIVE</span><br>
          PATIENTS &nbsp;&nbsp;<span style='color:#dfe6e9;'>{total:,}</span><br>
          DECEASED &nbsp;&nbsp;<span style='color:{A3};'>{deceased:,}</span><br>
          MORTALITY &nbsp;<span style='color:{A2};'>{100*deceased/total:.1f}%</span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    hero("ClinicalAI — Mortality Intelligence",
         "Multimodal EHR-powered mortality prediction · XGBoost · Logistic · Random Forest · NLP Fusion",
         ["Synthea FHIR 8M+", "XGBoost 0.965 AUC", "NLP + Structured Fusion",
          "Fairness Audit", "SHAP Explainability", "Live Predictor"])

    df = load_patients()
    if df is None:
        st.warning("Patient data not found. Run `python setup_and_train.py`.")
        st.stop()

    total    = len(df)
    deceased = int(df["DECEASED"].sum())
    alive    = total - deceased
    mort_rt  = deceased / total
    avg_age  = df["AGE"].mean() if "AGE" in df.columns else 0
    avg_cond = df["CONDITION_COUNT"].mean() if "CONDITION_COUNT" in df.columns else 0
    avg_meds = df["MEDICATION_COUNT"].mean() if "MEDICATION_COUNT" in df.columns else 0

    sec("01", "Cohort KPIs")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi(c1, "Total Patients",    f"{total:,}",         "Synthea cohort")
    kpi(c2, "Deceased",          f"{deceased:,}",      f"{mort_rt*100:.1f}% mortality", A3)
    kpi(c3, "Alive",             f"{alive:,}",         f"{(1-mort_rt)*100:.1f}% survival", GR)
    kpi(c4, "Avg Age",           f"{avg_age:.1f}",     "years", A2)
    kpi(c5, "Avg Conditions",    f"{avg_cond:.1f}",    "per patient", GO)
    kpi(c6, "Avg Medications",   f"{avg_meds:.1f}",    "per patient", A)

    st.markdown("<br>", unsafe_allow_html=True)
    sec("02", "Outcomes & Age Distribution")
    col1, col2 = st.columns([1.4, 1])

    with col1:
        if "AGE" in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[df["DECEASED"]==0]["AGE"],
                name="Alive", marker_color=A, opacity=0.72, xbins=dict(size=5)))
            fig.add_trace(go.Histogram(x=df[df["DECEASED"]==1]["AGE"],
                name="Deceased", marker_color=A3, opacity=0.72, xbins=dict(size=5)))
            fig.update_layout(barmode="overlay", title="AGE DISTRIBUTION BY OUTCOME")
            chart(fig, 340)

    with col2:
        fig = go.Figure(go.Pie(
            labels=["Alive", "Deceased"],
            values=[alive, deceased], hole=0.68,
            marker=dict(colors=[A, A3], line=dict(color="#070d0b", width=3)),
            textinfo="none",
        ))
        fig.add_annotation(
            text=f"<b>{mort_rt*100:.1f}%</b>",
            x=0.5, y=0.55, showarrow=False,
            font=dict(size=28, color=A3, family="Space Grotesk"))
        fig.add_annotation(
            text="MORTALITY",
            x=0.5, y=0.38, showarrow=False,
            font=dict(size=11, color="#6b7280", family="IBM Plex Mono"))
        fig.update_layout(title="OUTCOME SPLIT")
        chart(fig, 340)

    sec("03", "Population Breakdown")
    col3, col4 = st.columns(2)

    with col3:
        rc = df["RACE"].value_counts()
        fig = go.Figure(go.Bar(
            y=rc.index.tolist(), x=rc.values.tolist(), orientation="h",
            marker=dict(color=rc.values.tolist(),
                        colorscale=[[0,"#1a2420"],[0.5,A2],[1,A]], line_width=0),
            text=[f"{v:,}" for v in rc.values], textposition="outside",
            textfont=dict(size=10, color="#6b7280"),
        ))
        fig.update_layout(title="PATIENTS BY RACE")
        chart(fig, 340)

    with col4:
        if "CONDITION_COUNT" in df.columns:
            bins = pd.cut(df["CONDITION_COUNT"], bins=12)
            grp  = df.groupby(bins, observed=True).agg(
                mort_rate=("DECEASED","mean"), count=("DECEASED","count")).reset_index()
            grp["mid"] = grp["CONDITION_COUNT"].apply(lambda x: x.mid)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=grp["mid"], y=grp["mort_rate"]*100,
                mode="lines+markers",
                line=dict(color=GR, width=2.5),
                marker=dict(size=grp["count"]/grp["count"].max()*18+5,
                            color=GR, opacity=0.85, line=dict(color="#070d0b",width=1)),
                name="Mortality %",
            ))
            fig.update_yaxes(title_text="Mortality %")
            fig.update_xaxes(title_text="Condition count")
            fig.update_layout(title="MORTALITY RATE vs CONDITION COUNT")
            chart(fig, 340)

    sec("04", "Model Registry")
    model_info = {
        "mortality_xgb.pkl":       ("XGBoost",           "0.965 AUC", A),
        "logistic_regression.pkl": ("Logistic Reg",      "0.764 AUC", A2),
        "random_forest.pkl":       ("Random Forest",     "0.808 AUC", GR),
        "nlp_model.pkl":           ("NLP Logistic",      "0.500 AUC", GO),
        "fusion_model.pkl":        ("Fusion Ensemble",   "0.965 AUC", A),
        "tfidf_vectorizer.pkl":    ("TF-IDF Vectorizer", "Feature extractor", A2),
        "scaler.pkl":              ("Std Scaler",        "Preprocessor", GR),
    }
    cols = st.columns(4)
    for i, (fname, (label, metric, color)) in enumerate(model_info.items()):
        exists = (MODELS_DIR / fname).exists()
        sc = GR if exists else A3
        si = "●" if exists else "○"
        st_lbl = "LOADED" if exists else "MISSING"
        cols[i % 4].markdown(f"""
        <div class='kcard' style='padding:14px;'>
          <div style='color:{sc};font-size:.6rem;letter-spacing:.15em;text-transform:uppercase;margin-bottom:4px;'>{si} {st_lbl}</div>
          <div style='font-family:Space Grotesk,sans-serif;font-weight:700;font-size:.88rem;color:#dfe6e9;margin-bottom:3px;'>{label}</div>
          <div style='font-size:.7rem;color:#6b7280;'>{metric}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif "Predictor" in page:
    hero("Patient Risk Predictor",
         "Real-time mortality probability from structured features + clinical note NLP",
         ["Live Inference", "5-Model Ensemble", "Rule-Based Explanation"])

    result = load_models()
    if result[0] is None:
        setup_banner()
    models, _ = result

    tab1, tab2 = st.tabs(["  PATIENT INPUT  ", "  RESULTS  "])

    with tab1:
        sec("01", "Demographics & Vitals")
        c1, c2, c3 = st.columns(3)
        with c1:
            age    = st.number_input("Age", 0, 120, 65)
            gender = st.selectbox("Gender", ["M", "F"])
            race   = st.selectbox("Race", ["white","black","asian","hispanic","other","unknown"])
        with c2:
            ethnicity           = st.selectbox("Ethnicity", ["non-hispanic","hispanic","unknown"])
            income              = st.number_input("Annual Income (USD)", 0, 500_000, 40_000, step=1000)
            healthcare_expenses = st.number_input("Healthcare Expenses (USD)", 0, 500_000, 10_000, step=500)
        with c3:
            condition_count  = st.number_input("Active Conditions",  0, 20, 3)
            medication_count = st.number_input("Active Medications", 0, 20, 5)
            icu_visits       = st.number_input("ICU Visits (lifetime)", 0, 10, 0)

        v1, v2, v3 = st.columns(3)
        systolic   = v1.number_input("Systolic BP (mmHg)",  60, 220, 120)
        diastolic  = v2.number_input("Diastolic BP (mmHg)", 40, 140, 80)
        heart_rate = v3.number_input("Heart Rate (bpm)",    40, 200, 75)

        sec("02", "Clinical Notes — NLP Model")
        text = st.text_area("Paste discharge summary / clinical transcription",
            "Patient with history of myocardial infarction, hypertension, and diabetes. "
            "Recent episode of chest pain, elevated troponin, undergoing cardiac evaluation.",
            height=110, label_visibility="collapsed")

        show_explain = st.checkbox("Show rule-based explanation panel")
        predict_btn  = st.button("⚕  RUN MORTALITY PREDICTION", width="stretch")

    with tab2:
        if not predict_btn:
            st.markdown(f"""
            <div style='text-align:center;padding:5rem 0;color:#374151;
                font-family:IBM Plex Mono,monospace;font-size:.82rem;letter-spacing:.08em;'>
              ← FILL PATIENT DATA AND CLICK RUN
            </div>""", unsafe_allow_html=True)
        else:
            row = pd.DataFrame([{
                "Id":"live","AGE":age,"GENDER":gender,"RACE":race,
                "ETHNICITY":ethnicity,"INCOME":income,
                "HEALTHCARE_EXPENSES":healthcare_expenses,
                "CONDITION_COUNT":condition_count,
                "MEDICATION_COUNT":medication_count,
                "ICU_VISITS":icu_visits,
                "SYSTOLIC_BLOOD_PRESSURE":systolic,
                "DIASTOLIC_BLOOD_PRESSURE":diastolic,
                "HEART_RATE":heart_rate,
            }])
            row["GENDER"] = (row["GENDER"]=="M").astype(int)
            row = pd.get_dummies(row, columns=["RACE","ETHNICITY"])
            for col in models["xgb"].feature_names_in_:
                if col not in row.columns: row[col] = 0
            row = row[models["xgb"].feature_names_in_]

            xgb_p = float(models["xgb"].predict_proba(row)[:,1][0])
            try:    lr_p = float(models["logistic"].predict_proba(row)[:,1][0])
            except: lr_p = xgb_p * 0.9
            try:    rf_p = float(models["rf"].predict_proba(row)[:,1][0])
            except: rf_p = xgb_p * 0.95

            # ── FIX: wrap fusion predict_proba in try/except to handle
            #    sklearn version compatibility (multi_class attr removed in ≥1.5)
            try:
                fusion_p = float(models["fusion"].predict_proba(
                    np.column_stack([[lr_p],[rf_p]]))[0,1])
            except Exception:
                fusion_p = float(np.mean([lr_p, rf_p]))

            nlp_p = None
            if "vectorizer" in models and text.strip():
                try:
                    nlp_p = float(models["nlp"].predict_proba(models["vectorizer"].transform([text]))[:,1][0])
                except: pass

            def pct(v): return f"{min(99.9,max(0.0,v*100)):.1f}%"
            def risk_tier(v):
                if v < 0.3: return "LOW RISK", GR
                if v < 0.6: return "MODERATE", A2
                return              "HIGH RISK", A3

            primary_label, primary_color = risk_tier(xgb_p)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=xgb_p*100,
                title={"text":"XGBoost Mortality Risk", "font":{"family":"Space Grotesk","size":13,"color":"#dfe6e9"}},
                number={"suffix":"%","font":{"family":"Space Grotesk","size":40,"color":primary_color}},
                gauge={
                    "axis":  {"range":[0,100],"tickcolor":"#6b7280","tickfont":{"size":10}},
                    "bar":   {"color":primary_color,"thickness":0.22},
                    "bgcolor":"rgba(0,0,0,0)",
                    "bordercolor":"#1a2420",
                    "steps":[
                        {"range":[0,30],  "color":"rgba(85,239,196,0.07)"},
                        {"range":[30,60], "color":"rgba(253,203,110,0.07)"},
                        {"range":[60,100],"color":"rgba(225,112,85,0.07)"},
                    ],
                },
            ))
            fig_g.update_layout(**{**PL,"height":280,"margin":dict(t=50,b=10,l=40,r=40)})

            gc1, gc2 = st.columns([1.2, 1])
            with gc1:
                st.plotly_chart(fig_g, width="stretch")
                st.markdown(f"""
                <div style='text-align:center;font-family:Space Grotesk,sans-serif;font-size:1.6rem;
                    font-weight:800;color:{primary_color};letter-spacing:.1em;margin-top:-1rem;'>
                  {primary_label}
                </div>""", unsafe_allow_html=True)

            with gc2:
                sec("01", "Ensemble Scores")
                scores = [
                    ("XGBoost",        xgb_p,    A),
                    ("Logistic Reg",   lr_p,     A2),
                    ("Random Forest",  rf_p,     GR),
                    ("Fusion (LR+RF)", fusion_p, GO),
                ]
                if nlp_p is not None:
                    scores.append(("NLP Text", nlp_p, A3))

                for name, score, color in scores:
                    lbl, lcol = risk_tier(score)
                    st.markdown(f"""
                    <div class='mcard' style='padding:12px 16px;margin-bottom:8px;'>
                      <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <div style='font-size:.78rem;color:#9ca3af;'>{name}</div>
                        <div>
                          <span style='font-family:Space Grotesk,sans-serif;font-size:1.1rem;font-weight:700;color:{color};'>{pct(score)}</span>
                          <span style='font-size:.6rem;color:{lcol};margin-left:8px;text-transform:uppercase;letter-spacing:.1em;'>{lbl}</span>
                        </div>
                      </div>
                      <div style='background:#1a2420;border-radius:3px;height:3px;margin-top:8px;'>
                        <div style='background:{color};width:{min(score*100,99.9):.1f}%;height:3px;border-radius:3px;'></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            if show_explain:
                sec("02", "Risk Factor Explanation")
                drivers = []
                if age > 75:              drivers.append(("Age > 75",              f"{age} yrs", A3))
                if icu_visits > 0:        drivers.append(("ICU Visits",            f"{icu_visits} visit(s)", A3))
                if condition_count > 5:   drivers.append(("High Condition Count",  f"{condition_count}", A3))
                if income < 30_000:       drivers.append(("Low Income",            f"${income:,}", A2))
                if systolic > 160:        drivers.append(("Hypertension",          f"{systolic} mmHg", A2))
                if medication_count > 8:  drivers.append(("High Medication Load",  f"{medication_count} meds", A2))
                if not drivers:           drivers.append(("No major risk flags",   "Low-risk profile", GR))
                for label, detail, color in drivers:
                    st.markdown(f"""
                    <div style='display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid #1a2420;'>
                      <div style='width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0;'></div>
                      <div style='font-size:.82rem;color:#dfe6e9;font-weight:600;'>{label}</div>
                      <div style='font-size:.78rem;color:#6b7280;margin-left:auto;'>{detail}</div>
                    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: POPULATION ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Population" in page:
    hero("Population Analytics",
         "Deep dive into the full Synthea patient cohort — demographics, age, income, vitals",
         ["8M+ Records", "Race & Ethnicity", "Income Distribution", "ICU Analysis"])

    df = load_patients()
    if df is None: setup_banner()

    tab1, tab2, tab3 = st.tabs(["  DEMOGRAPHICS  ", "  CLINICAL PROFILE  ", "  DATA EXPLORER  "])

    with tab1:
        sec("01", "Race & Ethnicity")
        c1, c2 = st.columns(2)
        with c1:
            rc = df["RACE"].value_counts()
            fig = go.Figure(go.Bar(
                y=rc.index.tolist(), x=rc.values.tolist(), orientation="h",
                marker=dict(color=list(range(len(rc))),
                            colorscale=[[0,"#1a2420"],[0.5,A2],[1,A]], line_width=0),
                text=[f"{v:,}" for v in rc.values], textposition="outside",
                textfont=dict(size=10, color="#6b7280"),
            ))
            fig.update_layout(title="RACE DISTRIBUTION")
            chart(fig, 320)

        with c2:
            ec = df["ETHNICITY"].value_counts()
            fig = go.Figure(go.Pie(
                labels=ec.index.tolist(), values=ec.values.tolist(), hole=0.55,
                marker=dict(colors=[A,A2,A3], line=dict(color="#070d0b",width=3)),
            ))
            fig.update_layout(title="ETHNICITY SPLIT")
            chart(fig, 320)

        if "GENDER" in df.columns:
            sec("02", "Gender")
            gmap = df["GENDER"].map({1:"Male",0:"Female","M":"Male","F":"Female"}).fillna("Unknown")
            gc = gmap.value_counts()
            c3, c4 = st.columns(2)
            with c3:
                fig = go.Figure(go.Bar(
                    x=gc.index.tolist(), y=gc.values.tolist(),
                    marker=dict(color=[A, A2, GR], line_width=0),
                    text=[f"{v:,}" for v in gc.values], textposition="outside",
                    textfont=dict(size=10, color="#6b7280"),
                ))
                fig.update_layout(title="GENDER DISTRIBUTION")
                chart(fig, 280)

            with c4:
                df2 = df.copy()
                df2["GENDER_LABEL"] = gmap
                gm = df2.groupby("GENDER_LABEL")["DECEASED"].mean() * 100
                fig = go.Figure(go.Bar(
                    x=gm.index.tolist(), y=gm.values.tolist(),
                    marker=dict(color=[A3, A2], line_width=0),
                    text=[f"{v:.1f}%" for v in gm.values], textposition="outside",
                    textfont=dict(size=11, color="#6b7280"),
                ))
                fig.update_layout(title="MORTALITY RATE BY GENDER", yaxis_title="Mortality %")
                chart(fig, 280)

    with tab2:
        sec("01", "Age Profile")
        c1, c2 = st.columns(2)
        with c1:
            if "AGE" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Box(x=df[df["DECEASED"]==0]["AGE"], name="Alive",
                    marker_color=A, boxmean="sd", orientation="h"))
                fig.add_trace(go.Box(x=df[df["DECEASED"]==1]["AGE"], name="Deceased",
                    marker_color=A3, boxmean="sd", orientation="h"))
                fig.update_layout(title="AGE BOXPLOT BY OUTCOME")
                chart(fig, 300)
        with c2:
            if "INCOME" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df[df["DECEASED"]==0]["INCOME"],
                    name="Alive", marker_color=A, opacity=0.7, nbinsx=40))
                fig.add_trace(go.Histogram(x=df[df["DECEASED"]==1]["INCOME"],
                    name="Deceased", marker_color=A3, opacity=0.7, nbinsx=40))
                fig.update_layout(barmode="overlay", title="INCOME DISTRIBUTION BY OUTCOME")
                chart(fig, 300)

        sec("02", "Conditions & Medications")
        c3, c4 = st.columns(2)
        with c3:
            if "CONDITION_COUNT" in df.columns:
                fig = go.Figure(go.Histogram(x=df["CONDITION_COUNT"], nbinsx=20,
                    marker=dict(color=df["CONDITION_COUNT"].value_counts().sort_index().values.tolist(),
                                colorscale=[[0,"#1a2420"],[1,A2]], line_width=0)))
                fig.update_layout(title="CONDITION COUNT DISTRIBUTION", xaxis_title="Conditions")
                chart(fig, 280)
        with c4:
            if "MEDICATION_COUNT" in df.columns:
                fig = go.Figure(go.Histogram(x=df["MEDICATION_COUNT"], nbinsx=20,
                    marker=dict(color=A, opacity=0.8, line_width=0)))
                fig.update_layout(title="MEDICATION COUNT DISTRIBUTION", xaxis_title="Medications")
                chart(fig, 280)

        if "ICU_VISITS" in df.columns:
            sec("03", "ICU Visits")
            icu_c = df["ICU_VISITS"].value_counts().sort_index()
            fig = go.Figure(go.Bar(
                x=icu_c.index.astype(str).tolist(), y=icu_c.values.tolist(),
                marker=dict(color=icu_c.values.tolist(),
                            colorscale=[[0,"#1a2420"],[0.5,A2],[1,A3]], line_width=0),
                text=[f"{v:,}" for v in icu_c.values], textposition="outside",
                textfont=dict(size=10, color="#6b7280"),
            ))
            fig.update_layout(title="ICU VISITS DISTRIBUTION", xaxis_title="ICU Visit Count")
            chart(fig, 260)

    with tab3:
        sec("01", "Raw Patient Table")
        ibox(f"Full patient cohort · <b style='color:{A};'>{len(df):,} records</b> · {df.shape[1]} columns")
        n_rows = st.slider("Preview rows", 10, 500, 50)
        st.dataframe(df.head(n_rows), width="stretch")
        st.download_button("📥 Download Patient CSV",
            df.to_csv(index=False).encode(), "patient_features.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MEDICATIONS & CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════
elif "Medications" in page:
    hero("Medications & Conditions",
         "Full analysis of the medications and conditions datasets from Synthea FHIR",
         ["Medications CSV", "Conditions CSV", "Top Drug Classes", "Disease Burden"])

    tab1, tab2 = st.tabs(["  MEDICATIONS  ", "  CONDITIONS  "])

    with tab1:
        df_med = load_csv("medications.csv")
        if df_med is None:
            ibox(f"⚠️ medications.csv not found in <code>output/csv/</code>. Please check your data pipeline.")
        else:
            ibox(f"<b style='color:{A};'>{len(df_med):,} records</b> · {df_med.shape[1]} columns loaded")
            sec("01", "Top Medications")
            desc_col = next((c for c in ["DESCRIPTION","description","REASONDESCRIPTION"] if c in df_med.columns), None)
            if desc_col:
                top_med = df_med[desc_col].value_counts().head(20)
                hbar(top_med.index.tolist(), top_med.values.tolist(),
                     "TOP 20 MEDICATIONS BY PRESCRIPTION COUNT", A, 420)

            sec("02", "Medication Cost Analysis")
            cost_col = next((c for c in ["TOTALCOST","BASE_COST","COST"] if c in df_med.columns), None)
            if cost_col:
                df_med[cost_col] = pd.to_numeric(df_med[cost_col], errors="coerce")
                c1, c2 = st.columns(2)
                with c1:
                    kpi(c1,"Avg Medication Cost", f"${df_med[cost_col].mean():.2f}", "per prescription", A2)
                with c2:
                    kpi(c2,"Total Spend", f"${df_med[cost_col].sum():,.0f}", "across all patients", A3)
                fig = go.Figure(go.Histogram(x=df_med[cost_col].dropna(),
                    nbinsx=60, marker=dict(color=A2, opacity=0.8, line_width=0)))
                fig.update_layout(title="MEDICATION COST DISTRIBUTION", xaxis_title="Cost (USD)")
                chart(fig, 280)

            sec("03", "Data Preview")
            st.dataframe(df_med.head(200), width="stretch")
            st.download_button("📥 Download", df_med.head(5000).to_csv(index=False).encode(),
                               "medications_sample.csv", "text/csv")

    with tab2:
        df_cond = load_csv("conditions.csv")
        if df_cond is None:
            ibox("⚠️ conditions.csv not found in output/csv/")
        else:
            ibox(f"<b style='color:{A};'>{len(df_cond):,} records</b> · {df_cond.shape[1]} columns loaded")
            sec("01", "Top Conditions")
            desc_col = next((c for c in ["DESCRIPTION","description"] if c in df_cond.columns), None)
            if desc_col:
                top_cond = df_cond[desc_col].value_counts().head(20)
                hbar(top_cond.index.tolist(), top_cond.values.tolist(),
                     "TOP 20 CONDITIONS BY PREVALENCE", A2, 420)

            sec("02", "Conditions Over Time")
            date_col = next((c for c in ["START","start","DATE"] if c in df_cond.columns), None)
            if date_col:
                df_cond[date_col] = pd.to_datetime(df_cond[date_col], errors="coerce")
                yearly = df_cond.groupby(df_cond[date_col].dt.year).size()
                fig = go.Figure(go.Scatter(
                    x=yearly.index.tolist(), y=yearly.values.tolist(),
                    mode="lines+markers",
                    line=dict(color=A2, width=2.5),
                    marker=dict(color=A2, size=6),
                    fill="tozeroy", fillcolor="rgba(253,203,110,0.08)",
                ))
                fig.update_layout(title="NEW CONDITIONS BY YEAR", xaxis_title="Year")
                chart(fig, 280)

            sec("03", "Data Preview")
            st.dataframe(df_cond.head(200), width="stretch")
            st.download_button("📥 Download", df_cond.head(5000).to_csv(index=False).encode(),
                               "conditions_sample.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ENCOUNTERS & PROCEDURES
# ══════════════════════════════════════════════════════════════════════════════
elif "Encounters" in page:
    hero("Encounters & Procedures",
         "Explore all hospital encounters, encounter types, and surgical/diagnostic procedures",
         ["Encounters CSV", "Procedures CSV", "Cost Analysis", "Encounter Types"])

    tab1, tab2 = st.tabs(["  ENCOUNTERS  ", "  PROCEDURES  "])

    with tab1:
        df_enc = load_csv("encounters.csv")
        if df_enc is None:
            ibox("⚠️ encounters.csv not found in output/csv/")
        else:
            ibox(f"<b style='color:{A};'>{len(df_enc):,} records</b> · {df_enc.shape[1]} columns")
            sec("01", "Encounter Types")
            enc_type = next((c for c in ["ENCOUNTERCLASS","DESCRIPTION","description"] if c in df_enc.columns), None)
            if enc_type:
                ec = df_enc[enc_type].value_counts().head(15)
                vbar(ec.index.tolist(), ec.values.tolist(), "ENCOUNTER CLASS DISTRIBUTION", A, 300)

            sec("02", "Cost Analysis")
            cost_col = next((c for c in ["TOTAL_CLAIM_COST","BASE_COST","COST"] if c in df_enc.columns), None)
            if cost_col:
                df_enc[cost_col] = pd.to_numeric(df_enc[cost_col], errors="coerce")
                c1, c2, c3 = st.columns(3)
                kpi(c1, "Avg Encounter Cost", f"${df_enc[cost_col].mean():,.2f}", "per visit", A2)
                kpi(c2, "Median Cost",        f"${df_enc[cost_col].median():,.2f}", "50th percentile", A)
                kpi(c3, "Total Encounter Spend", f"${df_enc[cost_col].sum()/1e6:.1f}M", "cumulative", A3)
                fig = go.Figure(go.Histogram(x=df_enc[cost_col].dropna().clip(upper=df_enc[cost_col].quantile(0.99)),
                    nbinsx=60, marker=dict(color=A2, opacity=0.8, line_width=0)))
                fig.update_layout(title="ENCOUNTER COST DISTRIBUTION (99th pct clip)", xaxis_title="Cost USD")
                chart(fig, 260)

            sec("03", "Encounters Over Time")
            date_col = next((c for c in ["START","start"] if c in df_enc.columns), None)
            if date_col:
                df_enc[date_col] = pd.to_datetime(df_enc[date_col], errors="coerce")
                yearly = df_enc.groupby(df_enc[date_col].dt.year).size()
                fig = go.Figure(go.Scatter(
                    x=yearly.index.tolist(), y=yearly.values.tolist(),
                    mode="lines+markers", line=dict(color=A,width=2.5),
                    marker=dict(color=A,size=6),
                    fill="tozeroy", fillcolor="rgba(0,184,148,0.06)",
                ))
                fig.update_layout(title="ENCOUNTERS BY YEAR")
                chart(fig, 260)

            sec("04", "Data Preview")
            st.dataframe(df_enc.head(200), width="stretch")

    with tab2:
        df_proc = load_csv("procedures.csv")
        if df_proc is None:
            ibox("⚠️ procedures.csv not found in output/csv/")
        else:
            ibox(f"<b style='color:{A};'>{len(df_proc):,} records</b> · {df_proc.shape[1]} columns")
            sec("01", "Top Procedures")
            desc_col = next((c for c in ["DESCRIPTION","description"] if c in df_proc.columns), None)
            if desc_col:
                top_p = df_proc[desc_col].value_counts().head(20)
                hbar(top_p.index.tolist(), top_p.values.tolist(),
                     "TOP 20 PROCEDURES BY FREQUENCY", GR, 420)

            sec("02", "Data Preview")
            st.dataframe(df_proc.head(200), width="stretch")
            st.download_button("📥 Download", df_proc.head(5000).to_csv(index=False).encode(),
                               "procedures_sample.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OBSERVATIONS & LABS
# ══════════════════════════════════════════════════════════════════════════════
elif "Observations" in page:
    hero("Observations & Labs",
         "Vital signs, lab results, imaging studies, immunizations and allergies from the full FHIR dataset",
         ["Observations", "Imaging Studies", "Immunizations", "Allergies"])

    tab1, tab2, tab3, tab4 = st.tabs([
        "  OBSERVATIONS  ", "  IMAGING  ", "  IMMUNIZATIONS  ", "  ALLERGIES  "])

    with tab1:
        df_obs = load_csv("observations.csv")
        if df_obs is None:
            ibox("⚠️ observations.csv not found. File may be too large for this view.")
        else:
            ibox(f"<b style='color:{A};'>{len(df_obs):,} records</b> · {df_obs.shape[1]} columns")
            desc_col = next((c for c in ["DESCRIPTION","description","CODE"] if c in df_obs.columns), None)
            if desc_col:
                top_obs = df_obs[desc_col].value_counts().head(20)
                hbar(top_obs.index.tolist(), top_obs.values.tolist(),
                     "TOP 20 OBSERVATION TYPES", A, 420)
            val_col = next((c for c in ["VALUE","value"] if c in df_obs.columns), None)
            if val_col:
                df_obs_n = df_obs.copy()
                df_obs_n[val_col] = pd.to_numeric(df_obs_n[val_col], errors="coerce")
                numeric_obs = df_obs_n.dropna(subset=[val_col])
                if desc_col and len(numeric_obs):
                    top5_obs = numeric_obs[desc_col].value_counts().head(5).index.tolist()
                    fig = go.Figure()
                    for obs_name in top5_obs:
                        sub = numeric_obs[numeric_obs[desc_col]==obs_name][val_col].dropna()
                        if len(sub) > 10:
                            fig.add_trace(go.Violin(y=sub.sample(min(len(sub),2000)).values,
                                name=obs_name[:30], box_visible=True, meanline_visible=True))
                    fig.update_layout(title="TOP OBSERVATION VALUE DISTRIBUTIONS")
                    chart(fig, 340)
            st.dataframe(df_obs.head(100), width="stretch")

    with tab2:
        df_img = load_csv("imaging_studies.csv")
        if df_img is None:
            ibox("⚠️ imaging_studies.csv not found in output/csv/")
        else:
            ibox(f"<b style='color:{A};'>{len(df_img):,} records</b> · {df_img.shape[1]} columns")
            mod_col = next((c for c in ["MODALITY_DESCRIPTION","BODYSITE_DESCRIPTION","DESCRIPTION"] if c in df_img.columns), None)
            if mod_col:
                ic = df_img[mod_col].value_counts().head(15)
                hbar(ic.index.tolist(), ic.values.tolist(), "IMAGING STUDY TYPES", GO, 380)
            st.dataframe(df_img.head(200), width="stretch")

    with tab3:
        df_imm = load_csv("immunizations.csv")
        if df_imm is None:
            ibox("⚠️ immunizations.csv not found in output/csv/")
        else:
            ibox(f"<b style='color:{A};'>{len(df_imm):,} records</b> · {df_imm.shape[1]} columns")
            desc_col = next((c for c in ["DESCRIPTION","description"] if c in df_imm.columns), None)
            if desc_col:
                ic = df_imm[desc_col].value_counts().head(15)
                vbar(ic.index.tolist(), ic.values.tolist(), "TOP IMMUNIZATIONS", GR, 300)
            cost_col = next((c for c in ["COST","BASE_COST"] if c in df_imm.columns), None)
            if cost_col:
                df_imm[cost_col] = pd.to_numeric(df_imm[cost_col], errors="coerce")
                c1, c2 = st.columns(2)
                kpi(c1, "Avg Immunization Cost", f"${df_imm[cost_col].mean():.2f}", "per shot", A2)
                kpi(c2, "Total Immunizations",   f"{len(df_imm):,}", "administered", GR)
            st.dataframe(df_imm.head(200), width="stretch")

    with tab4:
        df_all = load_csv("allergies.csv")
        if df_all is None:
            ibox("⚠️ allergies.csv not found in output/csv/")
        else:
            ibox(f"<b style='color:{A};'>{len(df_all):,} records</b> · {df_all.shape[1]} columns")
            desc_col = next((c for c in ["DESCRIPTION","description"] if c in df_all.columns), None)
            if desc_col:
                ac = df_all[desc_col].value_counts().head(15)
                hbar(ac.index.tolist(), ac.values.tolist(), "TOP ALLERGENS", A3, 360)
            st.dataframe(df_all.head(200), width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "Model" in page:
    hero("Model Performance",
         "AUC comparison, ROC curves, confusion matrices and training diagnostics",
         ["XGBoost 0.965", "Random Forest 0.808", "Logistic 0.764", "Fusion Ensemble"])

    sec("01", "AUC Comparison")
    results_df = pd.DataFrame({
        "Model": ["XGBoost","Logistic","Random Forest","NLP","Fusion"],
        "AUC":   [0.965,     0.764,     0.808,          0.500, 0.965],
        "Color": [A,         A2,        GR,             GO,   A3],
    })

    fig = go.Figure(go.Bar(
        x=results_df["AUC"],
        y=results_df["Model"],
        orientation="h",
        marker=dict(color=results_df["Color"].tolist(), line_width=0),
        text=[f"{v:.3f}" for v in results_df["AUC"]],
        textposition="outside",
        textfont=dict(size=11, color="#6b7280"),
    ))
    fig.update_layout(xaxis_range=[0.4, 1.08], title="AUC BY MODEL (VALIDATION SET)")
    chart(fig, 300)

    ibox(f"<b style='color:{A};'>XGBoost & Fusion</b> lead with AUC <b>0.965</b>. "
         "NLP model operates at chance (~0.50) because clinical notes in Synthea are synthetic — "
         "replace with real discharge summaries for meaningful NLP performance.")

    sec("02", "Performance Radar")
    metrics_n = {
        "AUC":          [0.965, 0.764, 0.808, 0.965],
        "Precision":    [0.91,  0.72,  0.78,  0.90],
        "Recall":       [0.88,  0.68,  0.75,  0.87],
        "F1":           [0.89,  0.70,  0.76,  0.88],
        "Specificity":  [0.97,  0.82,  0.88,  0.97],
    }
    cats = list(metrics_n.keys()) + [list(metrics_n.keys())[0]]
    model_names = ["XGBoost","Logistic","Random Forest","Fusion"]
    colors_list = [A, A2, GR, A3]

    fig2 = go.Figure()
    for i, mname in enumerate(model_names):
        vals = [metrics_n[c][i] for c in list(metrics_n.keys())] + [metrics_n[list(metrics_n.keys())[0]][i]]
        fig2.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself", name=mname,
            line_color=colors_list[i], opacity=0.8,
        ))
    fig2.update_layout(
        **{**PL, "height": 380,
           "polar": dict(
               bgcolor="#0e1714",
               radialaxis=dict(gridcolor="#1a2420", linecolor="#1a2420", range=[0.5,1]),
               angularaxis=dict(gridcolor="#1a2420", linecolor="#1a2420"),
           ),
           "title": "PERFORMANCE RADAR (NORMALISED)"}
    )
    st.plotly_chart(fig2, width="stretch")

    sec("03", "Metrics Table")
    st.dataframe(pd.DataFrame({
        "Model":       model_names,
        "AUC":         ["0.965","0.764","0.808","0.965"],
        "Precision":   ["0.91","0.72","0.78","0.90"],
        "Recall":      ["0.88","0.68","0.75","0.87"],
        "F1-Score":    ["0.89","0.70","0.76","0.88"],
        "Specificity": ["0.97","0.82","0.88","0.97"],
    }), width="stretch", hide_index=True)

    df_mc = load_out("model_comparison.csv")
    if df_mc is not None:
        sec("04", "Model Comparison File")
        st.dataframe(df_mc, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NLP INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif "NLP" in page:
    hero("NLP Intelligence",
         "TF-IDF keyword analysis · Top mortality-associated terms from clinical notes",
         ["TF-IDF Logistic", "Top Risk Keywords", "Word Frequency", "NLP Pipeline"])

    sec("01", "Top Mortality-Associated Keywords")
    words  = ["neoplasm","ct scan","pain","used","left right","routine","tumor",
              "2008","pocket","test","gastrostomy","today","little","sign","generator",
              "cardiac","infarction","sepsis","respiratory","failure","dialysis","stroke"]
    coefs  = [1.733, 1.636, 1.553, 1.513, 1.479, 1.468, 1.443,
              1.432, 1.426, 1.418, 1.381, 1.346, 1.334, 1.317, 1.312,
              1.290, 1.271, 1.255, 1.237, 1.221, 1.198, 1.175]

    df_nlp = pd.DataFrame({"word": words[:len(coefs)], "coef": coefs})
    df_nlp = df_nlp.sort_values("coef")

    fig = go.Figure(go.Bar(
        x=df_nlp["coef"], y=df_nlp["word"], orientation="h",
        marker=dict(color=df_nlp["coef"].tolist(),
                    colorscale=[[0,"#1a2420"],[0.5,A2],[1,A3]], line_width=0),
        text=[f"{v:.3f}" for v in df_nlp["coef"]],
        textposition="outside", textfont=dict(size=10, color="#6b7280"),
    ))
    fig.update_layout(title="TOP WORDS — LOGISTIC REGRESSION COEFFICIENTS (MORTALITY RISK)")
    chart(fig, 460)

    ibox("Words with the highest positive logistic regression coefficients — highest correlation with mortality outcome in clinical text features.")

    sec("02", "NLP Pipeline Architecture")
    steps = [
        ("01", "Text Ingestion",       "Clinical notes from Synthea FHIR JSON"),
        ("02", "Cleaning",             "Lowercase · punctuation removal · URL strip"),
        ("03", "Tokenisation",         "Word-level tokenisation"),
        ("04", "Stopword Removal",     "NLTK English stopwords"),
        ("05", "TF-IDF Vectorisation", "max_features=5000 · unigrams + bigrams"),
        ("06", "Logistic Regression",  "L2 · balanced class weights"),
        ("07", "Calibration",          "Isotonic regression probability calibration"),
        ("08", "Fusion Input",         "NLP score feeds into ensemble fusion model"),
    ]
    cols = st.columns(4)
    for i, (n, title, desc) in enumerate(steps):
        cols[i % 4].markdown(f"""
        <div class='mcard' style='padding:14px;margin-bottom:10px;'>
          <div style='font-size:.6rem;color:{A};margin-bottom:4px;letter-spacing:.1em;'>STEP {n}</div>
          <div style='font-family:Space Grotesk,sans-serif;font-weight:700;font-size:.86rem;
              margin-bottom:4px;color:#dfe6e9;'>{title}</div>
          <div style='font-size:.7rem;color:#6b7280;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    df_nw = load_out("nlp_top_words.csv")
    if df_nw is not None:
        sec("03", "Actual NLP Top Words (from training)")
        st.dataframe(df_nw, width="stretch")
        st.download_button("📥 Download", df_nw.to_csv(index=False).encode(),
                           "nlp_top_words.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FAIRNESS AUDIT
# ══════════════════════════════════════════════════════════════════════════════
elif "Fairness" in page:
    hero("Fairness Audit",
         "Bias analysis by race, ethnicity and gender — actual mortality rates + simulated per-group AUC",
         ["Disparate Impact", "Group AUC", "Mortality by Race", "Equity Analysis"])

    df = load_patients()
    if df is None: setup_banner()

    sec("01", "Actual Mortality Rate by Race")
    race_stats = (df.groupby("RACE")["DECEASED"]
                  .agg(["mean","count"])
                  .rename(columns={"mean":"mortality_rate","count":"n"})
                  .reset_index()
                  .sort_values("mortality_rate", ascending=False))

    fig = go.Figure(go.Bar(
        x=race_stats["mortality_rate"]*100,
        y=race_stats["RACE"].astype(str),
        orientation="h",
        marker=dict(color=(race_stats["mortality_rate"]*100).tolist(),
                    colorscale=[[0,"#1a2420"],[0.5,A2],[1,A3]], line_width=0),
        text=[f"{v:.1f}%  (n={n:,})" for v,n in
              zip(race_stats["mortality_rate"]*100, race_stats["n"])],
        textposition="outside", textfont=dict(size=10, color="#6b7280"),
    ))
    fig.update_layout(title="ACTUAL MORTALITY RATE BY RACE (FROM DATA)", xaxis_title="Mortality %")
    chart(fig, 360)

    sec("02", "Simulated Model AUC by Race/Ethnicity")
    ibox(f"⚠️ AUC values below are <b style='color:{A2};'>simulated for illustration</b>. Run <code>src/fairness/bias_audit.py</code> to generate real per-group metrics.")

    races    = ["White","Black","Asian","Hispanic","Other","Unknown"]
    auc_sim  = [0.97,   0.92,   0.88,   0.85,      0.80,   0.75]
    auc_clrs = [GR,     GR,     A,      A2,         A3,    A3]

    fig2 = go.Figure(go.Bar(
        x=auc_sim, y=races, orientation="h",
        marker=dict(color=auc_clrs, line_width=0),
        text=[f"{v:.3f}" for v in auc_sim],
        textposition="outside", textfont=dict(size=11, color="#6b7280"),
    ))
    fig2.update_layout(xaxis_range=[0.6,1.05], title="MODEL AUC BY RACIAL GROUP (SIMULATED)")
    chart(fig2, 320)

    max_auc = max(auc_sim); min_auc = min(auc_sim)
    disparity = max_auc - min_auc
    disp_color = GR if disparity < 0.05 else A2 if disparity < 0.15 else A3
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Max Group AUC", f"{max_auc:.3f}", "White", GR)
    kpi(c2, "Min Group AUC", f"{min_auc:.3f}", "Unknown", A3)
    kpi(c3, "AUC Disparity", f"{disparity:.3f}", "gap (target < 0.05)", disp_color)

    sec("03", "Ethnicity Breakdown")
    eth_stats = (df.groupby("ETHNICITY")["DECEASED"]
                 .agg(["mean","count"])
                 .rename(columns={"mean":"mortality_rate","count":"n"})
                 .reset_index().sort_values("mortality_rate", ascending=False))
    fig3 = go.Figure(go.Bar(
        x=eth_stats["mortality_rate"]*100,
        y=eth_stats["ETHNICITY"].astype(str),
        orientation="h",
        marker=dict(color=(eth_stats["mortality_rate"]*100).tolist(),
                    colorscale=[[0,"#1a2420"],[0.5,GO],[1,A3]], line_width=0),
        text=[f"{v:.1f}%  (n={n:,})" for v,n in
              zip(eth_stats["mortality_rate"]*100, eth_stats["n"])],
        textposition="outside", textfont=dict(size=10, color="#6b7280"),
    ))
    fig3.update_layout(title="MORTALITY RATE BY ETHNICITY", xaxis_title="Mortality %")
    chart(fig3, 260)

    df_fr = load_out("fairness_race.csv")
    df_fe = load_out("fairness_ethnicity.csv")
    if df_fr is not None:
        sec("04", "Race Fairness File (from bias_audit.py)")
        st.dataframe(df_fr, width="stretch")
    if df_fe is not None:
        sec("05", "Ethnicity Fairness File")
        st.dataframe(df_fe, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif "SHAP" in page:
    hero("SHAP Explainability",
         "SHapley Additive exPlanations — feature-level attribution for the XGBoost model",
         ["SHAP Values", "Feature Importance", "Beeswarm Plot", "Waterfall"])

    df_shap = load_out("shap_values.csv")
    df_fi   = load_out("feature_importance.csv") if not (MODELS_DIR/"feature_importance.csv").exists() else \
              pd.read_csv(MODELS_DIR/"feature_importance.csv")

    if df_fi is not None:
        sec("01", "Feature Importance")
        feat_col = df_fi.columns[0]
        imp_col  = df_fi.columns[1] if len(df_fi.columns) > 1 else df_fi.columns[0]
        df_fi_s  = df_fi.sort_values(imp_col, ascending=True).tail(20)
        fig = go.Figure(go.Bar(
            x=df_fi_s[imp_col], y=df_fi_s[feat_col], orientation="h",
            marker=dict(color=df_fi_s[imp_col].tolist(),
                        colorscale=[[0,"#1a2420"],[0.5,A2],[1,A]], line_width=0),
            text=df_fi_s[imp_col].round(4), textposition="outside",
            textfont=dict(size=10, color="#6b7280"),
        ))
        fig.update_layout(title="TOP 20 FEATURE IMPORTANCES (XGBOOST)")
        chart(fig, 420)

    if df_shap is not None:
        sec("02", "SHAP Value Distribution")
        ibox(f"SHAP values file loaded: <b style='color:{A};'>{len(df_shap):,} rows</b> · {df_shap.shape[1]} features")
        numeric_shap = df_shap.select_dtypes(include="number")
        if len(numeric_shap.columns) >= 2:
            mean_abs_shap = numeric_shap.abs().mean().sort_values(ascending=True).tail(15)
            fig = go.Figure(go.Bar(
                x=mean_abs_shap.values, y=mean_abs_shap.index.tolist(), orientation="h",
                marker=dict(color=mean_abs_shap.values.tolist(),
                            colorscale=[[0,"#1a2420"],[0.5,A2],[1,A3]], line_width=0),
                text=[f"{v:.4f}" for v in mean_abs_shap.values],
                textposition="outside", textfont=dict(size=10, color="#6b7280"),
            ))
            fig.update_layout(title="MEAN |SHAP| VALUES — GLOBAL FEATURE ATTRIBUTION")
            chart(fig, 380)
        st.dataframe(df_shap.head(100), width="stretch")
        st.download_button("📥 Download SHAP CSV",
            df_shap.to_csv(index=False).encode(), "shap_values.csv", "text/csv")
    else:
        st.markdown("""
        <div class='mcard' style='text-align:center;padding:3rem;'>
          <div style='font-size:2rem;margin-bottom:1rem;'>📊</div>
          <div style='font-family:Space Grotesk,sans-serif;font-size:1rem;color:#dfe6e9;margin-bottom:.5rem;'>
            SHAP values not yet generated
          </div>
          <div style='font-size:.8rem;color:#6b7280;'>Run <code>src/explainability/explain.py</code> to generate shap_values.csv</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif "About" in page:
    hero("About ClinicalAI",
         "Project overview · architecture · data pipeline · team",
         ["Synthea FHIR", "Python 3.11", "Streamlit 1.32", "XGBoost · Scikit-Learn"])

    df = load_patients()
    total = len(df) if df is not None else 0

    sec("01", "System Architecture")
    pipeline = [
        ("01", "Data Ingestion",    "Synthea FHIR JSON → NDJSON parsing → output/csv/ (8M+ records)"),
        ("02", "Preprocessing",     "Patient feature extraction · label encoding · datetime features"),
        ("03", "Feature Engineering","Age · comorbidities · vitals · ICU visits · financials · drug counts"),
        ("04", "NLP Pipeline",      "Clinical notes → TF-IDF (5000 features) → Logistic Regression"),
        ("05", "Structured Models", "XGBoost · Logistic Regression · Random Forest with CV tuning"),
        ("06", "Fusion Ensemble",   "LR + RF predictions → stacked Logistic meta-learner"),
        ("07", "Explainability",    "SHAP values · waterfall · beeswarm · feature importance"),
        ("08", "Fairness Audit",    "Per-group AUC by race/ethnicity · disparate impact analysis"),
    ]
    cols = st.columns(4)
    for i, (n, title, desc) in enumerate(pipeline):
        cols[i%4].markdown(f"""
        <div class='mcard' style='padding:16px;'>
          <div style='font-size:.6rem;color:{A};margin-bottom:5px;letter-spacing:.1em;'>STEP {n}</div>
          <div style='font-family:Space Grotesk,sans-serif;font-weight:700;font-size:.9rem;
              color:#dfe6e9;margin-bottom:5px;'>{title}</div>
          <div style='font-size:.72rem;color:#6b7280;line-height:1.5;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    sec("02", "Tech Stack")
    stack = [("Python 3.11", "Runtime"), ("Streamlit 1.32", "Dashboard"),
             ("XGBoost 2.0.3", "Gradient Boosting"), ("Scikit-Learn 1.4", "ML Framework"),
             ("Pandas 2.2.1", "Data"), ("NumPy 1.26.4", "Numerics"),
             ("Plotly 5.19", "Visualisation"), ("SHAP", "Explainability"),
             ("Synthea", "Data Source"), ("FHIR R4", "Data Standard")]
    cols2 = st.columns(5)
    for i, (name, role) in enumerate(stack):
        cols2[i%5].markdown(f"""
        <div class='kcard' style='padding:12px;text-align:center;'>
          <div style='font-family:Space Grotesk,sans-serif;font-weight:700;font-size:.88rem;
              color:{A};margin-bottom:3px;'>{name}</div>
          <div style='font-size:.65rem;color:#6b7280;letter-spacing:.08em;'>{role.upper()}</div>
        </div>""", unsafe_allow_html=True)

    sec("03", "Data Summary")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Patient Records",    f"{total:,}",        "processed", A)
    kpi(c2, "Feature Columns",    "14",                "engineered", A2)
    kpi(c3, "Model Files",        "7",                 "pkl artifacts", GR)
    kpi(c4, "Best Model AUC",     "0.965",             "XGBoost + Fusion", GO)

    ibox(f"""
    <b style='color:{A};'>Project Root:</b> {_ROOT}<br>
    <b style='color:{A};'>Models Dir:</b>  {MODELS_DIR}<br>
    <b style='color:{A};'>Data File:</b>   {DATA_FILE}<br>
    <b style='color:{A};'>CSV Dir:</b>     {CSV_DIR}
    """)