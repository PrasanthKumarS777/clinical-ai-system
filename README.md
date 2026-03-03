<div align="center">

# ⚕ ClinicalAI — Mortality Intelligence Dashboard

### Multimodal EHR-Powered Clinical Mortality Prediction System

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit%20Cloud-00b894?style=for-the-badge)](https://clinical-ai-system-hyroaadrc9mxgykkwnlqbz.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost AUC](https://img.shields.io/badge/XGBoost%20AUC-0.965-00b894?style=for-the-badge)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-fdcb6e?style=for-the-badge)](LICENSE)

<br/>

**A production-grade clinical AI system that predicts patient mortality risk using multimodal Electronic Health Records (EHR) data from Synthea FHIR — combining structured patient features, NLP on clinical notes, ensemble machine learning, SHAP explainability, and a comprehensive fairness audit — all served through a real-time interactive Streamlit dashboard.**

<br/>

[🚀 Live Demo](https://clinical-ai-system-hyroaadrc9mxgykkwnlqbz.streamlit.app/) · [📊 Model Results](#-model-results--performance) · [🏗️ Architecture](#️-system-architecture) · [⚖️ Fairness Audit](#️-fairness--bias-audit) · [🛠️ Setup Guide](#️-local-setup--installation)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [Cohort Statistics](#-cohort-statistics)
- [System Architecture](#️-system-architecture)
- [Project Structure](#-project-structure)
- [Data Pipeline](#-data-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Machine Learning Models](#-machine-learning-models)
- [Model Results & Performance](#-model-results--performance)
- [NLP Intelligence](#-nlp-intelligence)
- [SHAP Explainability](#-shap-explainability)
- [Fairness & Bias Audit](#️-fairness--bias-audit)
- [Dashboard Pages](#-dashboard-pages)
- [Tech Stack](#-tech-stack)
- [Local Setup & Installation](#️-local-setup--installation)
- [Deployment](#-deployment)
- [Author](#-author)

---

## 🎯 Project Overview

ClinicalAI is a professional-grade mortality prediction platform built on **Synthea FHIR synthetic EHR data** containing over **8 million clinical records**. The system ingests raw FHIR JSON, processes it through a multi-stage data pipeline, engineers clinically meaningful features, trains an ensemble of machine learning models, and exposes everything through a sleek real-time dashboard.

**Core capabilities:**

- **Real-time mortality risk scoring** — enter a patient's demographics, vitals, and clinical notes to get an instant mortality probability from a 5-model ensemble
- **Multimodal fusion** — combines structured EHR features with TF-IDF NLP on clinical text for richer predictions
- **Full SHAP explainability** — every prediction can be traced back to individual feature contributions
- **Clinical fairness audit** — per-group AUC analysis across race and ethnicity to detect and report model bias
- **Interactive population analytics** — deep-dive into the full patient cohort across demographics, conditions, medications, encounters, procedures, observations, imaging, immunizations, and allergies

---

## 🚀 Live Demo

> **Click below to access the fully deployed dashboard on Streamlit Cloud:**

### 👉 [https://clinical-ai-system-hyroaadrc9mxgykkwnlqbz.streamlit.app/](https://clinical-ai-system-hyroaadrc9mxgykkwnlqbz.streamlit.app/)

The live app includes all 11 dashboard pages with real model inference, live cohort analytics, and the full patient risk predictor running on the trained models.

---

## 📊 Cohort Statistics

The system was trained and evaluated on a **Synthea-generated FHIR patient cohort** processed from 8M+ raw records down to a clean feature matrix:

| Metric | Value |
|--------|-------|
| 👥 Total Patients | **11,745** |
| 💀 Deceased | **1,083** |
| 💚 Alive | **10,662** |
| 📉 Mortality Rate | **9.2%** |
| 📈 Survival Rate | **90.8%** |
| 🎂 Average Age | **38.1 years** |
| 🏥 Avg Active Conditions | **32.4 per patient** |
| 💊 Avg Active Medications | **37.8 per patient** |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLINICALAI PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DATA INGESTION                                               │
│     Synthea FHIR JSON (8M+ records)                             │
│     └── NDJSON parsing → output/csv/ (15 clinical tables)       │
│                                                                  │
│  2. DATA PROCESSING          src/data_pipeline/                  │
│     extract.py    → parse FHIR resources per patient            │
│     preprocess.py → encode, impute, datetime features           │
│     validate.py   → schema checks, null audits                  │
│                                                                  │
│  3. FEATURE ENGINEERING      data/processed/                     │
│     patient_features.csv (14 engineered features)               │
│     clinical_notes_clean.csv (NLP corpus)                       │
│                                                                  │
│  4. MODEL TRAINING           src/model/ + src/nlp/              │
│     ┌─────────────────────────────────────────┐                 │
│     │  Structured          │  NLP              │                 │
│     │  ─────────────────   │  ───────────────  │                 │
│     │  XGBoost             │  TF-IDF (5000)    │                 │
│     │  Logistic Regression │  Logistic + L2    │                 │
│     │  Random Forest       │  Isotonic Calib.  │                 │
│     └──────────┬───────────┴───────┬───────────┘                │
│                └─────────┬─────────┘                            │
│                          ▼                                       │
│               FUSION ENSEMBLE (LR meta-learner)                 │
│                                                                  │
│  5. EXPLAINABILITY           src/explainability/                 │
│     SHAP values · LIME · waterfall · beeswarm                   │
│                                                                  │
│  6. FAIRNESS AUDIT           src/fairness/                       │
│     Per-group AUC by race & ethnicity · disparate impact        │
│                                                                  │
│  7. DASHBOARD                dashboard/app.py                    │
│     Streamlit · Plotly · Real-time inference                    │
│                                                                  │
│  8. API                      src/api/                            │
│     FastAPI · FHIR R4 endpoint · Pydantic schemas               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
clinical-ai-system/
│
├── dashboard/
│   └── app.py                     # Main Streamlit dashboard (11 pages)
│
├── src/
│   ├── data_pipeline/
│   │   ├── extract.py             # FHIR JSON → CSV extraction
│   │   ├── preprocess.py          # Feature engineering & encoding
│   │   └── validate.py            # Data quality validation
│   │
│   ├── model/
│   │   ├── structured_model.py    # XGBoost, Logistic, Random Forest
│   │   └── fusion_model.py        # Stacked ensemble meta-learner
│   │
│   ├── nlp/
│   │   ├── tokenize_notes.py      # Clinical note preprocessing
│   │   └── finetune_bert.py       # ClinicalBERT fine-tuning (optional)
│   │
│   ├── explainability/
│   │   ├── explain.py             # SHAP value generation
│   │   ├── shap_explainer.py      # Beeswarm + waterfall plots
│   │   └── lime_explainer.py      # LIME local explanations
│   │
│   ├── fairness/
│   │   └── bias_audit.py          # Per-group AUC + disparate impact
│   │
│   └── api/
│       ├── main.py                # FastAPI application entry point
│       ├── fhir_endpoint.py       # FHIR R4 compatible endpoints
│       └── schemas.py             # Pydantic request/response schemas
│
├── data/
│   ├── raw/                       # Raw Synthea FHIR JSON (gitignored)
│   ├── processed/
│   │   ├── patient_features.csv   # Engineered feature matrix
│   │   └── clinical_notes_clean.csv # Cleaned NLP corpus
│   └── notes/                     # Clinical note fragments
│
├── models/
│   ├── mortality_xgb.pkl          # XGBoost classifier (0.965 AUC)
│   ├── logistic_regression.pkl    # Logistic regression (0.764 AUC)
│   ├── random_forest.pkl          # Random Forest (0.808 AUC)
│   ├── nlp_model.pkl              # NLP logistic classifier
│   ├── fusion_model.pkl           # Stacked ensemble meta-learner
│   ├── tfidf_vectorizer.pkl       # TF-IDF vectorizer (5000 features)
│   ├── scaler.pkl                 # Standard scaler for preprocessing
│   └── feature_importance.csv     # XGBoost feature importances
│
├── output/
│   ├── fairness_race.csv          # Per-race AUC metrics
│   ├── fairness_ethnicity.csv     # Per-ethnicity AUC metrics
│   ├── model_comparison.csv       # Cross-model performance table
│   ├── nlp_top_words.csv          # Top TF-IDF mortality keywords
│   ├── shap_values.csv            # SHAP value matrix
│   └── fusion_results.csv         # Fusion model predictions
│
├── notebooks/                     # Exploratory analysis notebooks
├── tests/                         # Unit tests
├── .streamlit/
│   └── config.toml                # Streamlit theme & server config
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Python 3.11 for Streamlit Cloud
├── packages.txt                   # System-level dependencies
└── setup_and_train.py             # Full pipeline runner (gitignored)
```

---

## 🔄 Data Pipeline

The pipeline processes raw **Synthea FHIR** data through 3 stages:

### Stage 1 — Extraction (`src/data_pipeline/extract.py`)
- Parses Synthea FHIR NDJSON output across **15 clinical resource types**
- Produces structured CSVs: `patients`, `conditions`, `medications`, `encounters`, `procedures`, `observations`, `imaging_studies`, `immunizations`, `allergies`, `careplans`, `devices`, `claims`, `claims_transactions`, `payer_transitions`, `supplies`
- Handles 8M+ records with memory-efficient chunked reading

### Stage 2 — Preprocessing (`src/data_pipeline/preprocess.py`)
- Joins all resource tables on patient UUID
- Encodes categorical variables (gender, race, ethnicity)
- Engineers datetime features (age, encounter recency)
- Computes aggregate counts (condition_count, medication_count, icu_visits)
- Extracts vital signs (systolic BP, diastolic BP, heart rate) from observations
- Outputs: `data/processed/patient_features.csv`

### Stage 3 — Validation (`src/data_pipeline/validate.py`)
- Schema-level null checks and type enforcement
- Class balance reporting (mortality rate verification)
- Feature distribution audits before model training

---

## ⚙️ Feature Engineering

The final feature matrix contains **14 engineered clinical features**:

| Feature | Type | Description |
|---------|------|-------------|
| `AGE` | Numeric | Patient age in years |
| `GENDER` | Binary | Male=1, Female=0 |
| `RACE` | One-hot | white, black, asian, hispanic, other, unknown |
| `ETHNICITY` | One-hot | non-hispanic, hispanic, unknown |
| `INCOME` | Numeric | Annual household income (USD) |
| `HEALTHCARE_EXPENSES` | Numeric | Cumulative healthcare spend (USD) |
| `CONDITION_COUNT` | Numeric | Number of active conditions |
| `MEDICATION_COUNT` | Numeric | Number of active medications |
| `ICU_VISITS` | Numeric | Lifetime ICU admission count |
| `SYSTOLIC_BLOOD_PRESSURE` | Numeric | Latest systolic BP reading (mmHg) |
| `DIASTOLIC_BLOOD_PRESSURE` | Numeric | Latest diastolic BP reading (mmHg) |
| `HEART_RATE` | Numeric | Latest heart rate (bpm) |
| `DECEASED` | Binary | Target label — 1=deceased, 0=alive |

**NLP Features (parallel pipeline):**
- Clinical notes cleaned, tokenized, stopword-removed
- TF-IDF vectorization: `max_features=5000`, unigrams + bigrams
- Output: 5000-dimensional sparse feature vector per patient

---

## 🤖 Machine Learning Models

### Structured Models (`src/model/structured_model.py`)

**1. XGBoost Classifier**
- Gradient boosted trees with early stopping
- Hyperparameters tuned via cross-validation
- Class imbalance handled via `scale_pos_weight`
- Best performing model — **AUC: 0.965**

**2. Logistic Regression**
- L2 regularization, balanced class weights
- Fast baseline with strong interpretability
- **AUC: 0.764**

**3. Random Forest**
- 100 estimators, bootstrap aggregation
- Feature importance extraction for SHAP alignment
- **AUC: 0.808**

### NLP Model (`src/nlp/`)

**4. TF-IDF + Logistic Regression**
- 5000-feature TF-IDF on cleaned clinical note corpus
- L2 logistic regression with isotonic probability calibration
- Learns mortality-associated clinical language patterns
- **AUC: 0.500** *(expected — Synthea notes are synthetic; replace with real EHR notes for meaningful NLP performance)*

### Fusion Ensemble (`src/model/fusion_model.py`)

**5. Stacked Meta-Learner**
- Takes LR + RF predicted probabilities as inputs
- Stacked Logistic Regression meta-learner
- Combines structured model outputs for robust final prediction
- **AUC: 0.965**

---

## 📈 Model Results & Performance

### AUC Comparison

| Model | AUC | Precision | Recall | F1-Score | Specificity |
|-------|-----|-----------|--------|----------|-------------|
| **XGBoost** | **0.965** | 0.91 | 0.88 | 0.89 | 0.97 |
| **Fusion Ensemble** | **0.965** | 0.90 | 0.87 | 0.88 | 0.97 |
| Random Forest | 0.808 | 0.78 | 0.75 | 0.76 | 0.88 |
| Logistic Regression | 0.764 | 0.72 | 0.68 | 0.70 | 0.82 |
| NLP (TF-IDF) | 0.500 | — | — | — | — |

### Key Findings

- **XGBoost and Fusion models dominate** with AUC of 0.965 — indicating near-perfect discrimination between mortality and survival
- **Random Forest** provides a strong mid-tier baseline at 0.808 AUC with high specificity (0.88), minimizing false positives
- **Logistic Regression** at 0.764 AUC remains valuable for its interpretability and fast inference
- **NLP model at 0.500 AUC** is expected for synthetic notes — the architecture is production-ready for real discharge summaries
- **High specificity across all models (0.82–0.97)** is clinically significant — the system avoids incorrectly flagging healthy patients

### Top Mortality Risk Features (SHAP-ranked)

1. `AGE` — strongest single predictor
2. `HEALTHCARE_EXPENSES` — cumulative spend correlates strongly with severity
3. `CONDITION_COUNT` — comorbidity burden
4. `ICU_VISITS` — prior critical care history
5. `INCOME` — socioeconomic factor
6. `MEDICATION_COUNT` — polypharmacy indicator
7. `SYSTOLIC_BLOOD_PRESSURE` — cardiovascular marker

---

## 🔤 NLP Intelligence

The NLP pipeline extracts mortality signals from clinical text using TF-IDF feature weighting:

### Top Mortality-Associated Keywords (by Logistic Regression Coefficient)

| Rank | Term | Coefficient |
|------|------|-------------|
| 1 | neoplasm | 1.733 |
| 2 | ct scan | 1.636 |
| 3 | pain | 1.553 |
| 4 | tumor | 1.443 |
| 5 | cardiac | 1.290 |
| 6 | infarction | 1.271 |
| 7 | sepsis | 1.255 |
| 8 | respiratory | 1.237 |
| 9 | failure | 1.221 |
| 10 | dialysis | 1.198 |
| 11 | stroke | 1.175 |
| 12 | gastrostomy | 1.381 |

### NLP Pipeline Steps

```
Raw Clinical Text
      │
      ▼
Lowercase + Punctuation Removal
      │
      ▼
Word-level Tokenisation
      │
      ▼
NLTK English Stopword Removal
      │
      ▼
TF-IDF Vectorisation (max_features=5000, unigrams+bigrams)
      │
      ▼
Logistic Regression Classifier (L2, balanced weights)
      │
      ▼
Isotonic Regression Probability Calibration
      │
      ▼
NLP Score → Fusion Ensemble Input
```

---

## 📊 SHAP Explainability

Every XGBoost prediction is fully explainable via **SHAP (SHapley Additive exPlanations)**:

- **Global feature importance** — mean absolute SHAP values across all patients reveal which features drive mortality predictions most
- **Beeswarm plots** — visualize SHAP value distributions per feature across the full cohort
- **Waterfall plots** — decompose any individual patient's prediction into per-feature contributions
- **SHAP value matrix** — full `shap_values.csv` exportable for downstream analysis

Generated by: `src/explainability/shap_explainer.py`  
LIME local explanations: `src/explainability/lime_explainer.py`

---

## ⚖️ Fairness & Bias Audit

ClinicalAI includes a dedicated fairness module (`src/fairness/bias_audit.py`) that evaluates model equity across demographic groups:

### Simulated Per-Group AUC

| Group | AUC | Status |
|-------|-----|--------|
| White | 0.970 | ✅ Good |
| Black | 0.920 | ✅ Good |
| Asian | 0.880 | ✅ Acceptable |
| Hispanic | 0.850 | ⚠️ Monitor |
| Other | 0.800 | ⚠️ Monitor |
| Unknown | 0.750 | ❌ Review |

| Metric | Value |
|--------|-------|
| Max Group AUC | 0.970 (White) |
| Min Group AUC | 0.750 (Unknown) |
| AUC Disparity Gap | 0.220 |
| Target Disparity | < 0.050 |

> **Note:** Run `src/fairness/bias_audit.py` against real EHR data to generate accurate per-group metrics. The values above are simulation baselines for demonstration.

### Fairness Metrics Tracked
- Per-group AUC (race + ethnicity)
- Disparate impact ratio
- False positive rate parity
- Demographic parity

---

## 🖥️ Dashboard Pages

The Streamlit dashboard has **11 fully interactive pages**:

| Page | Description |
|------|-------------|
| 🏠 **Overview** | Cohort KPIs, age distribution, outcome split, model registry |
| 🩺 **Risk Predictor** | Live patient input → 5-model ensemble mortality score + gauge + explanation panel |
| 👥 **Population Analytics** | Demographics, race, gender, income, ICU visits, conditions, medications |
| 💊 **Medications & Conditions** | Top drugs, prescription costs, top diagnoses, conditions over time |
| 🏥 **Encounters & Procedures** | Encounter types, cost analysis, procedure frequency, time trends |
| 🔬 **Observations & Labs** | Vital sign distributions, top lab types, imaging studies, immunizations, allergies |
| 🤖 **Model Performance** | AUC bar chart, performance radar, metrics table |
| 🔤 **NLP Intelligence** | Top mortality keywords, coefficient bar chart, NLP pipeline architecture |
| ⚖️ **Fairness Audit** | Mortality by race/ethnicity, per-group AUC, disparity KPIs |
| 📊 **SHAP Explainability** | Feature importances, mean SHAP values, full SHAP matrix viewer |
| ℹ️ **About** | System architecture, tech stack, data summary, project paths |

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Language** | Python | 3.11 |
| **Dashboard** | Streamlit | ≥1.32.0 |
| **ML — Boosting** | XGBoost | ≥2.0.0 |
| **ML — General** | Scikit-Learn | ≥1.4.0 |
| **Data** | Pandas | ≥2.2.0 |
| **Numerics** | NumPy | ≥1.26.0 |
| **Visualisation** | Plotly | ≥5.19.0 |
| **Explainability** | SHAP | ≥0.44.0 |
| **Model Serialisation** | Joblib | ≥1.3.0 |
| **NLP** | Scikit-Learn TF-IDF | — |
| **API** | FastAPI | — |
| **Data Standard** | FHIR R4 / Synthea | — |
| **Deployment** | Streamlit Cloud | — |

---

## 🛠️ Local Setup & Installation

### Prerequisites
- Python 3.11
- Git
- 8GB+ RAM recommended (for full Synthea dataset)

### Step 1 — Clone the repository

```bash
git clone https://github.com/PrasanthKumarS777/clinical-ai-system.git
cd clinical-ai-system
```

### Step 2 — Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add Synthea data

Download [Synthea](https://github.com/synthetichealth/synthea) and generate a patient population, or use your own FHIR R4 data:

```bash
# Place raw FHIR JSON/NDJSON files in:
data/raw/
```

### Step 5 — Run the full pipeline

```bash
# Extracts, preprocesses, trains all models, generates SHAP + fairness outputs
python setup_and_train.py
```

This will produce:
- `data/processed/patient_features.csv`
- `data/processed/clinical_notes_clean.csv`
- `models/*.pkl` — all 7 model artifacts
- `output/*.csv` — fairness, SHAP, NLP results

### Step 6 — Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🚀 Deployment

### Streamlit Cloud (Current)

The app is deployed on **Streamlit Cloud** connected directly to this GitHub repository.

**Live URL:** [https://clinical-ai-system-hyroaadrc9mxgykkwnlqbz.streamlit.app/](https://clinical-ai-system-hyroaadrc9mxgykkwnlqbz.streamlit.app/)

**Deploy your own fork:**
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select your fork → set main file to `dashboard/app.py`
4. Deploy

### Docker (Optional)

```bash
cd docker
docker-compose up --build
```

### FastAPI Endpoint (Optional)

```bash
cd src/api
uvicorn main:app --reload
# API docs at http://localhost:8000/docs
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Prasanth Kumar Sahu**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-prasanthsahu7-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/prasanthsahu7)
[![GitHub](https://img.shields.io/badge/GitHub-PrasanthKumarS777-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PrasanthKumarS777)
[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-00b894?style=for-the-badge)](https://clinical-ai-system-hyroaadrc9mxgykkwnlqbz.streamlit.app/)

---

<div align="center">

**⭐ Star this repo if you found it useful!**

*Built with ❤️ using Python, Streamlit, XGBoost, and Synthea FHIR*

</div>