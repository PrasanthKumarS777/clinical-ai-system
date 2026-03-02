from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.path.abspath("."))

app = FastAPI(title="Clinical AI API", version="1.0.0")

# Load models on startup
xgb_model  = joblib.load("models/mortality_xgb.pkl")
nlp_model  = joblib.load("models/nlp_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
fusion     = joblib.load("models/fusion_model.pkl")

class PatientInput(BaseModel):
    age: int
    gender: int
    race: int
    ethnicity: int
    healthcare_expenses: float
    income: float
    icu_visits: int
    condition_count: int
    medication_count: int
    diastolic_bp: float
    heart_rate: float
    systolic_bp: float
    body_mass_index: float = 27.0
    creatinine: float = 1.0
    glucose: float = 100.0
    clinical_note: str = ""

@app.get("/")
def root():
    return {"message": "Clinical AI API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

@app.post("/predict")
def predict(patient: PatientInput):
    try:
        X = pd.DataFrame([{
            "AGE": patient.age,
            "GENDER": patient.gender,
            "RACE": patient.race,
            "ETHNICITY": patient.ethnicity,
            "HEALTHCARE_EXPENSES": patient.healthcare_expenses,
            "INCOME": patient.income,
            "ICU_VISITS": patient.icu_visits,
            "CONDITION_COUNT": patient.condition_count,
            "MEDICATION_COUNT": patient.medication_count,
            "DIASTOLIC_BLOOD_PRESSURE": patient.diastolic_bp,
            "HEART_RATE": patient.heart_rate,
            "SYSTOLIC_BLOOD_PRESSURE": patient.systolic_bp,
            "BODY_MASS_INDEX": patient.body_mass_index,
            "CREATININE": patient.creatinine,
            "GLUCOSE": patient.glucose
        }])

        xgb_prob = float(xgb_model.predict_proba(X)[0][1])

        text_tfidf = vectorizer.transform([patient.clinical_note])
        nlp_prob   = float(nlp_model.predict_proba(text_tfidf)[0][1])

        X_meta      = np.array([[xgb_prob, nlp_prob]])
        fusion_prob = float(fusion.predict_proba(X_meta)[0][1])

        risk = "HIGH" if fusion_prob > 0.5 else "MEDIUM" if fusion_prob > 0.2 else "LOW"

        return {
            "xgboost_probability":  round(xgb_prob, 4),
            "nlp_probability":      round(nlp_prob, 4),
            "fusion_probability":   round(fusion_prob, 4),
            "risk_level":           risk,
            "survival_probability": round(1 - fusion_prob, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
