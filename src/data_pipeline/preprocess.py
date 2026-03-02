import pandas as pd
import numpy as np
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.data_pipeline.extract import load_all

def calculate_age(birthdate, deathdate=None):
    ref = pd.to_datetime(deathdate) if pd.notna(deathdate) else datetime.now()
    birth = pd.to_datetime(birthdate)
    return (ref - birth).days // 365

def build_patient_features(patients, encounters, conditions, medications, observations):
    print("Building patient features...")

    patients["AGE"] = patients.apply(
        lambda r: calculate_age(r["BIRTHDATE"], r["DEATHDATE"]), axis=1
    )
    patients["DECEASED"] = patients["DEATHDATE"].notna().astype(int)

    icu = encounters[encounters["ENCOUNTERCLASS"].isin(["inpatient", "emergency"])]
    icu_counts = icu.groupby("PATIENT").size().reset_index(name="ICU_VISITS")

    cond_counts = conditions.groupby("PATIENT").size().reset_index(name="CONDITION_COUNT")

    med_counts = medications.groupby("PATIENT").size().reset_index(name="MEDICATION_COUNT")

    key_labs = ["Creatinine", "Glucose", "Diastolic Blood Pressure",
                "Systolic Blood Pressure", "Heart rate", "Body Mass Index"]
    obs_filtered = observations[observations["DESCRIPTION"].isin(key_labs)].copy()
    obs_filtered["VALUE"] = pd.to_numeric(obs_filtered["VALUE"], errors="coerce")
    obs_pivot = obs_filtered.groupby(["PATIENT", "DESCRIPTION"])["VALUE"].mean().unstack()
    obs_pivot.columns = [c.replace(" ", "_").upper() for c in obs_pivot.columns]
    obs_pivot = obs_pivot.reset_index().rename(columns={"PATIENT": "Id"})

    df = patients[["Id", "AGE", "GENDER", "RACE", "ETHNICITY",
                   "HEALTHCARE_EXPENSES", "INCOME", "DECEASED"]].copy()
    df = df.merge(icu_counts, left_on="Id", right_on="PATIENT", how="left").drop(columns=["PATIENT"])
    df = df.merge(cond_counts, left_on="Id", right_on="PATIENT", how="left").drop(columns=["PATIENT"])
    df = df.merge(med_counts, left_on="Id", right_on="PATIENT", how="left").drop(columns=["PATIENT"])
    df = df.merge(obs_pivot, on="Id", how="left")

    df["ICU_VISITS"] = df["ICU_VISITS"].fillna(0)
    df["CONDITION_COUNT"] = df["CONDITION_COUNT"].fillna(0)
    df["MEDICATION_COUNT"] = df["MEDICATION_COUNT"].fillna(0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Clip vitals to valid physiological ranges
    df["DIASTOLIC_BLOOD_PRESSURE"] = df["DIASTOLIC_BLOOD_PRESSURE"].clip(lower=40, upper=150)

    df["GENDER"] = df["GENDER"].map({"M": 0, "F": 1}).fillna(0)

    print(f"Patient feature matrix built: {df.shape}")
    print(f"Mortality rate: {df['DECEASED'].mean():.2%}")
    return df

def preprocess_notes(notes):
    print("Preprocessing clinical notes...")
    notes = notes.copy()
    notes["transcription"] = notes["transcription"].str.lower()
    notes["transcription"] = notes["transcription"].str.replace(r"\s+", " ", regex=True)
    notes["transcription"] = notes["transcription"].str.strip()
    notes["text_length"] = notes["transcription"].str.len()
    notes = notes[notes["text_length"] > 100]
    print(f"Notes after cleaning: {len(notes)}")
    return notes

if __name__ == "__main__":
    data = load_all()
    features = build_patient_features(
        data["patients"], data["encounters"],
        data["conditions"], data["medications"],
        data["observations"]
    )
    notes = preprocess_notes(data["notes"])
    features.to_csv("data/processed/patient_features.csv", index=False)
    notes.to_csv("data/processed/clinical_notes_clean.csv", index=False)
    print("Saved to data/processed/")
    print(features.head(3))
