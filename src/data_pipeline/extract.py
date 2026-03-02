import pandas as pd
import os

DATA_DIR = "data/raw/synthea/csv"
NOTES_DIR = "data/raw/notes"

def load_patients():
    path = os.path.join(DATA_DIR, "patients.csv")
    df = pd.read_csv(path)
    print(f"Patients loaded: {len(df)}")
    return df

def load_observations():
    path = os.path.join(DATA_DIR, "observations.csv")
    df = pd.read_csv(path, low_memory=False)
    print(f"Observations loaded: {len(df)}")
    return df

def load_conditions():
    path = os.path.join(DATA_DIR, "conditions.csv")
    df = pd.read_csv(path)
    print(f"Conditions loaded: {len(df)}")
    return df

def load_encounters():
    path = os.path.join(DATA_DIR, "encounters.csv")
    df = pd.read_csv(path)
    print(f"Encounters loaded: {len(df)}")
    return df

def load_medications():
    path = os.path.join(DATA_DIR, "medications.csv")
    df = pd.read_csv(path)
    print(f"Medications loaded: {len(df)}")
    return df

def load_clinical_notes():
    path = os.path.join(NOTES_DIR, "mtsamples.csv")
    df = pd.read_csv(path)
    df = df[["description", "medical_specialty", "transcription", "keywords"]]
    df = df.dropna(subset=["transcription"])
    print(f"Clinical notes loaded: {len(df)}")
    return df

def load_all():
    print("--- Loading all datasets ---")
    data = {
        "patients": load_patients(),
        "observations": load_observations(),
        "conditions": load_conditions(),
        "encounters": load_encounters(),
        "medications": load_medications(),
        "notes": load_clinical_notes()
    }
    print("--- All datasets loaded successfully ---")
    return data

if __name__ == "__main__":
    data = load_all()
    for name, df in data.items():
        print(f"{name}: {df.shape}")
