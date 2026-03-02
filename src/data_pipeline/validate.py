import pandas as pd
import sys, os
sys.path.insert(0, os.path.abspath("."))

def validate_patient_features(df):
    print("\n--- Validating patient_features.csv ---")
    errors = []

    # Shape check
    assert df.shape[0] > 0, "No rows found!"
    assert df.shape[1] == 14, f"Expected 14 columns, got {df.shape[1]}"
    print(f"✅ Shape: {df.shape}")

    # No duplicate patient IDs
    dupes = df["Id"].duplicated().sum()
    if dupes > 0:
        errors.append(f"❌ Duplicate patient IDs: {dupes}")
    else:
        print("✅ No duplicate patient IDs")

    # Age range check
    invalid_age = df[(df["AGE"] < 0) | (df["AGE"] > 120)].shape[0]
    if invalid_age > 0:
        errors.append(f"❌ Invalid AGE values: {invalid_age} rows")
    else:
        print("✅ AGE range valid (0–120)")

    # DECEASED must be 0 or 1
    invalid_deceased = df[~df["DECEASED"].isin([0, 1])].shape[0]
    if invalid_deceased > 0:
        errors.append(f"❌ Invalid DECEASED values: {invalid_deceased} rows")
    else:
        print("✅ DECEASED is binary (0/1)")

    # GENDER must be 0 or 1
    invalid_gender = df[~df["GENDER"].isin([0, 1])].shape[0]
    if invalid_gender > 0:
        errors.append(f"❌ Invalid GENDER values: {invalid_gender} rows")
    else:
        print("✅ GENDER is binary (0/1)")

    # No nulls in critical columns
    critical_cols = ["Id", "AGE", "GENDER", "DECEASED", "ICU_VISITS",
                     "CONDITION_COUNT", "MEDICATION_COUNT"]
    for col in critical_cols:
        nulls = df[col].isnull().sum()
        if nulls > 0:
            errors.append(f"❌ Nulls in {col}: {nulls}")
        else:
            print(f"✅ No nulls in {col}")

    # Vitals range checks
    vitals = {
        "DIASTOLIC_BLOOD_PRESSURE": (40, 150),
        "SYSTOLIC_BLOOD_PRESSURE": (60, 250),
        "HEART_RATE": (30, 200),
    }
    for col, (low, high) in vitals.items():
        if col in df.columns:
            out = df[(df[col] < low) | (df[col] > high)].shape[0]
            if out > 0:
                errors.append(f"⚠️  {col} has {out} out-of-range values")
            else:
                print(f"✅ {col} within expected range ({low}–{high})")

    return errors

def validate_notes(df):
    print("\n--- Validating clinical_notes_clean.csv ---")
    errors = []

    assert df.shape[0] > 0, "No notes found!"
    print(f"✅ Notes count: {df.shape[0]}")

    nulls = df["transcription"].isnull().sum()
    if nulls > 0:
        errors.append(f"❌ Null transcriptions: {nulls}")
    else:
        print("✅ No null transcriptions")

    short = df[df["text_length"] <= 100].shape[0]
    if short > 0:
        errors.append(f"❌ Notes with text_length <= 100: {short}")
    else:
        print("✅ All notes have text_length > 100")

    return errors

if __name__ == "__main__":
    features = pd.read_csv("data/processed/patient_features.csv")
    notes = pd.read_csv("data/processed/clinical_notes_clean.csv")

    errs1 = validate_patient_features(features)
    errs2 = validate_notes(notes)

    all_errors = errs1 + errs2
    print("\n--- Validation Summary ---")
    if all_errors:
        for e in all_errors:
            print(e)
    else:
        print("✅ All validations passed! Data is clean and ready for modeling.")
