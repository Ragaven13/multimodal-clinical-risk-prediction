import pandas as pd
from pathlib import Path


def load_data():
    base = Path("data/raw/mimiciv")

    admissions = pd.read_csv(base / "hosp/admissions.csv.gz")
    patients = pd.read_csv(base / "hosp/patients.csv.gz")
    icu = pd.read_csv(base / "icu/icustays.csv.gz")

    return admissions, patients, icu


def build_dataset(admissions, patients, icu):
    # Merge patients with admissions
    df = admissions.merge(patients, on="subject_id", how="left")

    # Create ICU label (1 if patient had ICU stay)
    icu_flag = icu[["hadm_id"]].drop_duplicates()
    icu_flag["icu_admission"] = 1

    df = df.merge(icu_flag, on="hadm_id", how="left")
    df["icu_admission"] = df["icu_admission"].fillna(0)

    # Select useful features
    df = df[[
        "subject_id",
        "hadm_id",
        "anchor_age",
        "gender",
        "race",
        "hospital_expire_flag",
        "icu_admission"
    ]]

    return df


def save_dataset(df):
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/final_dataset.csv", index=False)
    print("Saved dataset to data/processed/final_dataset.csv")


def main():
    admissions, patients, icu = load_data()

    df = build_dataset(admissions, patients, icu)

    print(df.head())
    print("\nDataset shape:", df.shape)

    save_dataset(df)


if __name__ == "__main__":
    main()