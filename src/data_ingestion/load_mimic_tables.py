from pathlib import Path
import pandas as pd


def get_data_path():
    base_raw = Path("data/raw")

    paths = {
        "mimiciv_hosp": base_raw/"mimiciv/" / "hosp",
         "mimiciv_icu": base_raw / "mimiciv" / "icu",
        "mimicv_note":base_raw / "mimicv_note",
        "mimicv_cxr_jpg":base_raw/"mimic_cxr_jpg",
    }

    return paths

def check_paths_exists(paths: dict):
    print("\nChecking datasets folders...\n")
    for name , path in paths.items():
        if path.exists:
            print(f"[OK] Path exists{name}: {path}")
        else:
            print(f"[MISSING] Path dosen't exists{name}: {path}")

def load_csv_if_exists(file_path: Path):
    if file_path.exists():
        print(f"Loading: {file_path}")
        return pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}")
        return None

def main():
    paths = get_data_path()
    check_paths_exists(paths)

    admissions_path = paths["mimiciv_hosp"] / "admissions.csv.gz"
    patients_path = paths["mimiciv_hosp"] / "patients.csv.gz"
    icustays_path = paths["mimiciv_icu"] / "icustays.csv.gz"

    admissions = load_csv_if_exists(admissions_path)
    patients = load_csv_if_exists(patients_path)
    icustays = load_csv_if_exists(icustays_path)

    print("\nPreview summary:\n")

    if admissions is not None:
        print("Admissions shape:", admissions.shape)
        print(admissions.head(), "\n")

    if patients is not None:
        print("Patients shape:", patients.shape)
        print(patients.head(), "\n")

    if icustays is not None:
        print("ICU stays shape:", icustays.shape)
        print(icustays.head(), "\n")


if __name__ == "__main__":
    main()