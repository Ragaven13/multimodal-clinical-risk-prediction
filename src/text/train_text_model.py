import random
from re import L
from pathlib import Path
import pandas as pd


def load_notes(sample_size=10000):
    print("Loading discharge notes...")

    notes_path = Path("/Users/anandharagaven/Documents/multimodal-clinical-predictions/data/raw/mimic-iv-note/note/discharge.csv.gz")

    if not notes_path.exists():
        raise FileNotFoundError(f"File not found: {notes_path}")

    df_notes = pd.read_csv(notes_path, compression="gzip")

    print("Notes shape:", df_notes.shape)

    # Sample
    df_notes = df_notes.sample(n=sample_size, random_state=42)

    return df_notes

def load_labels():
    print("Loading ICU Labels... ")

    labels_path = Path("/Users/anandharagaven/Documents/multimodal-clinical-predictions/data/processed/final_dataset.csv")

    if not labels_path.exists():
        raise FileNotFoundError(f"File not found: {labels_path}")
    
    df_labels = pd.read_csv(labels_path)

    print("Labels shape: ", df_labels.shape)

    return df_labels


def merge_notes_labels(df_notes, df_labels):
    print("Merging notes with ICU Lables... ")
    
    df_notes = df_notes[["subject_id","hadm_id","text"]]

    df_labels = df_labels[["subject_id","hadm_id","icu_admission"]]

    df_merged = pd.merge(
        df_notes,
        df_labels,
        on = ["subject_id","hadm_id"],
        how = "inner"
    )

    print("Merged shape: ", df_merged.shape)
    print(df_merged.head())

    return df_merged



if __name__ == "__main__":
    df_notes = load_notes()
    df_labels = load_labels()
    df_final = merge_notes_labels(df_notes,df_labels)

