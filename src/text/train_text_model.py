import random
from re import L
from pathlib import Path
import pandas as pd

def load_notes(sample_size =10000):
    print("Loading clinical notes...")
    path = Path("/Users/anandharagaven/Documents/multimodal-clinical-predictions/data/raw/mimic-iv-note/note/discharge.csv.gz")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path, compression="gzip")

    print("Orginal shape: ", df.shape)

    df = df.sample(n= sample_size, random_state=42)

    print("Sampled Size: ", df.shape)
    print(df.head())

    return df

if __name__ == "__main__":
    df = load_notes()

