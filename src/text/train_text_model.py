from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------
# LOAD NOTES
# ---------------------------------------------------
def load_notes(sample_size=10000):
    print("Loading discharge notes...")

    path = Path("/Users/anandharagaven/Documents/multimodal-clinical-predictions/data/raw/mimic-iv-note/note/discharge.csv.gz")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, compression="gzip")
    print("Notes shape:", df.shape)

    df = df.sample(n=sample_size, random_state=42)

    return df


# ---------------------------------------------------
# LOAD LABELS
# ---------------------------------------------------
def load_labels():
    print("Loading ICU labels...")

    path = Path("/Users/anandharagaven/Documents/multimodal-clinical-predictions/data/processed/final_dataset.csv")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    print("Labels shape:", df.shape)

    return df


# ---------------------------------------------------
# MERGE NOTES + LABELS
# ---------------------------------------------------
def merge_notes_labels(df_notes, df_labels):
    print("Merging notes with ICU labels...")

    df_notes = df_notes[["subject_id", "hadm_id", "text"]]
    df_labels = df_labels[["subject_id", "hadm_id", "icu_admission"]]

    df_merged = pd.merge(
        df_notes,
        df_labels,
        on=["subject_id", "hadm_id"],
        how="inner"
    )

    print("Merged shape:", df_merged.shape)
    return df_merged


# ---------------------------------------------------
# CLEAN TEXT
# ---------------------------------------------------
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = " ".join(text.split())

    return text


def prepare_text_dataset(df):
    print("Cleaning text and preparing final dataset...")

    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    df["icu_admission"] = df["icu_admission"].astype(int)

    df = df[["text", "icu_admission"]]

    print(df.head())
    print("\nClass distribution:")
    print(df["icu_admission"].value_counts())

    return df


# ---------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------
def tokenize_data(texts, tokenizer, max_length=256):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )


# ---------------------------------------------------
# METRICS
# ---------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    print("Preparing data...")

    df_notes = load_notes()
    df_labels = load_labels()
    df_merged = merge_notes_labels(df_notes, df_labels)
    df = prepare_text_dataset(df_merged)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"],
        df["icu_admission"],
        test_size=0.2,
        random_state=42,
        stratify=df["icu_admission"]
    )

    print("Train size:", len(train_texts))
    print("Test size:", len(test_texts))

    # ---------------------------------------------------
    # LOAD TOKENIZER + MODEL
    # ---------------------------------------------------
    model_name = "emilyalsentzer/Bio_ClinicalBERT"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # ---------------------------------------------------
    # TOKENIZE
    # ---------------------------------------------------
    train_encodings = tokenize_data(train_texts.tolist(), tokenizer)
    test_encodings = tokenize_data(test_texts.tolist(), tokenizer)

    train_labels = torch.tensor(train_labels.tolist())
    test_labels = torch.tensor(test_labels.tolist())

    # ---------------------------------------------------
    # DATASET CLASS
    # ---------------------------------------------------
    class ClinicalNotesDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = ClinicalNotesDataset(train_encodings, train_labels)
    test_dataset = ClinicalNotesDataset(test_encodings, test_labels)

    # ---------------------------------------------------
    # TRAINING CONFIG
    # ---------------------------------------------------
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to="none",
    )

    # ---------------------------------------------------
    # TRAINER
    # ---------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating...")
    results = trainer.evaluate()
    print("\nEvaluation results:")
    print(results)


if __name__ == "__main__":
    main()