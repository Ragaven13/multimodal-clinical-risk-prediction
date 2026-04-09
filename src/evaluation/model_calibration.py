from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def load_dataset():
    return pd.read_csv("data/processed/final_dataset.csv")


def train_model(df):
    y = df["icu_admission"].astype(int)
    X = df[["anchor_age", "gender", "race"]]

    numeric_features = ["anchor_age"]
    categorical_features = ["gender", "race"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    return y_test, y_prob


def evaluate_thresholds(y_test, y_prob):
    thresholds = np.linspace(0.1, 0.9, 30)

    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    df = pd.DataFrame(results)

    return df


def find_best_threshold(df):
    # Keep high recall models
    df_filtered = df[df["recall"] >= 0.90]

    # Among those, pick best precision
    best = df_filtered.sort_values(by="precision", ascending=False).iloc[0]

    print("\nBest threshold (recall >= 0.90):")
    print(best)

    return best


def plot_curve(df):
    plt.figure()

    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.plot(df["threshold"], df["f1"], label="F1 Score")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs Metrics")
    plt.legend()

    output_path = "docs/threshold_curve.png"
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

    plt.close()


def main():
    print("Running model calibration...")

    df = load_dataset()
    y_test, y_prob = train_model(df)

    results_df = evaluate_thresholds(y_test, y_prob)

    best = find_best_threshold(results_df)

    plot_curve(results_df)

    results_df.to_csv("data/processed/threshold_results.csv", index=False)
    print("Saved threshold table.")

    print("\nFINAL RECOMMENDED THRESHOLD:", best["threshold"])


if __name__ == "__main__":
    main()