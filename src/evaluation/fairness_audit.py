from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def load_dataset():
    file_path = Path("data/processed/final_dataset.csv")
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    return pd.read_csv(file_path)


def train_model(df):
    y = df["icu_admission"].astype(int)
    X = df[["anchor_age", "gender", "race"]]

    numeric_features = ["anchor_age"]
    categorical_features = ["gender", "race"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = 0.3
    y_pred = (y_prob >= threshold).astype(int)

    results_df = X_test.copy()
    results_df["y_true"] = y_test.values
    results_df["y_pred"] = y_pred

    return results_df


def safe_confusion_matrix(y_true, y_pred):
    """
    Always return tn, fp, fn, tp for binary labels 0/1.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp


def fairness_by_group(results_df, group_col):
    print(f"\nFairness analysis by {group_col}:\n")

    groups = sorted(results_df[group_col].dropna().unique())
    rows = []

    for group in groups:
        subset = results_df[results_df[group_col] == group].copy()

        if len(subset) == 0:
            continue

        y_true = subset["y_true"]
        y_pred = subset["y_pred"]

        tn, fp, fn, tp = safe_confusion_matrix(y_true, y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        actual_positive = int((y_true == 1).sum())
        predicted_positive = int((y_pred == 1).sum())

        row = {
            "group": group,
            "size": len(subset),
            "actual_positive": actual_positive,
            "predicted_positive": predicted_positive,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
        rows.append(row)

        print(f"{group_col} = {group}")
        print(f"  Size               : {len(subset)}")
        print(f"  Actual ICU cases   : {actual_positive}")
        print(f"  Predicted ICU cases: {predicted_positive}")
        print(f"  Accuracy           : {accuracy:.4f}")
        print(f"  Precision          : {precision:.4f}")
        print(f"  Recall             : {recall:.4f}")
        print(f"  TN                 : {tn}")
        print(f"  FP                 : {fp}")
        print(f"  FN                 : {fn}")
        print(f"  TP                 : {tp}")
        print("-" * 40)

    return pd.DataFrame(rows)


def main():
    print("Running corrected fairness audit...")

    df = load_dataset()
    results_df = train_model(df)

    race_results = fairness_by_group(results_df, "race")
    gender_results = fairness_by_group(results_df, "gender")

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    race_path = output_dir / "fairness_race.csv"
    gender_path = output_dir / "fairness_gender.csv"

    race_results.to_csv(race_path, index=False)
    gender_results.to_csv(gender_path, index=False)

    print(f"\nSaved race fairness results to: {race_path}")
    print(f"Saved gender fairness results to: {gender_path}")


if __name__ == "__main__":
    main()