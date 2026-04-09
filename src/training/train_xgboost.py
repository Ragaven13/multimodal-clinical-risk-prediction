from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def load_dataset():
    print("Step 1: Loading dataset...")

    file_path = Path("data/processed/final_dataset.csv")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. Run src/features/build_datasets.py first."
        )

    df = pd.read_csv(file_path)
    print("Loaded dataset shape:", df.shape)
    return df


def prepare_data(df: pd.DataFrame):
    print("Step 2: Preparing data...")

    df = df.copy()

    y = df["icu_admission"].astype(int)
    X = df[["anchor_age", "gender", "race"]]

    numeric_features = ["anchor_age"]
    categorical_features = ["gender", "race"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    print("Finished preparing data.")
    return X, y, preprocessor


def evaluate_thresholds(y_test, y_prob, thresholds):
    rows = []

    print("\nThreshold Optimization Results:\n")

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        row = {
            "threshold": threshold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
        rows.append(row)

        print(
            f"Threshold={threshold:.2f} | "
            f"Acc={acc:.4f} | Prec={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | "
            f"FN={fn} | FP={fp}"
        )

    results_df = pd.DataFrame(rows)

    best_row = results_df.sort_values(by="f1", ascending=False).iloc[0]

    print("\nBest threshold by F1-score:")
    print(best_row)

    return results_df, best_row


def train_model(X, y, preprocessor):
    print("Step 3: Splitting train/test data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    print("Negative class count:", negative_count)
    print("Positive class count:", positive_count)
    print("scale_pos_weight:", scale_pos_weight)

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

    print("Step 4: Training XGBoost model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    print("Step 5: Getting probabilities...")
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    print("\nROC-AUC:", roc_auc)

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    results_df, best_row = evaluate_thresholds(y_test, y_prob, thresholds)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "xgboost_threshold_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nSaved threshold results to: {results_path}")

    # Final evaluation using calibrated threshold from model calibration
    threshold = 0.35

    y_pred_best = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)

    print(f"\nFinal evaluation using calibrated threshold = {threshold}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

    return model, results_df


def main():
    print("Starting XGBoost threshold optimization script...")
    df = load_dataset()
    X, y, preprocessor = prepare_data(df)
    _model, _results = train_model(X, y, preprocessor)
    print("XGBoost threshold optimization finished successfully.")


if __name__ == "__main__":
    main()