from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
    X = df[["anchor_age", "gender", "race", "hospital_expire_flag"]]

    numeric_features = ["anchor_age", "hospital_expire_flag"]
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

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=300,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("Step 4: Training balanced Logistic Regression model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    print("Step 5: Making predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\nAccuracy:", accuracy)
    print("ROC-AUC:", roc_auc)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    tn, fp, fn, tp = cm.ravel()
    print("\nDetailed Confusion Matrix Values:")
    print("True Negatives :", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives :", tp)

    return model


def main():
    print("Starting training script...")
    df = load_dataset()
    X, y, preprocessor = prepare_data(df)
    _ = train_model(X, y, preprocessor)
    print("Training script finished successfully.")


if __name__ == "__main__":
    main()