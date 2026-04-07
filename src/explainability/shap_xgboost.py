from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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


def train_xgboost_model(X, y, preprocessor):
    print("Step 3: Splitting train/test data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    print("Step 4: Building preprocessing pipeline...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    print("Step 5: Training XGBoost model...")
    model = XGBClassifier(
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
    )

    model.fit(X_train_processed, y_train)
    print("Model training complete.")

    return model, X_train_processed, X_test_processed, y_test, feature_names


def create_shap_outputs(model, X_test_processed, feature_names):
    print("Step 6: Computing SHAP values...")

    # Use a subset for speed
    max_samples = 2000
    if X_test_processed.shape[0] > max_samples:
        X_shap = X_test_processed[:max_samples]
    else:
        X_shap = X_test_processed

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    output_dir = Path("docs/shap_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 7: Creating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
    plt.tight_layout()
    summary_path = output_dir / "shap_summary_plot.png"
    plt.savefig(summary_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved summary plot to: {summary_path}")

    print("Step 8: Creating SHAP bar plot...")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    bar_path = output_dir / "shap_bar_plot.png"
    plt.savefig(bar_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved bar plot to: {bar_path}")

    return shap_values, X_shap


def save_top_feature_importance(shap_values, feature_names):
    print("Step 9: Saving top feature importance table...")

    importance = abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": importance
    }).sort_values(by="mean_abs_shap", ascending=False)

    output_dir = Path("docs/shap_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "shap_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    print("Top 10 SHAP features:")
    print(importance_df.head(10))
    print(f"Saved SHAP feature importance to: {csv_path}")


def main():
    print("Starting SHAP explainability pipeline...")
    df = load_dataset()
    X, y, preprocessor = prepare_data(df)
    model, X_train_processed, X_test_processed, y_test, feature_names = train_xgboost_model(
        X, y, preprocessor
    )
    shap_values, X_shap = create_shap_outputs(model, X_test_processed, feature_names)
    save_top_feature_importance(shap_values, feature_names)
    print("SHAP explainability pipeline finished successfully.")


if __name__ == "__main__":
    main()