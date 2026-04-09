from pathlib import Path
from tokenize import group
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def load_dataset():
    file_path = Path("data/processed/final_dataset.csv")
    df = pd.read_csv(file_path)
    return df


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
    threshold = 0.3
    y_pred = (y_prob >= threshold).astype(int)

    return X_test, y_test, y_pred


def fairness_by_group(X_test, y_test, y_pred, group_col):
    print(f"\n Fairness analysis by  {group_col}: \n")


    df = X_test.copy()
    df["y_true"] = y_test
    df["y_pred"] = y_pred

    groups = df[group_col].unique()

    results = []


    for g in groups:
        subset = df[df[group_col] == g]

        acc = accuracy_score(subset["y_true"],subset["y_pred"])
        rec = accuracy_score(subset["y_true"], subset["y_pred"])


        results.append({
            "group" : g,
            "size": len(subset),
            "accuracy":acc,
            "recall":rec
        })

        print(f"{group_col} = {g}")
        print(f" Size: {len(subset)}")
        print(f" Accuracy: {acc:.4f}")
        print(f" Recall : {rec:.4f}")
        print("_"*30)

    return pd.DataFrame(results)

def main():
    print("Running fairness audit...")

    df = load_dataset()
    X_test, y_test, y_pred = train_model(df)

    race_results = fairness_by_group(X_test, y_test, y_pred, "race")
    gender_results = fairness_by_group(X_test, y_test, y_pred, "gender")

    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)

    race_results.to_csv(output_dir / "fairness_race.csv", index=False)
    gender_results.to_csv(output_dir / "fairness_gender.csv", index=False)

    print("\nSaved fairness results to data/processed/")


if __name__ == "__main__":
    main()