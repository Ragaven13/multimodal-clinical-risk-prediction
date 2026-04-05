from pathlib import Path

import pandas as pd
from pandas.core.arrays import categorical
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, _name_estimators
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def load_dataset():
    print("Loading Dataset.. ")

    file_path = Path("data/processed/final_dataset.csv")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. Run src/features/build_datasets.py first."
        )


    df = pd.read_csv(file_path)
    print("Loaded Datset shape: ", df.shape)
    return df



def prepare_data(df: pd.DataFrame):
    print("Step 2: Preparing data..")

    df = df.copy()

    y = df["icu_admission"].astype(int)
    X = df[["anchor_age","gender","race","hospital_expire_flag"]]

    numerical_features = ["anchor_age","hospital_expire_flag"]
    categorical_features = ["gender","race"]

    numerical_transformer = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps = [
            ("imputer",SimpleImputer(strategy ="most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown="ignore")),

        ]
    )

    preprocessor= ColumnTransformer(
        transformers = [
            ("num",numerical_transformer,numerical_features),
            ("cat",categorical_transformer,categorical_features)
        ]
    )

    print("Finished Preparing Data.. ")
    return X, y, preprocessor



def train_model(X,y,preprocessor):
    print("Step 3 : Splitting train/test data... ")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training set shape: ", X_train.shape)
    print("Test set shape: ", X_test.shape)

    # Handling imbalance 
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1 ).sum()
    scale_pos_weight = negative_count / positive_count

    print("Negative classs count: ", negative_count)
    print("Positive class count: ", positive_count)
    print("scale_pos_weight: ", scale_pos_weight)


    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    _name_estimators =100,
                    max_depths= 5,
                    learning_rate = 0.1,
                    subsample = 0.8,
                    colsample_bytree = 0.8,
                    objective = "binary:logistic",
                    eval_metric="logloss",
                    random_state = 42,
                    scale_pos_weight  = scale_pos_weight,
                    n_jobs = -1,

                ),
            ),
        ]
        )

    print("Step 4: Training XGBoost model... ")
    model.fit(X_train,y_train)
    print("Model training complete. ")


    print("Step 5: Making Prediction.. ")
    threshold = 0.3
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)


    print("\n Accuracy: ", accuracy)
    print("ROC-AUC: ",roc_auc)
    print("\nConfusion Matrix:\n",cm)
    print("\nClassification Report:\n", classification_report(y_test,y_pred))

    tn , fp, fn,tp = cm.ravel()

    print("\n Detailed confusion matrix Values: ")
    print("True Negatives: ", tn )
    print("False Positives: ", fp)
    print("False Negatives: ",fn)
    print("True Positives: ", tp)

    return model

def main():
    print("Starting XGBoost training script.... ")
    df = load_dataset()
    X, y,  preprocessor = prepare_data(df)
    _ = train_model(X,y,preprocessor)
    print("XGBoost training script finished sucessfully. ")


if __name__ == "__main__":
    main()












