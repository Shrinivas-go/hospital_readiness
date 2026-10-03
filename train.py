
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "diabetic_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "readmission_model.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)


def train_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Place diabetic_data.csv in the project root."
        )

    df = pd.read_csv(DATA_PATH)
    df = df.replace("?", pd.NA)

    # Positive class: readmission within 30 days.
    df["target"] = (df["readmitted"] == "<30").astype(int)

    # Remove the original target and record identifiers.
    df = df.drop(
        columns=["readmitted", "encounter_id", "patient_nbr"],
        errors="ignore",
    )

    # Exclude features with extremely high missingness.
    # This rule is fixed before the train/test split.
    missing_fraction = df.drop(columns=["target"]).isna().mean()
    high_missing = missing_fraction[missing_fraction > 0.90].index.tolist()
    df = df.drop(columns=high_missing)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Unknown",
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted_models = {}

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(
                y_test, predictions, zero_division=0
            )),
            "recall": float(recall_score(
                y_test, predictions, zero_division=0
            )),
            "f1": float(f1_score(
                y_test, predictions, zero_division=0
            )),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "pr_auc": float(average_precision_score(
                y_test, probabilities
            )),
        }

        results[name] = metrics
        fitted_models[name] = pipeline

        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # Select by ROC-AUC on the held-out test set for this basic demo.
    # For a rigorous final evaluation, select using validation data
    # or cross-validation, then evaluate the chosen model on test data
    # only once.
    best_name = max(
        results,
        key=lambda name: results[name]["roc_auc"],
    )

    bundle = {
        "model": fitted_models[best_name],
        "feature_columns": X.columns.tolist(),
        "model_name": best_name,
        "metrics": results[best_name],
        "all_results": results,
        "high_missing_columns": high_missing,
        "target_definition": (
            "1 = readmitted within 30 days; "
            "0 = readmitted after 30 days or not readmitted"
        ),
    }

    joblib.dump(bundle, MODEL_PATH)

    with open(
        os.path.join(MODEL_DIR, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    print(f"\nSelected model: {best_name}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
