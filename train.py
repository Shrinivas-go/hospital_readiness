import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "diabetic_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.environ.get(
    "READMISSION_MODEL_PATH",
    os.path.join(MODEL_DIR, "readmission_model.joblib"),
)
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH)

# Keep the model aligned with the fields the public web form actually collects.
FEATURE_COLUMNS = [
    "age",
    "gender",
    "race",
    "time_in_hospital",
    "num_medications",
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
]
CATEGORICAL_COLUMNS = ["age", "gender", "race"]
NUMERICAL_COLUMNS = [
    "time_in_hospital",
    "num_medications",
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
]


def load_training_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(
            DATA_PATH,
            usecols=FEATURE_COLUMNS + ["readmitted"],
        )

    # The CSV is deliberately not stored in Git. Fetch the public UCI source
    # during a fresh deployment build, then train from only the needed fields.
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "Install ucimlrepo or place diabetic_data.csv in the project root."
        ) from exc

    print("Downloading the UCI training data...")
    dataset = fetch_ucirepo(id=296)
    features = dataset.data.features
    targets = dataset.data.targets

    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in features.columns
    ]
    if missing_columns:
        raise ValueError(
            "The UCI dataset is missing required features: "
            + ", ".join(missing_columns)
        )
    if "readmitted" in targets.columns:
        readmitted = targets["readmitted"]
    elif len(targets.columns) == 1:
        readmitted = targets.iloc[:, 0]
    else:
        raise ValueError("The UCI dataset does not include a readmitted target.")

    data = features.loc[:, FEATURE_COLUMNS].copy()
    data["readmitted"] = readmitted.to_numpy()
    return data


def train_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    df = load_training_data().replace("?", pd.NA)
    df["target"] = (df["readmitted"] == "<30").astype(int)

    X = df[FEATURE_COLUMNS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERICAL_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test, predictions, zero_division=0
        )),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
    }

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "model_name": "Logistic Regression",
        "metrics": metrics,
        "all_results": {"Logistic Regression": metrics},
        "high_missing_columns": [],
        "target_definition": (
            "1 = readmitted within 30 days; "
            "0 = readmitted after 30 days or not readmitted"
        ),
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)

    with open(
        os.path.join(MODEL_DIR, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump({"Logistic Regression": metrics}, file, indent=2)

    print("Selected model: Logistic Regression")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
