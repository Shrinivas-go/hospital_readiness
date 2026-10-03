
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "readmission_model.joblib"
)


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), (
        "Model file missing. Run python train.py first."
    )


def test_model_prediction():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    # Construct a row with every expected model feature.
    # The pipeline imputes missing values.
    patient = pd.DataFrame(
        [{column: None for column in bundle["feature_columns"]}]
    )

    # Supply example values for common numerical features.
    example_values = {
        "time_in_hospital": 4,
        "num_medications": 10,
        "number_inpatient": 0,
        "number_emergency": 0,
        "number_outpatient": 0,
    }

    for column, value in example_values.items():
        if column in patient.columns:
            patient.loc[0, column] = value

    probabilities = model.predict_proba(patient)

    assert probabilities.shape == (1, 2)
    assert abs(probabilities[0].sum() - 1.0) < 1e-6
    assert 0 <= probabilities[0, 1] <= 1


if __name__ == "__main__":
    test_model_file_exists()
    test_model_prediction()
    print("All basic model tests passed.")
