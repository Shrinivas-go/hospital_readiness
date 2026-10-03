
import os
import joblib
import pandas as pd

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "readmission_model.joblib"
)

bundle = None


def load_model():
    global bundle

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run: python train.py"
        )

    bundle = joblib.load(MODEL_PATH)


def get_risk_level(probability):
    # Demonstration thresholds only; choose final thresholds
    # using validation data and the intended use case.
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    return "High"


def predict_readmission(patient_data):
    if bundle is None:
        load_model()

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    # Ensure input columns match the training columns.
    patient_df = pd.DataFrame([patient_data])
    patient_df = patient_df.reindex(columns=feature_columns)

    probabilities = model.predict_proba(patient_df)[0]
    risk_probability = float(probabilities[1])

    return {
        "model": bundle["model_name"],
        "readmission_probability": round(
            risk_probability * 100, 2
        ),
        "risk_level": get_risk_level(risk_probability),
        "disclaimer": (
            "Research prototype only. Not for clinical "
            "decision-making."
        ),
    }


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": bundle is not None,
    })


@app.route("/api/metrics", methods=["GET"])
def metrics():
    if bundle is None:
        try:
            load_model()
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 503

    return jsonify({
        "model_name": bundle["model_name"],
        "metrics": bundle["metrics"],
        "model_comparison": bundle["all_results"],
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not request.is_json:
        return jsonify({
            "error": "Send patient information as JSON."
        }), 400

    patient_data = request.get_json(silent=True)

    if not isinstance(patient_data, dict) or not patient_data:
        return jsonify({
            "error": "Patient information is required."
        }), 400

    try:
        result = predict_readmission(patient_data)
        return jsonify(result)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": (
                "Prediction failed. Check that the input "
                "features match the trained model."
            )
        }), 400


@app.route("/predict", methods=["POST"])
def predict_form():
    try:
        patient_data = request.form.to_dict()
        result = predict_readmission(patient_data)

        return render_template(
            "index.html",
            result=result,
        )
    except FileNotFoundError as exc:
        return render_template(
            "index.html",
            error=str(exc),
        ), 503
    except Exception:
        app.logger.exception("Form prediction failed")
        return render_template(
            "index.html",
            error="Prediction failed. Please check the input.",
        ), 400


if __name__ == "__main__":
    try:
        load_model()
        print("Readmission model loaded successfully.")
    except FileNotFoundError as exc:
        print(exc)

    app.run(host="127.0.0.1", port=5000, debug=False)
