import os
import joblib
import pandas as pd

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "READMISSION_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "readmission_model.joblib"),
)
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH)
DEFAULT_CORS_ORIGINS = {
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    "https://shrinivas-go.github.io",
}

bundle = None


def get_allowed_origins():
    configured = os.environ.get("CORS_ALLOWED_ORIGINS")
    if configured:
        return {
            origin.strip()
            for origin in configured.split(",")
            if origin.strip()
        }
    return DEFAULT_CORS_ORIGINS


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin in get_allowed_origins():
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Vary"] = "Origin"
    return response


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
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    if bundle is None:
        try:
            load_model()
        except FileNotFoundError as exc:
            return jsonify({
                "status": "error",
                "model_loaded": False,
                "error": str(exc),
            }), 503

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


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def api_predict():
    if request.method == "OPTIONS":
        return "", 204

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


if __name__ == "__main__":
    try:
        load_model()
        print("Readmission model loaded successfully.")
    except FileNotFoundError as exc:
        print(exc)

    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
