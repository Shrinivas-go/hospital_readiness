# Hospital Readmission Risk Prototype

A small educational demo for estimating 30-day hospital readmission risk. The GitHub Pages site serves the form, and a Flask API serves predictions. This is not a clinical tool; do not use predictions to make patient-care decisions.

## Public site

- Front end: <https://shrinivas-go.github.io/hospital_readiness/>
- API: `https://hospital-readiness-api-shrinivas-2026.onrender.com`

The front end calls the API over HTTPS. The Flask API allows requests only from this GitHub Pages origin and the local development origins.

## Deploy the API

The repository includes a Render Blueprint in `render.yaml`. In Render, create a Blueprint from this public repository and apply the configuration. It creates a free Python web service in Singapore, installs the requirements, trains a compact model, and starts Gunicorn. Its `/health` endpoint checks that the model has loaded.

The model is intentionally limited to the fields shown in the form and is saved as a compressed artifact during the build. The training script uses a local `diabetic_data.csv` when present; otherwise it fetches the public dataset through UCI's repository client. The patient dataset and model artifact are not committed to GitHub.

Render's free web services spin down after 15 minutes without traffic, so a first visit after inactivity can take about a minute to wake up. See [Render's free service limits](https://render.com/docs/free).

## Run locally

Python 3.12 is recommended.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python app.py
```

### macOS or Linux

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python app.py
```

Open <http://127.0.0.1:5000/>. `train.py` uses the local CSV if available or downloads the UCI dataset as needed. UCI identifies this as patient data with sensitive demographic attributes; it remains outside this public repository. The dataset is licensed CC BY 4.0; cite Clore, Cios, DeShazo, and Strack (2014), DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J).

The API exposes `GET /health`, `GET /api/metrics`, and `POST /api/predict`.
