# Hospital Readmission Risk Prototype

A small Flask app that estimates the chance of a hospital readmission within 30 days using the UCI Diabetes 130-US Hospitals dataset. This is an educational research prototype, not a clinical tool. Do not use its predictions to make patient-care decisions.

## Requirements

- Python 3.10 or newer
- The `diabetic_data.csv` file from the [UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008)

The dataset includes patient health records and sensitive demographic attributes. Keep the downloaded CSV on your machine; it is intentionally excluded from this repository. UCI lists the dataset under CC BY 4.0. Cite Clore, Cios, DeShazo, and Strack (2014), DOI: [10.24432/C5230J](https://doi.org/10.24432/C5230J).

## Setup

Download the dataset from UCI, extract `diabetic_data.csv`, and place it in this project folder. Then create an environment and install the dependencies.

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### macOS or Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train and run

From the project folder, train the model and start the web app:

```sh
python train.py
python app.py
```

Open <http://127.0.0.1:5000/>. The model is saved under `models/` and is ignored by Git because it is a large generated artifact. Run `python train.py` again on another machine to create its local model.

The app also exposes `GET /health`, `GET /api/metrics`, and `POST /api/predict`.
