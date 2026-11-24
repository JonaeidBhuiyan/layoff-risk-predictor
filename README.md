# Workforce Intelligence — Layoff Risk Predictor

A polished frontend for the supplied FastAPI + scikit-learn preprocessor + XGBoost Booster backend.

## Project structure

```text
Layoff-Risk-Project/
├── main.py
├── preprocessor.pkl
├── xgb_model.json
└── static/
    ├── index.html
    ├── style.css
    └── app.js
```

Copy the three files from `static/` into your existing project's `static/` folder. Keep your trained model files beside `main.py`.

## Run locally on Windows

Open PowerShell in the project folder, activate your virtual environment, and run:

```powershell
python -m uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/
```

Do not open `index.html` directly and do not use the VS Code Live Server port for this same-origin setup. FastAPI serves the page, CSS, JavaScript, and API endpoints together.

## API integration

The frontend uses these same-origin endpoints:

- `GET /api/health`
- `GET /api/options`
- `POST /api/predict`

`API_BASE_URL` in `static/app.js` is intentionally an empty string. This is correct when FastAPI serves the frontend at `/`.

## Included features

- Dynamic dropdowns populated from the fitted encoder through `/api/options`
- 20 user-facing inputs matching the `CompanyData` schema in `main.py`
- Three illustrative scenario presets
- Synchronized range sliders and exact numeric inputs
- Optional fields sent as `null` for training-median imputation in the backend
- Prediction score gauge and application-defined risk band display
- What-if comparison list from the backend response
- Locally stored recent history, CSV export, copy summary, and print report
- Responsive desktop, tablet, and mobile layout

## Important model interpretation

The backend returns a regression score on a 0–100 scale. It is not a calibrated probability of layoffs. Risk bands are defined by the application, and what-if comparisons describe model sensitivity rather than causal effects.
