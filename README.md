# Predictive Maintenance Using Machine Learning

## Project Overview
An end-to-end machine learning project using engine data to develop and evaluate predictive maintenance models.

## Objectives
- Validate and explore engine data.
- Prepare training and testing datasets.
- Compare models and tune hyperparameters.
- Evaluate model performance.
- Provide predictions through a Streamlit application.
- Automate the workflow using GitHub Actions.

## Project Structure
- `data/`: Input dataset.
- `src/`: Validation, preparation, training, and evaluation scripts.
- `models/`: Trained model.
- `artifacts/`: Metrics and model comparison results.
- `reports/`: Evaluation figures and reports.
- `app.py`: Streamlit application.
- `requirements.txt`: Python dependencies.
- `.github/workflows/`: Automated pipeline.

## Run Locally
Install dependencies and launch the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer
For educational purposes. Predictions should not be the sole basis for real-world maintenance decisions.
